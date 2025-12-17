# gencast_fp/predict/predict.py
import os
import dataclasses
import logging
import numpy as np
import xarray
import haiku as hk
import jax

from typing import Optional

from graphcast import autoregressive
from graphcast import casting
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import normalization
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import xarray_tree
from graphcast import gencast
from graphcast import denoiser
from graphcast import nan_cleaning


def _parse_file_parts(file_name: str):
    return dict(part.split("-", 1) for part in file_name.split("_"))


def _construct_wrapped_graphcast(
    model_config,
    task_config,
    diffs_stddev_by_level,
    mean_by_level,
    stddev_by_level,
    min_by_level,
):
    predictor = gencast.GraphCast(
        model_config=model_config,
        task_config=task_config,
    )

    # Modify inputs/outputs to `FMGraphCast.GraphCast` to handle conversion to
    # from/to float32 to/from BFloat16.
    predictor = casting.Bfloat16Cast(predictor)

    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )

    # Wraps everything so the one-step model can produce trajectories.
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    
    return predictor

def run_predict(
    date: str,
    input_dir: str,
    out_dir: str,
    ckpt_path: Optional[str] = None,
    res_value: float = 0.25,
    nsteps: int = 40,                # 10-day rollout (6h steps)
) -> str:
    """
    Run GraphCast prediction for a single date.

    Returns the path to the written NetCDF file.
    """
    logging.info("Starting prediction")
    logging.info(f"date={date}, input_dir={input_dir}, out_dir={out_dir}, "
                 f"res={res_value}, nsteps={nsteps}")

    # Resolve dataset paths
    dataset_file_value = (
        f"graphcast-dataset-source-era5_date-{date}"
        f"_res-{res_value}_levels-13_steps-{nsteps}.nc"
    )
    dataset_file = os.path.join(input_dir, dataset_file_value)
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Input file not found: {dataset_file}")
    os.makedirs(out_dir, exist_ok=True)
    out_file_value = (
        f"graphcast-dataset-prediction-era5_date-{date}"
        f"_res-{res_value}_levels-13_steps-{nsteps}.nc"
    )
    out_file = os.path.join(out_dir, out_file_value)

    # Resolve checkpoints and stats
    # Default to repo-relative checkpoints if ckpt_path not given.
    if ckpt_path is None:
        # Expect this file structure:
        #   gencast_fp/predict/predict.py (this file)
        #   ../../checkpoints/gencast/gencast-params-GenCast_1p0deg_Mini_<2019.npz
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = os.path.join(
            script_dir,
            "../../checkpoints/graphcast/params_GraphCast_operational.npz",
        )
    diffs_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../checkpoints/graphcast/stats-diffs_stddev_by_level.nc",
    )
    mean_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../checkpoints/graphcast/stats-mean_by_level.nc",
    )
    stddev_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../checkpoints/graphcast/stats-stddev_by_level.nc",
    )
    min_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../checkpoints/graphcast/stats-min_by_level.nc",
    )

    logging.info(f"Checkpoint: {ckpt_path}")
    logging.info("Loading checkpoint and configs...")
    with open(ckpt_path, "rb") as f:
        ckpt = checkpoint.load(f, gencast.CheckPoint)
    params = ckpt.params
    state = {}
    task_config = ckpt.task_config
    model_config = ckpt.model_config

    logging.info("Loading dataset and normalization stats...")
    with open(dataset_file, "rb") as f:
        example_batch = xarray.load_dataset(f).compute()
    assert example_batch.dims["time"] >= 3

    with open(diffs_file, "rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with open(mean_file, "rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with open(stddev_file, "rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()
    with open(min_file, "rb") as f:
        min_by_level = xarray.load_dataset(f).compute()

    # Extract train/eval tensors
    train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("6h", "6h"),  # 1-AR training slice
        **dataclasses.asdict(task_config),
    )
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("6h", f"{nsteps*6}h"),
        **dataclasses.asdict(task_config),
    )

    logging.info(f"Devices available: {len(jax.local_devices())}")

    def _forward_wrapped():
        predictor = _construct_wrapped_graphcast(
            model_config,
            task_config,
            diffs_stddev_by_level,
            mean_by_level,
            stddev_by_level,
            min_by_level,
        )
        return predictor

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = _forward_wrapped()
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    @hk.transform_with_state
    def loss_fn(inputs, targets, forcings):
        predictor = _forward_wrapped()
        loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics),
        )

    if params is None:
        init_jitted = jax.jit(loss_fn.init)
        params, state = init_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=train_inputs,
            targets=train_targets,
            forcings=train_forcings,
        )

    run_forward_jitted = jax.jit(
        lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
    )
    run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")

    # Rollout
    rng = jax.random.PRNGKey(0)
    #rngs = np.stack([jax.random.fold_in(rng, i) for i in range(ensemble_members)], axis=0)

    logging.info("Starting autoregressive rollout...")

    predictions=rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=run_forward_pmap,
        rngs=rng,
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
    )
    predictions.to_netcdf(out_file)

    logging.info(f"Prediction written: {out_file}")
    return out_file


if __name__ == "__main__":
    # Optional standalone CLI for this module
    import argparse
    parser = argparse.ArgumentParser(description="GenCast Mini Prediction")
    parser.add_argument("--date", "-s", type=str, required=True)
    parser.add_argument("--input_dir", "-i", type=str, required=True)
    parser.add_argument("--out_dir", "-o", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--nsteps", type=int, default=30)
    parser.add_argument("--res", type=float, default=1.0)
    parser.add_argument("--ensemble", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_predict(
        date=args.date,
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        ckpt_path=args.ckpt,
        res_value=args.res,
        nsteps=args.nsteps,
        ensemble_members=args.ensemble,
    )

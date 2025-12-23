# gencast_fp/predict/predict.py
import os
import functools
import dataclasses
import logging
import numpy as np
import pandas as pd
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
from graphcast import graphcast


def _parse_file_parts(file_name: str):
    return dict(part.split("-", 1) for part in file_name.split("_"))

def xarray_load_ds(f):
    return xarray.load_dataset(f).compute()

def load_dataset(input_dir, date, res_value, nsteps):

    # Normalize whatever we get (str, np.datetime64, Timestamp) to a Timestamp
    if not isinstance(date, pd.Timestamp):
        date = pd.to_datetime(date)

    # Match the filename convention: YYYY-MM-DDTHH
    date_str = date.strftime("%Y-%m-%dT%H")

    # Resolve dataset paths
    input_file_value = (
        f"graphcast-dataset-source-era5_date-{date_str}"
        f"_res-{res_value}_levels-13_steps-{nsteps}.nc"
    )
    input_file = os.path.join(input_dir, input_file_value)
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    return input_file

def prepare_out_dir(out_dir, date, res_value, nsteps):

    # Normalize whatever we get (str, np.datetime64, Timestamp) to a Timestamp
    if not isinstance(date, pd.Timestamp):
        date = pd.to_datetime(date)

    os.makedirs(out_dir, exist_ok=True)

    # Match the filename convention: YYYY-MM-DDTHH
    date_str = date.strftime("%Y-%m-%dT%H")

    out_file_value = (
        f"graphcast-dataset-prediction-era5_date-{date_str}"
        f"_res-{res_value}_levels-13_steps-{nsteps}.nc"
    )
    out_file = os.path.join(out_dir, out_file_value)
    return out_file

def load_ckpt_files(container_meta, ckpt_path: str = None):
    if ckpt_path is None:
        # Expect this file structure:
        #   gencast_fp/predict/predict.py (this file)
        #   ../../checkpoints/gencast/gencast-params-GenCast_1p0deg_Mini_<2019.npz
        ckpt_path = os.path.join(
            container_meta, "params_GraphCast_operational.npz"
        )
    diffs_file = os.path.join(
        container_meta, "stats_diffs_stddev_by_level.nc"
    )
    mean_file = os.path.join(container_meta, "stats_mean_by_level.nc")
    stddev_file = os.path.join(
        container_meta, "stats_stddev_by_level.nc"
    )

    with open(ckpt_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)

    ckpt_and_stats = {
        "ckpt": ckpt,
        "diffs": xarray_load_ds(diffs_file),
        "mean": xarray_load_ds(mean_file),
        "stddev": xarray_load_ds(stddev_file),
    }

    return ckpt_and_stats

def _construct_wrapped_graphcast(
    model_config,
    task_config,
    diffs_stddev_by_level,
    mean_by_level,
    stddev_by_level,
):
    predictor = graphcast.GraphCast(model_config,task_config)

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
    ckpt_path: str = None,
    res_value: float = 0.25,
    nsteps: int = 40,                # 10-day rollout (6hr steps)
    container_meta: str = "/discover/nobackup/jli30/development/GraphCast_DSG/qefm-core/graphcast",
    ckpt_and_stats: dict = None,
) -> str:
    """
    Run GraphCast prediction for a single date.

    Returns the path to the written NetCDF file.
    """
    logging.info("Starting prediction")
    logging.info(f"date={date}, input_dir={input_dir}, out_dir={out_dir}, "
                 f"res={res_value}, nsteps={nsteps}")

    input_file = load_dataset(input_dir, date, res_value, nsteps)
    out_file = prepare_out_dir(out_dir, date, res_value, nsteps)


    # Skip if file already exists
    if os.path.exists(out_file):
        logging.info(f'Skipping {out_file}, prediction already exists.')
        return out_file

    # Load checkpoint and stats
    if ckpt_and_stats is None:
        ckpt_and_stats = load_ckpt_files(container_meta, ckpt_path)

    # Extract model info and task info from checkpoint
    ckpt = ckpt_and_stats["ckpt"]
    params = ckpt.params
    state = {}
    task_config = ckpt.task_config
    model_config = ckpt.model_config

    # Load example batch to create train/eval data properly
    with open(input_file, "rb") as f:
        example_batch = xarray.load_dataset(f).compute()
    assert example_batch.dims["time"] >= 3     

    # Extract train/eval tensors
    train_inputs, train_targets, train_forcings = (
        data_utils.extract_inputs_targets_forcings(
            example_batch,
            target_lead_times=slice("6h", "6h"),  # 1-AR training slice
            **dataclasses.asdict(task_config),
        )
    )
    eval_inputs, eval_targets, eval_forcings = (
        data_utils.extract_inputs_targets_forcings(
            example_batch,
            target_lead_times=slice(
                "6h", f"{(example_batch.dims['time']-2)*6}h"
            ),
            **dataclasses.asdict(task_config),
        )
    )

    logging.info(f"Devices available: {len(jax.local_devices())}")

    def _forward_wrapped():
        predictor = _construct_wrapped_graphcast(
            model_config,
            task_config,
            ckpt_and_stats["diffs"],
            ckpt_and_stats["mean"],
            ckpt_and_stats["stddev"],
        )
        return predictor
    
    # Jax doesn't seem to like passing configs as args through the jit. Passing it
    # in via partial (instead of capture by closure) forces jax to invalidate the
    # jit cache if you change configs.
    def with_configs(fn):
        return functools.partial(
        fn, model_config=model_config, task_config=task_config)

    # Always pass params and state, so the usage below are simpler
    def with_params(fn):
        return functools.partial(fn, params=params, state=state)
    
    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]
    
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
    init_jitted = jax.jit(with_configs(run_forward.init))
    
    if params is None:
        init_jitted = jax.jit(loss_fn.init)
        params, state = init_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=train_inputs,
            targets=train_targets,
            forcings=train_forcings,
        )

    run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
    run_forward.apply))))

    # Rollout
    rng = jax.random.PRNGKey(0)
    #rngs = np.stack([jax.random.fold_in(rng, i) for i in range(ensemble_members)], axis=0)

    logging.info("Starting autoregressive rollout...")

    predictions=rollout.chunked_prediction(
        predictor_fn=run_forward_jitted,
        rng=rng,
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
    )
    predictions.to_netcdf(out_file)

    logging.info(f"Prediction written: {out_file}")
    return out_file

def run_predict_multiday(
    start_date: str,
    end_date: str,
    input_dir: str,
    out_dir: str,
    res_value: float = 0.25,
    nsteps: int = 40,  # 10-day rollout (6h steps)
    container_meta: str = "/opt/qefm-core/gencast",
    ckpt_and_stats: dict = None,
):
    """Predict multiple days' worth of rollouts.
    Calls run_predict for each day in the start_date and end_date range."""
    #
    # start_date = np.datetime64(start_date)
    # end_date = np.datetime64(end_date)
    # date_range = np.arange(
    #     start_date, end_date + np.timedelta64(1, "D"), dtype="datetime64[D]"
    # )
    fmt = "%Y-%m-%d:%H"

    # Parse exact hour from input
    start_ts = pd.to_datetime(start_date, format=fmt)
    end_ts = pd.to_datetime(end_date,   format=fmt)

    # Generate a date range in 12-hour increments
    date_range = pd.date_range(start=start_ts, end=end_ts, freq="6h")

    for current_date in date_range:

        logging.info("======================================================")
        logging.info(f"Running prediction on date: {current_date}")
        out_fn = run_predict(
            current_date,
            input_dir,
            out_dir,
            res_value = res_value,
            nsteps = nsteps,
            container_meta = container_meta,
            ckpt_and_stats = ckpt_and_stats,
        )
        logging.info(f"Prediction saved to file: {out_fn}")
        logging.info("======================================================")
    return out_dir

if __name__ == "__main__":
    # Optional standalone CLI for this module
    import argparse
    parser = argparse.ArgumentParser(description="GraphCast Operational Prediction")
    parser.add_argument("--date", "-s", type=str, required=True)
    parser.add_argument("--input_dir", "-i", type=str, required=True)
    parser.add_argument("--out_dir", "-o", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--nsteps", type=int, default=40)
    parser.add_argument("--res", type=float, default=0.25)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_predict(
        date=args.date,
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        ckpt_path=args.ckpt,
        res_value=args.res,
        nsteps=args.nsteps,
    )

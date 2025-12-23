# GraphCast end-to-end workflow

[![DOI](https://zenodo.org/badge/1042752062.svg)](https://doi.org/10.5281/zenodo.17662736)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://github.com/nasa-nccs-hpda/GraphCast_DSG/README.md)
[![Build Docker](https://github.com/nasa-nccs-hpda/GraphCast_DSG/actions/workflows/dockerhub.yml/badge.svg?event=release)](https://github.com/nasa-nccs-hpda/GraphCast_DSG/actions/workflows/dockerhub.yml)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/nasa-nccs-hpda/GraphCast_DSG)
![Docker Image Version](https://img.shields.io/docker/v/nasanccs/graphcast-dsg?label=Docker)
![License](https://img.shields.io/github/license/nasa-nccs-hpda/GraphCast_DSG)

This workflow is to generate GraphCast predictions with ERA5 as inputs.
Follow the steps below to set up and run. The workflow currently only works on DISCOVER filesystems.

## Quickstart

The workflow runs preprocessing, prediction, and postprocessing for a given date
range using the Discover systems. Due to the system configuration, you will need access 
to a datamove node for preprocessing, which retrieving EAR5 data from Google Storge public dataset, 
and a single GPU A100 to run prediction and postprocessing.

Note that the following command can be run from any Discover login node.

For a single day (end_date defaults to the same day):

Step-1: preprocessing
```bash
sbatch --partition=datamove --mem=200G -t 1:00:00 -J preprocess-gc --wrap="module load singularity; singularity exec -B $NOBACKUP,/css,/gpfsm/dmd/css,/nfs3m,/gpfsm /discover/nobackup/projects/QEFM/containers/graphcast-dsg-containers/graphcast-dsg-latest graphcast-dsg preprocess --start_date 2025-12-19:12 --output_dir /discover/nobackup/jli30/development/GraphCast_DSG/tests/graphcast-run"
```

Step-2: prediction & postprocessing
```bash
sbatch --partition=gpu_a100 --constraint=rome --ntasks=10 --gres=gpu:1 --mem-per-gpu=100G -t 1:00:00 -J graphcast --wrap="module load singularity; singularity exec --nv -B $NOBACKUP,/css,/gpfsm/dmd/css,/nfs3m,/gpfsm /discover/nobackup/projects/QEFM/containers/graphcast-dsg-containers/graphcast-dsg-latest graphcast-dsg run --start_date 2025-12-19:12 --output_dir /discover/nobackup/jli30/development/GraphCast_DSG/tests/graphcast-run"
```

if you want to run for multiple past days, add arguments "--end_date yyyy-mm-dd:hh" to above commands


Example slurm file submission script for preprocessing:

```bash
#!/bin/bash
#SBATCH --partition=datamove
#SBATCH --ntasks=10
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --job-name=preprocess-gc
#SBATCH --output=preprocess_%j.out
#SBATCH --error=preprocess__%j.err


# Load modules
source /usr/share/modules/init/bash
module purge
module load singularity

# Run the container
singularity exec \
    -B $NOBACKUP,/css,/gpfsm/dmd/css,/nfs3m,/gpfsm \
    /discover/nobackup/projects/QEFM/containers/graphcast-dsg-containers/graphcast-dsg-latest \
    graphcast-dsg preprocess \
    --start_date 2025-12-19:12 \
    --output_dir /discover/nobackup/jli30/development/GraphCast_DSG/tests/graphcast-run
```

Example slurm file submission script for prediction & postprocessing:

```bash
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --ntasks=10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=100G
#SBATCH --time=10:00:00
#SBATCH --job-name=graphcast
#SBATCH --output=graphcast_%j.out
#SBATCH --error=graphcast_%j.err


# Load modules
source /usr/share/modules/init/bash
module purge
module load singularity

# Run the container
singularity exec --nv \
    -B $NOBACKUP,/css,/gpfsm/dmd/css,/nfs3m,/gpfsm \
    /discover/nobackup/projects/QEFM/containers/graphcast-dsg-containers/graphcast-dsg-latest \
    graphcast-fp run \
    --start_date 2025-12-19:12 \
    --output_dir /discover/nobackup/jli30/development/GraphCast_DSG/tests/graphcast-run
```

## Making your own changes

If you want to do additional development not available in the container package,
we recommend for you to Fork this repository and point to the new changes as a
PYTHONPATH variable inside the container.

Assuming you clone the sofware to Discover in the path `/discover/nobackup/myusername/GraphCast_DSG`,
you would need to change your container argument to use the `--env` as:

```bash
sbatch --partition=gpu_a100 --constraint=rome --ntasks=10 --gres=gpu:1 --mem-per-gpu=100G -t 10:00:00 -J graphcast --wrap="module load singularity; singularity exec --nv -B $NOBACKUP,/css,/gpfsm/dmd/css,/nfs3m,/gpfsm --env PYTHONPATH="/discover/nobackup/myusername/GraphCast_DSG" /discover/nobackup/projects/QEFM/containers/graphcast-dsg-containers/graphcast-dsg-latest graphcast-dsg run --start_date 2025-12-19:12 --output_dir /discover/nobackup/jli30/development/GraphCast_DSG/tests/graphcast-run"
```

## Dependencies

Additional details and flexibility of the commands are listed below.
A container has been made available here:

```bash
/discover/nobackup/projects/QEFM/containers/graphcast-dsg-containers/graphcast-dsg-latest
```

### Downloading the Container

If you would like to download the container yourself, you will need to run the following
command. The latest version has the most up to date changes, while specific releases are
attached to a given version from the repository.

#### Latest Release

```bash
singularity build --sandbox graphcast-dsg-latest docker://nasanccs/graphcast-dsg:latest
```

#### Specific Version

```bash
singularity build --sandbox graphcast-dsg-0.2.0 docker://nasanccs/graphcast-dsg:0.2.0
```

A version of this container is located at:

```bash
/discover/nobackup/projects/QEFM/containers/graphcast-dsg-containers/graphcast-dsg-latest
```

## Pipeline Details

In addition, individual steps of the pipeline can be run using the container and CLI. Some examples with arguments
are listed below. The pipeline has 3 steps: preprocess, predict, and postprocess. While we advice
to run the full pipeline, sometimes is easier to develop in stages.

### Preprocessing

```bash
usage: graphcast_dsg_cli.py preprocess [-h] --start_date START_DATE [--end_date END_DATE] [--output_dir OUTPUT_DIR]  [--res_value RES_VALUE] [--nsteps NSTEPS]

options:
  -h, --help            show this help message and exit
  --start_date START_DATE
                        Start date to process (YYYY-MM-DD:HH)
  --end_date END_DATE   End date to process (YYYY-MM-DD:HH)
  --output_dir OUTPUT_DIR
                        Output directory for preprocessed files
  --res_value RES_VALUE
                        Resoluton (default 1.0 resolution)
  --nsteps NSTEPS       Number of steps for rollout (default 30, 15 days)
```

### Prediction

```bash
usage: graphcast_dsg_clili.py predict [-h] --start_date START_DATE --end_date END_DATE --input_dir INPUT_DIR --output_dir OUTPUT_DIR [--ckpt CKPT] [--nsteps NSTEPS] [--res RES] [--container_meta CONTAINER_META]

options:
  -h, --help            show this help message and exit
  --start_date START_DATE
                        YYYY-MM-DD
  --end_date END_DATE   YYYY-MM-DD
  --input_dir INPUT_DIR, -i INPUT_DIR
                        Preprocessed input directory
  --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                        Where to write predictions
  --ckpt CKPT           Path to GraphCast .npz checkpoint (overrides container default)
  --nsteps NSTEPS
  --res RES
  --container_meta CONTAINER_META
                        Where to load default ckpt/configs if --ckpt not passed
```

### Postprocessing

```bash
usage: graphcast_dsg_cli.py postprocess [-h] --start_date START_DATE --end_date END_DATE --input_dir INPUT_DIR --predictions_dir PREDICTIONS_DIR [--output_dir OUTPUT_DIR]

options:
  -h, --help            show this help message and exit
  --start_date START_DATE
                        Start date (YYYY-MM-DD)
  --end_date END_DATE   End date (YYYY-MM-DD)
  --input_dir INPUT_DIR
                        Directory with GEOS inputs (for initial conditions)
  --predictions_dir PREDICTIONS_DIR
                        Directory with GraphCast predictions
  --output_dir OUTPUT_DIR
                        Directory for CF-compliant NetCDF outputs
```

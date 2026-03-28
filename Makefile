SHELL := /bin/bash
ROOT  := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
DATE  := $(shell date '+%Y%m%d-%H%M%S')

RSYNC               = rsync --archive --verbose --compress --rsh='ssh -o ClearAllForwardings=yes'

REMOTE_HOST        ?= pp-flowers
REMOTE_PATH        ?= projects/flowers

CONDA_ENV_NAME      = flowers

export PYTHONDONTWRITEBYTECODE = 1
export PYTHONUNBUFFERED        = 1
export PYTHONPATH              = $(ROOT)/src

export PIP_DEFAULT_TIMEOUT     = 300
export POETRY_REQUESTS_TIMEOUT = 300

export HF_HUB_ETAG_TIMEOUT     = 300
export HF_DATASETS_OFFLINE     = 0

# -----------------------------------------------------------------------------
# default
# -----------------------------------------------------------------------------

.DEFAULT_GOAL = env-shell

# -----------------------------------------------------------------------------
# conda and linux install and configuration for a new machine
# -----------------------------------------------------------------------------

.PHONY: conda-install
conda-install:
	@wget -qc -O '${HOME}/miniconda.sh' 'https://repo.anaconda.com/miniconda/Miniconda3-py312_25.9.1-3-Linux-x86_64.sh'
	@mkdir -p "${HOME}/opt"
	@bash '${HOME}/miniconda.sh' -b -f -p "${HOME}/opt/miniconda"
	@mkdir -p "${HOME}/.local/bin"
	@ln -sfT "${HOME}/opt/miniconda/bin/conda" "${HOME}/.local/bin/conda"
	@rm -vf '${HOME}/miniconda.sh'

.PHONY: conda-setup
conda-setup:
	@conda config --system --set solver libmamba
	@conda tos accept --override-channels --channel 'https://repo.anaconda.com/pkgs/main'
	@conda tos accept --override-channels --channel 'https://repo.anaconda.com/pkgs/r'
	@conda config --system --remove channels defaults
	@conda config --system --add channels conda-forge
	@conda config --system --add channels nvidia
	@conda config --show-sources
	@conda config --show channels

# -----------------------------------------------------------------------------
# conda environment
# -----------------------------------------------------------------------------

.PHONY: env-init-conda
env-init-conda:
	@conda create --yes --copy --name "$(CONDA_ENV_NAME)" \
		conda-forge::python=3.12.12 \
		conda-forge::poetry=2.2.1

.PHONY: env-init-poetry
env-init-poetry:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		poetry install --no-root --no-directory

.PHONY: env-init-xla
env-init-xla:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		pip3 install \
			--find-links https://storage.googleapis.com/libtpu-releases/index.html \
		  	--find-links https://storage.googleapis.com/libtpu-wheels/index.html \
		  	--no-cache-dir \
		   	'torch_xla[tpu]==2.9'

.PHONY: env-update
env-update:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		poetry update

.PHONY: env-list
env-list:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		poetry show --tree

.PHONY: env-remove
env-remove:
	@conda env remove --yes --name "$(CONDA_ENV_NAME)"

.PHONY: env-shell
env-shell:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		bash

.PHONY: env-python
env-python:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		python3

.PHONY: env-info
env-info:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		conda info

# -----------------------------------------------------------------------------
# training: PyTorch DDP
# -----------------------------------------------------------------------------

.PHONY: train
train: export OMP_NUM_THREADS=1
train: export PYTHONOPTIMIZE=0
train: export CUDA_VISIBLE_DEVICES=0
train:
	@bin/run python3 src/trainer.py train | tee "$(ROOT)/work/run-trainer-$(DATE).log"

.PHONY: train-reset
train-reset: clean-work train

# -----------------------------------------------------------------------------
# test
# -----------------------------------------------------------------------------

.PHONY: tests
tests:
	@bin/run pytest -v -p no:cacheprovider "$(ROOT)/tst/"

# -----------------------------------------------------------------------------
# tensorboard
# -----------------------------------------------------------------------------

.PHONY: tensorboard
tensorboard:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		tensorboard \
			--logdir "$(ROOT)/work/tensorboard/" \
			--load_fast false \
			--host "127.0.0.1" \
			--port "38001"

# -----------------------------------------------------------------------------
# system
# -----------------------------------------------------------------------------

.PHONY: vmstat
vmstat:
	@vmstat --unit M --timestamp --wide 3 | tee "$(ROOT)/work/vmstat-$(DATE).log"

.PHONY: gpustat
gpustat:
	@nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv --loop=5

# -----------------------------------------------------------------------------
# cleanup
# -----------------------------------------------------------------------------

.PHONY: clean-pycache
clean-pycache:
	@find "$(ROOT)/src" -type d -name '__pycache__' -print0 | xargs -0 -r -n 1 rm --recursive --verbose
	@find "$(ROOT)/tst" -type d -name '__pycache__' -print0 | xargs -0 -r -n 1 rm --recursive --verbose

.PHONY: clean-work-logs
clean-work-logs:
	@find "$(ROOT)/work" -type f -name '*.log' -print0 | xargs -0 -r -n 1 rm --recursive --verbose

.PHONY: clean-work-tensorboard
clean-work-tensorboard:
	@rm --recursive --verbose --force "$(ROOT)/work/tensorboard"

.PHONY: clean-work-snapshot
clean-work-snapshot:
	@rm --recursive --verbose --force "$(ROOT)/work/snapshot"

.PHONY: clean-work
clean-work: clean-work-logs clean-work-tensorboard clean-work-snapshot

.PHONY: clean
clean: clean-pycache clean-work

# -----------------------------------------------------------------------------
# rsync push
# -----------------------------------------------------------------------------

.PHONY: rsync-push
rsync-push:
	@$(RSYNC) \
		--exclude='/.git' \
		--exclude='/.idea' \
		--exclude='/cache/*.zip' \
		--exclude='/work/*' \
		--exclude='/data/*' \
		--exclude='*.log' \
		--exclude='__pycache__' \
		--exclude='.pytest_cache' \
		--exclude='.ipynb_checkpoints' \
		'$(ROOT)/' \
		'$(REMOTE_HOST):$(REMOTE_PATH)'

# -----------------------------------------------------------------------------
# rsync pull
# -----------------------------------------------------------------------------

.PHONY: rsync-pull
rsync-pull:
	@$(RSYNC) \
		--rsh="ssh -o ClearAllForwardings=yes" \
		--exclude='/.git' \
		--exclude='/.idea' \
		--exclude='/cache/*.zip' \
		--exclude='/work/*' \
		--exclude='/data/*' \
		--exclude='*.log' \
		--exclude='__pycache__' \
		--exclude='.pytest_cache' \
		--exclude='.ipynb_checkpoints' \
		'$(REMOTE_HOST):$(REMOTE_PATH)' \
		'$(ROOT)/'

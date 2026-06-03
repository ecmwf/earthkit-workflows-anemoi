# earthkit-workflows-anemoi

<p align="center">
  <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity">
    <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity/sandbox_badge.svg" alt="Static Badge">
  </a>

<a href="https://codecov.io/gh/ecmwf/earthkit-workflows-anemoi">
    <img src="https://codecov.io/gh/ecmwf/earthkit-workflows-anemoi/branch/develop/graph/badge.svg" alt="Code Coverage">
  </a>

<a href="https://opensource.org/licenses/apache-2-0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0">
  </a>

<a href="https://github.com/ecmwf/earthkit-workflows-anemoi/releases">
    <img src="https://img.shields.io/github/v/release/ecmwf/earthkit-workflows-anemoi?color=blue&label=Release&style=flat-square" alt="Latest Release">
  </a>
</p>

> \[!IMPORTANT\]
> This software is **Sandbox** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

Earthkit-Workflows-Anemoi is a Python library for connecting [anemoi-inference](https://github.com/ecmwf/anemoi-inference) to [earthkit-workflows](https://github.com/ecmwf/earthkit-workflows). Allowing for the inference tasks to be run as part of a larger DAG. It provides an API to directly create a graph consisting of initial condition retrieval and model execution, or to run inference off other source nodes which themselves are the initial conditions.

## Installation

The package has a split dependency structure to allow flexible installation:

### For workflow creation (minimal)

To create workflows without running them locally:

```bash
pip install earthkit-workflows-anemoi
```

This installs only the core dependencies needed to define and serialize workflows.

### For workflow execution (full runtime)

To both create and execute workflows locally:

```bash
pip install 'earthkit-workflows-anemoi[runtime]'
```

This includes `anemoi-inference` and `anemoi-datasets` required for local execution.

### For development

```bash
git clone https://github.com/ecmwf/earthkit-workflows-anemoi.git
cd earthkit-workflows-anemoi
pip install -e '.[dev]'
```

Additionally you may want to install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## Quick Start

To create a graph relying on anemoi-inference to get the initial conditions the following can be used:

```python

from earthkit.workflows.plugins import anemoi as anemoi_workflows

CKPT = {'huggingface': 'ecmwf/aifs-single-1.0'}

model_action = anemoi_workflows.fluent.from_input(CKPT, 'mars', '2022-01-01T00:00', lead_time = '7D', ensemble_members=51)
model_action

```

**Note:** For models with multiple datasets (supported since `anemoi-inference` 0.11.0), the resulting action includes a `dataset` dimension. Use `.select({"dataset": "era5"})` to select a specific dataset if needed.

Given other nodes as the initial conditions:

```python

from earthkit.workflows.plugins import anemoi as anemoi_workflows
from earthkit.workflows import fluent

SOURCE_NODES: fluent.Action
CKPT = {'huggingface': 'ecmwf/aifs-single-1.0'}

SOURCE_NODES.anemoi.infer(CKPT, lead_time = '7D', ensemble_members = 51)

```

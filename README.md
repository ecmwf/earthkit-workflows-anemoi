# anemoi-cascade

<p align="center">
  <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity">
    <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity/sandbox_badge.svg" alt="Static Badge">
  </a>

<a href="https://codecov.io/gh/ecmwf/anemoi-cascade">
    <img src="https://codecov.io/gh/ecmwf/anemoi-cascade/branch/develop/graph/badge.svg" alt="Code Coverage">
  </a>

<a href="https://opensource.org/licenses/apache-2-0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0">
  </a>

<a href="https://github.com/ecmwf/anemoi-cascade/releases">
    <img src="https://img.shields.io/github/v/release/ecmwf/anemoi-cascade?color=blue&label=Release&style=flat-square" alt="Latest Release">
  </a>
</p>

> \[!IMPORTANT\]
> This software is **Sandbox** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

Anemoi-Cascade is a Python library for connecting [anemoi-inference](https://github.com/ecmwf/anemoi-inference) to [earthkit-workflows](https://github.com/ecmwf/earthkit-workflows). Allowing for the inference tasks to be run as part of a larger DAG. It provides an API to directly create a graph consisting of initial condition retrieval and model execution, or to run inference off other source nodes which themselves are the initial conditions.

## Installation

Install via `pip` with:

```bash
pip install 'anemoi-cascade[all]'
```

For development, you can use `pip install -e .` Additionally you may want to install pre-commit hooks via

```bash
pip install pre-commit
pre-commit install
```

## Quick Start

To create a graph relying on anemoi-inference to get the initial conditions the following can be used:

```python

import anemoi.cascade

CKPT = {'huggingface': 'ecmwf/aifs-single-1.0'}

model_action = anemoi.cascade.fluent.from_input(CKPT, 'mars', '2022-01-01T00:00', lead_time = '7D', ensemble_members=51)
model_action

```

Given other nodes as the initial conditions:

```python

import anemoi.cascade
from earthkit.workflows import fluent

SOURCE_NODES: fluent.Action
CKPT = {'huggingface': 'ecmwf/aifs-single-1.0'}

SOURCE_NODES.anemoi.infer(CKPT, lead_time = '7D', ensemble_members = 51)

```

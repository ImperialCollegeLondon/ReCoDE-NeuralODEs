# Neural Ordinary Differential Equations

## Description

This project will walk through solving Ordinary Differential Equations (ODEs)
within an autograd framework (PyTorch), utilising the inbuilt tools to effectively
differentiate the parameters and solutions of them, and finally incorporating
Neural Networks to demonstrate how to effectively learn dynamics from data.

## Learning Outcomes

- Writing a python module geared towards research that can be used by others
- How to take research/theoretical concepts and turn them into code
- How numerical integration works
- How neural networks work
- How neural networks and numerical integration can be combined

| Task                         | Time       |
|------------------------------|------------|
| Reading                      | 8 hours    |
| Running Notebooks            | 4-12 hours |
| Practising with Own Dynamics | 4+ hours   |

## Requirements

### Academic

- Knowledge of calculus, specifically in derivatives, integrals and limits.
- A rudimentary understanding of how floating-point/finite precision algebra works on computers.
- Basic python programming skills, knowledge of iteration, branching, etc.
- A bref understanding of vectorised computation. How CPUs/GPUs process different data in parallel

### System

- Python 3.10 or newer
- Poetry
- CUDA-capable GPU (for GPU training of networks)

## Getting Started

### Setting up Python Environment

1. Install Python 3.10 or above
2. Install `pipx` following the instructions here: <https://pipx.pypa.io/stable/installation/>
3. Install Poetry using the instructions here: <https://python-poetry.org/docs/#installing-with-pipx>
4. Once Poetry is set up and usable, go to the root directory of this repository and run `poetry lock` followed by `poetry install`. This should install the project dependencies into a Poetry managed virtual environment.
5. To run the code, use:
   1. `poetry run [SCRIPT NAME]` to run any script in the repository.
   2. `poetry shell` to enter a shell with the appropriate `python` and dependencies set up. From there you can use `python [SCRIPT NAME]` to run any script.
   3. `poetry run jupyter notebook` to start a jupyter notebook in the repository environment from which the notebooks can be run.
6. If using the code as a dependency (i.e. as a module that is imported in your own script), then you'll need to run `pip install .` which will install the locally available package into the current python environment.

### How to Use this Repository

1. Start by reading `Chapter 1 - Introduction to Ordinary Differential Equations (ODEs)` and refer to the introductory notebooks for the implementation of the concepts.
2. Study the jupyter notebooks on the implementations in further detail: [Fill with notebook names for introductory material]
3. Study Chapter 2 for a walk-through of the module structure
4. Study jupyter notebooks for training scripts as well as visualisation of results

## Project Structure

```log
.
├── neuralode
│   ├── integrators
│   ├── models
│   ├── plot
│   └── ...
├── notebooks
|   ├── 01-simple-integration-routines.ipynb
|   ├── 02-arbitrary-adaptive-tableaus.ipynb
|   ├── 03-the-adjoint-method.ipynb
|   ├── 04-driven-harmonic-oscillator.ipynb
|   └── 05-the-inverted-pendulum.ipynb
├── docs
|   └── 01-introduction.md
|   ...
```

## License

This project is licensed under the [BSD-3-Clause license](LICENSE.md)

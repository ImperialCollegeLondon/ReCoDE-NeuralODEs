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

## ---- Fill in this part later: ----
<!-- How long should they spend reading and practising using your Code.
Provide your best estimate -->

| Task       | Time    |
| ---------- |---------|
| Reading    | 4 hours |
| Practising | 8 hours |

## --------

## Requirements

<!--
If your exemplar requires students to have a background knowledge of something
especially this is the place to mention that.

List any resources you would recommend to get the students started.

If there is an existing exemplar in the ReCoDE repositories link to that.
-->

### Academic

- Knowledge of calculus, specifically in derivatives, integrals and limits.
- A rudimentary understanding of how floating-point/finite precision algebra works on computers.
- Basic python programming skills, knowledge of iteration, branching, etc.
- A bref understanding of vectorised computation. How CPUs/GPUs process different data in parallel

<!-- List the system requirements and how to obtain them, that can be as simple
as adding a hyperlink to as detailed as writting step-by-step instructions.
How detailed the instructions should be will vary on a case-by-case basis.

Here are some examples:

- 50 GB of disk space to hold Dataset X
- Anaconda
- Python 3.11 or newer
- Access to the HPC
- PETSc v3.16
- gfortran compiler
- Paraview
-->

### System

- Python 3.10 or newer
- Poetry
- CUDA-capable GPU (for GPU training of networks)

<!-- Instructions on how the student should start going through the exemplar.

Structure this section as you see fit but try to be clear, concise and accurate
when writing your instructions.

For example:
Start by watching the introduction video,
then study Jupyter notebooks 1-3 in the `intro` folder
and attempt to complete exercise 1a and 1b.

Once done, start going through through the PDF in the `main` folder.
By the end of it you should be able to solve exercises 2 to 4.

A final exercise can be found in the `final` folder.

Solutions to the above can be found in `solutions`.
-->

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

<!-- An overview of the files and folder in the exemplar.
Not all files and directories need to be listed, just the important
sections of your project, like the learning material, the code, the tests, etc.

A good starting point is using the command `tree` in a terminal(Unix),
copying its output and then removing the unimportant parts.

You can use ellipsis (...) to suggest that there are more files or folders
in a tree node.

-->

## Project Structure

```log
.
├── neuralode
│   ├── SUBMODULE 1
│   └── SUBMODULE 2
├── notebooks
|   ├── FILL WITH NOTEBOOK NAMES
├── docs
├── main
```

<!-- Change this to your License. Make sure you have added the file on GitHub -->

## License

This project is licensed under the [BSD-3-Clause license](LICENSE.md)

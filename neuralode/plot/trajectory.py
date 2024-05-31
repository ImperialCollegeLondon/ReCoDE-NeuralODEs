import typing
import matplotlib.pyplot as plt
import torch
import numpy


def plot_trajectory_with_reference(
    integration_states,
    ideal_solution_trajectory: list[
        typing.Union[torch.Tensor, numpy.array, float]
    ] = None,
    axes: plt.Axes = None,
    method_label: str = "Euler Method",
):
    plot_times = [i[1].cpu().item() for i in integration_states]
    plot_states = [i[0].cpu().item() for i in integration_states]

    if axes is None:
        fig = plt.figure(figsize=(12, 5), tight_layout=True)
        axes = []
    else:
        fig = axes[0].get_figure()

    if len(axes) < 2:
        ax = fig.add_subplot(211)
        axes.append(ax)
    else:
        ax = axes[0]
    ax.plot(
        plot_times,
        plot_states,
        marker=".",
        markersize=0.5,
        label=method_label,
        linestyle="--",
    )
    ax.set_ylabel(r"$x(t)$")
    ax.legend()
    if ideal_solution_trajectory is not None:
        ax.plot(plot_times, ideal_solution_trajectory, label="Ideal Solution")
        if len(axes) < 2:
            ax = fig.add_subplot(212, sharex=ax)
            axes.append(ax)
        else:
            ax = axes[1]
        ax.plot(
            plot_times,
            [abs(i - j).item() for i, j in zip(ideal_solution_trajectory, plot_states)],
            marker="x",
            markersize=0.5,
            label=f"{method_label} Residual",
        )
        ax.set_yscale("log")
        ax.set_xlabel(r"Time $(\mathrm{s})$")
        ax.set_ylabel(r"$\Delta x$")
        ax.legend()
    return fig, axes


def plot_trajectory(
    integration_states, axes: plt.Axes = None, method_label: str = "Euler Method"
):
    if axes is None:
        fig = plt.figure(figsize=(12, 5), tight_layout=True)
        axes = [fig.add_subplot(111)]
    return plot_trajectory_with_reference(
        integration_states,
        ideal_solution_trajectory=None,
        axes=axes,
        method_label=method_label,
    )

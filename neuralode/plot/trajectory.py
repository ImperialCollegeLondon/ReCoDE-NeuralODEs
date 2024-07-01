import typing
import matplotlib.pyplot as plt
import torch
import numpy


def plot_trajectory_with_reference(
    integration_states,
    reference_trajectory: list[typing.Union[torch.Tensor, numpy.array, float]] = None,
    axes: plt.Axes = None,
    method_label: str = "Euler Method",
):
    """
    Plots the trajectory specified by given (x,t) pairs where x can be multidimensional.

    :param integration_states: A list with tuples of (x, t) pairs; x can be a multidimensional tensor
    :param reference_trajectory: A list with reference values of x at each t specified in integration_states
    :param axes: Optional argument to pass in plotting axes
    :param method_label: Label for the integration states line plots
    :return:
    """
    plot_states = [i[0].detach().cpu() for i in integration_states]
    plot_times = [i[1].detach().cpu() for i in integration_states]
    if reference_trajectory is not None:
        ref_traj = [
            i.detach().cpu() if torch.is_tensor(i) else i for i in reference_trajectory
        ]

    if axes is None:
        fig = plt.figure(figsize=(12, 5), tight_layout=True)
        axes = []
    else:
        fig = axes[0].get_figure()

    if len(axes) < 2 and reference_trajectory is not None:
        ax = fig.add_subplot(211)
        axes.append(ax)
    else:
        ax = axes[0]
    # If we have multiple components, we plot all of them
    for cidx in range(plot_states[0].ravel().numel()):
        ax.plot(
            plot_times,
            [i.ravel()[cidx] for i in plot_states],
            marker=".",
            markersize=4.0,
            label=f"{method_label}:[{cidx}]" if method_label is not None else None,
            linestyle="--",
        )
    ax.set_ylabel(r"$\vec{x}(t)$")
    if reference_trajectory is not None:
        for cidx in range(ref_traj[0].ravel().numel()):
            ax.plot(
                plot_times,
                [i.ravel()[cidx] for i in ref_traj],
                label=f"Reference:[{cidx}]",
                alpha=0.5,
            )
        ax.legend()
        if len(axes) < 2:
            ax = fig.add_subplot(212, sharex=ax)
            axes.append(ax)
        else:
            ax = axes[1]
        ax.plot(
            plot_times,
            [torch.linalg.norm(i - j).item() for i, j in zip(ref_traj, plot_states)],
            marker="x",
            markersize=0.5,
            label=f"{method_label} Residual",
        )
        ax.set_yscale("log")
        ax.set_xlabel(r"Time $(\mathrm{s})$")
        ax.set_ylabel(r"$\Delta x$")
        ax.legend()
    else:
        ax.legend()
    return fig, axes


def plot_trajectory(
    integration_states, axes: plt.Axes = None, method_label: str = "Euler Method"
):
    """
    Convenience function for plotting trajectories without a reference.

    :param integration_states:
    :param axes:
    :param method_label:
    :return:
    """
    if axes is None:
        fig = plt.figure(figsize=(12, 5), tight_layout=True)
        axes = [fig.add_subplot(111)]
    return plot_trajectory_with_reference(
        integration_states,
        reference_trajectory=None,
        axes=axes,
        method_label=method_label,
    )


def plot_nbody(states):
    fig = plt.figure(figsize = (12, 12))
    ax = fig.add_subplot(111, projection = '3d')
    for body_idx in range(states.shape[1]):
        m_xyz = states[...,body_idx,:3]
        ax.plot(*m_xyz.permute(1, 0).detach().cpu().numpy())
    return fig, ax

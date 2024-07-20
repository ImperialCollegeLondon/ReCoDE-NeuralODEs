import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_pendulum(pendulum_x, pendulum_y, cart_x, cart_y, time, force):
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(211, aspect="equal")
    ax.scatter(pendulum_x[0].item(), pendulum_y[0].item(), marker="o")
    ax.scatter(pendulum_x[-1].item(), pendulum_y[-1].item(), marker="x")
    ax.plot(pendulum_x.cpu().numpy(), pendulum_y.cpu().numpy())
    ax.plot(cart_x.cpu().numpy(), cart_y.cpu().numpy())
    for cx, cy, wx, wy in zip(cart_x, cart_y, pendulum_x, pendulum_y):
        ax.plot(
            [cx.item(), wx.item()],
            [cy.item(), wy.item()],
            linewidth=0.25,
            linestyle="--",
            color="k",
        )
    ax.scatter(cart_x[0].item(), 0.0, marker="o")
    ax.scatter(cart_x[-1].item(), 0.0, marker="x")
    ax.axhline(cart_y.cpu().mean(), linewidth=0.5, color="k")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2, 2)
    ax = fig.add_subplot(212)
    ax.plot(time.cpu().numpy(), force.cpu().numpy())
    ax.set_ylabel(r"$F\ \left[\mathrm{N}\right]$")
    ax.set_xlabel(r"$t\ \left[\mathrm{s}\right]$")
    return fig


def animate_pendulum(
    pendulum_x,
    pendulum_y,
    cart_x,
    cart_y,
    system_times,
    forces=None,
    frame_time=1000 / 60,
):
    plt.ioff()
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")

    t_initial, t_final = system_times.min().item(), system_times.max().item()
    frame_times = np.linspace(
        t_initial, t_final, int(1000 * (t_final - t_initial) / frame_time + 0.5)
    )

    pendulum_x = np.interp(
        frame_times, system_times.cpu().numpy(), pendulum_x.cpu().numpy()
    )
    pendulum_y = np.interp(
        frame_times, system_times.cpu().numpy(), pendulum_y.cpu().numpy()
    )
    cart_x = np.interp(frame_times, system_times.cpu().numpy(), cart_x.cpu().numpy())
    cart_y = np.interp(frame_times, system_times.cpu().numpy(), cart_y.cpu().numpy())
    if forces is not None:
        forces = np.interp(
            frame_times, system_times.cpu().numpy(), forces.cpu().numpy()
        )

    (pole_plot,) = ax.plot(
        [cart_x[0], pendulum_x[0].item()],
        [cart_y[0], pendulum_y[0]],
        linewidth=0.25,
        linestyle="--",
        color="k",
    )
    (pendulum_head_plot,) = ax.plot(pendulum_x[0], pendulum_y[0], marker="o")
    (cart_plot,) = ax.plot(cart_x[0], cart_y[0], marker="o")
    if forces is not None:
        forces_plot = ax.arrow(
            cart_x[0],
            cart_y[0],
            forces[0],
            0.0,
            head_width=0.1,
            color="k",
            head_starts_at_zero=True,
        )
    ax.axhline(cart_y.mean(), linewidth=0.5, color="k")

    def animate(frame_index):
        pole_plot.set_data(
            [cart_x[frame_index], pendulum_x[frame_index]],
            [cart_y[frame_index], pendulum_y[frame_index].item()],
        )
        pendulum_head_plot.set_data(
            [pendulum_x[frame_index]], [pendulum_y[frame_index]]
        )
        cart_plot.set_data([cart_x[frame_index]], [cart_y[frame_index]])
        if forces is not None:
            forces_plot.set_data(
                x=cart_x[frame_index], y=cart_y[frame_index], dx=forces[frame_index]
            )

    ani = animation.FuncAnimation(
        fig, animate, frames=cart_x.shape[0], interval=frame_time
    )
    ax.set_xlim(
        min(-2.5, cart_x.min().item(), pendulum_x.min().item()),
        max(2.5, cart_x.max().item(), pendulum_x.max().item()),
    )
    ax.set_ylim(-2, 2)
    try:
        html_ani = ani.to_html5_video()
    except RuntimeError:
        html_ani = ani.to_jshtml()

    plt.close(fig)
    plt.ion()
    return html_ani

import torch

from neuralode.integrators import routines

__all__ = ["IntegrateRK4"]


class IntegrateRK4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, x0, t0, t1, dt):
        """
        A general integration routine for solving an Initial Value Problem
        using the Classical Fourth-Order Method.

        Instead of naively summing the changes, we use compensated summation.

        :param fn: the function to be integrated
        :param initial_state: the initial state to integrate from
        :param initial_time: the initial time to integrate from
        :param final_time: the final time to integrate to
        :param timestep: the time increments to integrate with
        :return: a tuple of ((the final state, the final time), the intermediates states [list])
        """
        # We write out the butcher tableau for easier computation
        butcher_tableau = torch.tensor(
            [
                # c0, a00, a01, a02, a03
                [0.0, 0.0, 0.0, 0.0, 0.0],
                # c1, a10, a11, a12, a13
                [0.5, 0.5, 0.0, 0.0, 0.0],
                # c2, a20, a21, a22, a23
                [0.5, 0.0, 0.5, 0.0, 0.0],
                # c3, a30, a31, a32, a33
                [1.0, 0.0, 0.0, 1.0, 0.0],
                #     b0,  b1,  b2,  b3
                [0.0, 1 / 6, 2 / 6, 2 / 6, 1 / 6],
            ],
            dtype=x0.dtype,
            device=x0.device,
        )

        # The names for the variables have been shortened for concision, and
        # to avoid overlap with variables in the outer scope
        # I have also left the annotations as they are invaluable for tracking the method.
        c_time = t0.clone()
        c_state = x0.clone()

        c_state, c_time, i_states = routines.integrate_system(
            fn, c_state, c_time, t1, dt, butcher_tableau
        )

        ctx.save_for_backward((c_state, c_time, i_states))
        ctx.integration_function = fn

        return (c_state, c_time), i_states

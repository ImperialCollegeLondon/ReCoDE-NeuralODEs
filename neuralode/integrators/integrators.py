import torch
import typing

from neuralode.integrators import classes
from neuralode.integrators import routines

__all__ = [
    "get_forward_method",
    "get_backward_method",
    "get_integrator",
    "IntegrateRK4",
]


def get_forward_method(integrator_type, use_local_extrapolation):
    def __internal_forward(
        fn: typing.Callable[[torch.Tensor, torch.Tensor, typing.Any], torch.Tensor],
        x0: torch.Tensor,
        t0: torch.Tensor,
        t1: torch.Tensor,
        dt: torch.Tensor,
        integrator_kwargs: dict[str, torch.Tensor],
        *additional_dynamic_args,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        A general integration routine for solving an Initial Value Problem
        using any arbitrary Butcher Tableau

        Instead of naively summing the changes, we use compensated summation.

        :param fn: the function to be integrated
        :param initial_state: the initial state to integrate from
        :param initial_time: the initial time to integrate from
        :param final_time: the final time to integrate to
        :param timestep: the time increments to integrate with
        :param atol: The absolute tolerance for the error in an adaptive integration
        :param rtol: The relative tolerance for the error in an adaptive integration
        :param additional_dynamic_args: additional arguments to pass to the function
        :return: a tuple of ((the final state, the final time), the intermediate states [list[torch.Tensor]], the error values [list[torch.Tensor]])
        """

        def forward_fn(state, time):
            return fn(state, time, *additional_dynamic_args)

        c_state = x0.clone()
        c_time = t0.clone()

        integrator_kwargs = {key: val.clone() for key, val in integrator_kwargs.items()}
        integrator_kwargs.update(
            {
                "tableau": integrator_type.integrator_tableau.clone().to(
                    x0.device, x0.dtype
                ),
                "is_adaptive": integrator_type.is_adaptive,
                "number_of_stages": integrator_type.number_of_stages,
                "integrator_order": integrator_type.integrator_order,
                "use_local_extrapolation": use_local_extrapolation,
            }
        )

        c_state, c_time, i_states, error_in_state = routines.integrate_system(
            forward_fn, c_state, c_time, t1, dt, integrator_kwargs
        )

        intermediate_states, intermediate_times = zip(*i_states)

        # As we said, these need to be converted to tensors for proper tracking
        intermediate_states = torch.stack(intermediate_states, dim=0)
        intermediate_times = torch.stack(intermediate_times, dim=0)

        # We should also put the errors we're returning into a tensor too
        error_in_state = torch.stack(error_in_state, dim=0)

        return (
            c_state,
            c_time,
            intermediate_states,
            intermediate_times,
            error_in_state.detach(),
        )

    return __internal_forward


def get_backward_method():
    def __internal_backward(
        ctx: torch.autograd.function.FunctionCtx,
        d_c_state: torch.Tensor,
        d_c_time: torch.Tensor,
        d_intermediate_states: torch.Tensor,
        d_intermediate_times: torch.Tensor,
        d_error_in_state: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        This function computes the gradient of the input variables for `__internal_forward` by exploiting the fact
        that PyTorch can track the whole graph of operations used to derive a specific result. Thus each time backward is called,
        we compute the actual graph of operations and propagate derivatives through it. Unfortunately, this is an exceptionally
        slow method of computation that also uses a lot of memory.

        This is implemented here as a demonstration of how we could compute gradients and how these are expected to be propagated back
        to the autograd tape.

        :param ctx:
        :param d_c_state:
        :param d_c_time:
        :param d_intermediate_states:
        :param d_intermediate_times:
        :param d_error_in_state:
        :return:
        """
        # First, we retrieve our integration function that we stored in `integration_function`
        fn = ctx.integration_function
        # Then we retrieve the input variables and clone them to avoid influencing them in the later operations
        x0, t0, t1, dt, atol, rtol, _, _, _, _, *additional_dynamic_args = [
            i.clone().requires_grad_(True) for i in ctx.saved_tensors
        ]
        inputs = fn, x0, t0, t1, dt, atol, rtol, *additional_dynamic_args

        def forward_fn(x, t):
            return fn(x, t, *additional_dynamic_args)

        if any(ctx.needs_input_grad):
            # We ensure that gradients are enabled so that autograd tracks the variable operations
            with torch.enable_grad():

                def forward_fn(state, time):
                    return fn(state, time, *additional_dynamic_args)

                c_state = x0.clone()
                c_time = t0.clone()

                c_state, c_time, _, _ = routines.integrate_system(
                    forward_fn, c_state, c_time, t1, dt, ctx.integrator_kwargs
                )

            # We collate the outputs that we can compute gradients for
            # with this method, we are restricted to the final state and time
            outputs = (
                c_state,
                c_time,
            )  # , intermediate_states, intermediate_times, error_in_state
            grad_outputs = (
                d_c_state,
                d_c_time,
                d_intermediate_states,
                d_intermediate_times,
                d_error_in_state,
            )

            # We also only consider the input and output variables that actually have gradients enabled
            inputs_with_grad = [
                i for idx, i in enumerate(inputs) if ctx.needs_input_grad[idx]
            ]
            outputs_with_grad = [
                idx for idx, i in enumerate(outputs) if i.grad_fn is not None
            ]

            grad_of_inputs_with_grad = torch.autograd.grad(
                [outputs[idx] for idx in outputs_with_grad],
                inputs_with_grad,
                grad_outputs=[grad_outputs[idx] for idx in outputs_with_grad],
                allow_unused=True,
                materialize_grads=True,
            )
        else:
            grad_of_inputs_with_grad = None
        # For each input we must return a gradient
        # We create a list of None values
        # (this tells autograd that there is no gradient for those variables).
        # And for each variable that does have a gradient, we fill the values in
        # before returning the list
        input_grads = [None for _ in range(len(inputs))]
        if grad_of_inputs_with_grad:
            for idx in range(len(inputs)):
                if ctx.needs_input_grad[idx]:
                    input_grads[idx], *grad_of_inputs_with_grad = (
                        grad_of_inputs_with_grad
                    )
        return tuple(input_grads)

    return __internal_backward


def get_integrator(
    integrator_tableau: torch.Tensor,
    integrator_order: int,
    use_local_extrapolation: bool = True,
    integrator_name: str = None,
) -> typing.Type[torch.autograd.Function]:
    __integrator_type = classes.create_integrator_class(
        integrator_tableau, integrator_order, use_local_extrapolation, integrator_name
    )

    # Forward integration method
    __internal_forward = get_forward_method(__integrator_type, use_local_extrapolation)
    # Backward integration method
    __internal_backward = get_backward_method()

    classes.finalise_integrator_class(
        __integrator_type, __internal_forward, __internal_backward
    )

    return __integrator_type


IntegrateRK4 = get_integrator(
    torch.tensor(
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
        dtype=torch.float64,
    ),
    integrator_order=4,
    integrator_name="IntegratorRK4",
)

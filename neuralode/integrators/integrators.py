from typing import Tuple, Any

import torch
import typing
import math

from neuralode.integrators import classes
from neuralode.integrators import routines
from neuralode.integrators import helpers
from neuralode.integrators import signatures

__all__ = [
    "get_forward_method",
    "get_backward_method",
    "get_vmap_method",
    "get_integrator",
    "MidpointIntegrator",
    "RK4Integrator",
    "AdaptiveRK45Integrator",
    "AdaptiveRK87Integrator",
]


def get_forward_method(
    integrator_type, use_local_extrapolation
) -> signatures.forward_method_signature:
    def __internal_forward(
        fn: typing.Callable[[torch.Tensor, torch.Tensor, typing.Any], torch.Tensor],
        x0: torch.Tensor,
        t0: torch.Tensor,
        t1: torch.Tensor,
        dt: torch.Tensor,
        integrator_kwargs: dict[str, torch.Tensor],
        *additional_dynamic_args: list[torch.Tensor],
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


def get_backward_method(integrator_type) -> signatures.backward_method_signature:
    def __internal_backward(
        ctx: torch.autograd.function.FunctionCtx,
        d_c_state: torch.Tensor,
        d_c_time: torch.Tensor,
        d_intermediate_states: torch.Tensor,
        d_intermediate_times: torch.Tensor,
        d_error_in_state: torch.Tensor,
    ) -> tuple[Any, ...]:
        """
        This function computes the gradient of the input variables for `__internal_forward` by implementing
        the adjoint method. This involves computing the adjoint state and adjoint state equation and systematically
        integrating backwards, from t1 to t0, accumulating the gradient to obtain the gradient wrt. the input variables.

        :param ctx: Function context for storing variables and function from the forward pass
        :param d_c_state: incoming gradient of c_state
        :param d_c_time: incoming gradient of c_time
        :param d_intermediate_states: incoming gradient of intermediate_states
        :param d_intermediate_times: incoming gradient of intermediate_times
        :param d_error_in_state: incoming gradient of error_in_state. This output is non-differentiable.
        :return: The gradients wrt. all the inputs.
        """

        # First, we retrieve our integration function that we stored in `__internal_forward`
        forward_fn: signatures.integration_fn_signature = ctx.forward_fn

        integrator_kwargs = ctx.integrator_kwargs

        # Then we retrieve the input variables
        (
            x0,
            t0,
            t1,
            dt,
            c_state,
            c_time,
            intermediate_times,
            *additional_dynamic_args,
        ) = ctx.saved_tensors

        inputs = forward_fn, x0, t0, t1, dt, integrator_kwargs, *additional_dynamic_args
        input_grads: list[torch.Tensor | None] = [None for _ in range(len(inputs))]

        if any(ctx.needs_input_grad):
            # Construct the adjoint equation
            adjoint_fn = helpers.construct_adjoint_fn(forward_fn, c_state.shape)
            
            # We ensure that gradients are enabled so that autograd tracks the variable operations
            # For pointwise functionals, the initial adjoint state is simply the incoming gradients
            parameter_shapes = [i.shape for i in additional_dynamic_args]
            packed_reverse_state = torch.cat(
                [
                    c_state.ravel(),
                    (d_c_state + d_intermediate_states[-1]).ravel(),
                ]
            )
            if len(additional_dynamic_args) > 0:
                packed_reverse_state = torch.cat(
                    [
                        packed_reverse_state,
                        torch.zeros(
                            sum(map(math.prod, parameter_shapes)),
                            device=c_state.device,
                            dtype=c_state.dtype,
                        ),
                    ]
                )

            current_adj_time = t1
            current_adj_state = packed_reverse_state

            if torch.any(d_intermediate_states != 0.0):
                adj_indices = torch.arange(
                    c_state.numel(), 2 * c_state.numel(), device=c_state.device
                )
                # We only need to account for the incoming gradients if any are non-zero
                for next_adj_time, d_inter_state in zip(
                    intermediate_times[1:-1].flip(dims=[0]),
                    d_intermediate_states[1:-1].flip(dims=[0]),
                ):
                    # The incoming gradients of the intermediate states are the gradients of the state defined at
                    # various points in time. For each of these incoming gradients, we need to integrate up to that
                    # temporal boundary and add them to adjoint state
                    if torch.all(d_inter_state == 0.0):
                        # No need to integrate up to the boundary if the incoming gradients are zero
                        continue
                    current_adj_state, current_adj_time, _, _, _ = integrator_type.apply(adjoint_fn,
                                                                                         current_adj_state,
                                                                                         current_adj_time,
                                                                                         next_adj_time, -dt,
                                                                                         integrator_kwargs,
                                                                                         *additional_dynamic_args)
                    current_adj_state = torch.scatter(current_adj_state, 0, adj_indices, d_inter_state.ravel())

            final_adj_state, final_adj_time, _, _, _ = integrator_type.apply(adjoint_fn, current_adj_state,
                                                                             current_adj_time, t0, -dt, integrator_kwargs,
                                                                             *additional_dynamic_args)

            # This should be equivalent to the initial state we passed in, but it will
            # be appropriately attached to the autograd graph for higher order derivatives
            if torch.is_grad_enabled() and any(i.requires_grad for i in [d_c_state, d_c_time, d_intermediate_states]):
                adj_initial_state = final_adj_state[:c_state.numel()].reshape(c_state.shape)
            else:
                adj_initial_state = x0.clone()
            adj_variables = final_adj_state[c_state.numel():2*c_state.numel()].reshape(c_state.shape)
            adj_parameter_gradients = final_adj_state[2*c_state.numel():]

            # The gradients of the incoming state are equal to the gradients from the first element of the
            # intermediate state plus the lagrange variables
            input_grads[1] = adj_variables.reshape(c_state.shape) + d_intermediate_states[0].ravel().reshape(c_state.shape)

            # The gradient of the initial time is equal to the gradient from the first element of the intermediate times
            # minus the product of the lagrange variables and the derivative of the system at the initial time
            input_grads[2] = (d_intermediate_times[0].ravel() - torch.sum(adj_variables.ravel()*forward_fn(adj_initial_state, final_adj_time, *additional_dynamic_args).ravel())).reshape(c_time.shape)
            # The gradient of the final time is equal to the gradient from the gradient in the final state
            # plus the product of the lagrange variables and the derivative of the system at the final time
            input_grads[3] = ((d_c_time + d_intermediate_times[-1]) + torch.sum((d_c_state + d_intermediate_states[-1]).ravel()*forward_fn(c_state, c_time, *additional_dynamic_args).ravel())).reshape(t0.shape)

            parameter_gradients = []

            for p_shape, num_elem in zip(parameter_shapes, map(math.prod, parameter_shapes)):
                parameter_gradients.append(adj_parameter_gradients[:num_elem].reshape(p_shape))
                adj_parameter_gradients = adj_parameter_gradients[num_elem:]

            input_grads[6:] = parameter_gradients
            inputs_grad_not_finite = list(map(lambda x: False if x is None else (~x.isfinite()).any(), input_grads))
            if any(inputs_grad_not_finite):
                raise ValueError(f"Encountered non-finite grads for inputs: {[
                    inp_idx for inp_idx, inp_grad_is_not_finite in enumerate(inputs_grad_not_finite) if inp_grad_is_not_finite
                ]}")
        return tuple(input_grads)

    return __internal_backward


def get_vmap_method(integrator_type):
    def __internal_vmap(
        info,
        in_dims,
        forward_fn: typing.Callable[
            [torch.Tensor, torch.Tensor, typing.Any], torch.Tensor
        ],
        x0: torch.Tensor,
        t0: torch.Tensor,
        t1: torch.Tensor,
        dt: torch.Tensor,
        integrator_kwargs,
        *additional_dynamic_args,
    ):
        _, x0_bdim, t0_bdim, t1_bdim, _, _, *further_arg_dims = in_dims

        # The strategy is: expand {x0, t0, t1} to all have the dimension
        # being vmapped over.
        # Then, call back into Integrator(expanded_x0, expanded_t0, expanded_t1).

        def maybe_expand_bdim_at_front(x, x_bdim):
            if x_bdim is None:
                return x.expand(info.batch_size, *x.shape)
            return x.movedim(x_bdim, 0)

        # If the Tensor doesn't have the dimension being vmapped over,
        # expand it out. Otherwise, move it to the front of the Tensor
        x0 = maybe_expand_bdim_at_front(x0, x0_bdim)

        # If both t0 and t1 are 0d tensors, then we can simply compute forward_fn in a batched fashion allowing
        # us to avoid calling the integration multiples times. Otherwise, we must call it once for each set of
        # initial conditions, initial times and final times.
        fn_can_be_batched = t0_bdim is None and t1_bdim is None
        if not fn_can_be_batched:
            t0 = maybe_expand_bdim_at_front(t0, t0_bdim)
            t1 = maybe_expand_bdim_at_front(t1, t1_bdim)

            res = [
                integrator_type.apply(
                    forward_fn,
                    _x0,
                    _t0,
                    _t1,
                    dt,
                    integrator_kwargs,
                    *additional_dynamic_args,
                )
                for _x0, _t0, _t1 in zip(x0, t0, t1)
            ]
            res = [
                torch.stack([i[0] for i in res]),
                torch.stack([i[1] for i in res]),
                *[[i[idx] for i in res] for idx in range(2, len(res[0]))],
            ]
            return tuple(res), (0, 0, None, None, None)
        else:
            batched_forward_fn = torch.vmap(
                forward_fn, in_dims=(x0_bdim, t0_bdim, *further_arg_dims)
            )
            print(batched_forward_fn(x0, t0, *additional_dynamic_args))
            res = integrator_type.apply(
                batched_forward_fn,
                x0,
                t0,
                t1,
                dt,
                integrator_kwargs,
                *additional_dynamic_args,
            )
            res = (
                res[0],
                maybe_expand_bdim_at_front(res[1], None),
                res[2].transpose(1, 0),
                maybe_expand_bdim_at_front(res[3], None),
                maybe_expand_bdim_at_front(res[4], None),
            )
            return tuple(res), (0, 0, 0, 0, 0)

    return __internal_vmap


def get_integrator(
    integrator_tableau: torch.Tensor,
    integrator_order: int,
    use_local_extrapolation: bool = True,
    integrator_name: str = None,
) -> typing.Type[classes.Integrator]:
    __integrator_type = classes.create_integrator_class(
        integrator_tableau, integrator_order, use_local_extrapolation, integrator_name
    )

    # Forward integration method
    __internal_forward = get_forward_method(__integrator_type, use_local_extrapolation)
    # Backward integration method
    __internal_backward = get_backward_method(__integrator_type)
    # Enables batching along arbitrary dimensions using `torch.vmap`
    __internal_vmap = get_vmap_method(__integrator_type)

    classes.finalise_integrator_class(
        __integrator_type, __internal_forward, __internal_backward, __internal_vmap
    )

    return __integrator_type


MidpointIntegrator = get_integrator(torch.tensor([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [torch.inf, 0.0, 1.0]
], dtype=torch.float64), integrator_order=2, integrator_name = "MidpointIntegrator")


RK4Integrator = get_integrator(
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
            [torch.inf, 1 / 6, 2 / 6, 2 / 6, 1 / 6],
        ],
        dtype=torch.float64,
    ),
    integrator_order=4,
    integrator_name="RK4Integrator",
)

AdaptiveRK45Integrator = get_integrator(torch.tensor([
    [0.0,       0.0,         0.0,        0.0,         0.0,      0.0,          0.0,      0.0 ],
    [1/5,       1/5,         0.0,        0.0,         0.0,      0.0,          0.0,      0.0 ],
    [3/10,      3/40,        9/40,       0.0,         0.0,      0.0,          0.0,      0.0 ],
    [4/5,       44/45,      -56/15,      32/9,        0.0,      0.0,          0.0,      0.0 ],
    [8/9,       19372/6561, -25360/2187, 64448/6561, -212/729,  0.0,          0.0,      0.0 ],
    [1.0,       9017/3168,  -355/33,     46732/5247,  49/176,  -5103/18656,   0.0,      0.0 ],
    [1.0,       35/384,      0.0,        500/1113,    125/192, -2187/6784,    11/84,    0.0 ],
    [torch.inf, 35/384,      0.0,        500/1113,    125/192, -2187/6784,    11/84,    0.0 ],
    [torch.inf, 5179/57600,  0.0,        7571/16695,  393/640, -92097/339200, 187/2100, 1/40]
], dtype=torch.float64), integrator_order = 5, integrator_name = "AdaptiveRK45Integrator")


AdaptiveRK87Integrator = get_integrator(torch.tensor([
    [0.0,                    0.0,                  0.0,      0.0,       0.0,                      0.0,                    0.0,                     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
    [1/18,                   1/18,                 0.0,      0.0,       0.0,                      0.0,                    0.0,                     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
    [1/12,                   1/48,                 1/16,     0.0,       0.0,                      0.0,                    0.0,                     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
    [1/8,                    1/32,                 0.0,      3/32,      0.0,                      0.0,                    0.0,                     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
    [5/16,                   5/16,                 0.0,     -75/64,     75/64,                    0.0,                    0.0,                     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
    [3/8,                    3/80,                 0.0,      0.0,       3/16,                     3/20,                   0.0,                     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
    [59/400,                 29443841/614563906,   0.0,      0.0,       77736538/692538347,      -28693883/1125000000,    23124283/1800000000,     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
    [93/200,                 16016141/946692911,   0.0,      0.0,       61564180/158732637,       22789713/633445777,     545815736/2771057229,   -180193667/1043307555,    0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
    [5490023248/9719169821,  39632708/573591083,   0.0,      0.0,      -433636366/683701615,     -421739975/2616292301,   100302831/723423059,     790204164/839813087,     800635310/3783071287,    0.0,                     0.0,                   0.0,                  0.0, 0.0],
    [13/20,                  246121993/1340847787, 0.0,      0.0,      -37695042795/15268766246, -309121744/1061227803,  -12992083/490766935,      6005943493/2108947869,   393006217/1396673457,    123872331/1001029789,    0.0,                   0.0,                  0.0, 0.0],
    [1201146811/1299019798, -1028468189/846180014, 0.0,      0.0,       8478235783/508512852,     1311729495/1432422823, -10304129995/1701304382, -48777925059/3047939560,  15336726248/1032824649, -45442868181/3398467696,  3065993473/597172653,  0.0,                  0.0, 0.0],
    [1,                      185892177/718116043,  0.0,      0.0,      -3185094517/667107341,    -477755414/1098053517,  -703635378/230739211,     5731566787/1027545527,   5232866602/850066563,   -4093664535/808688257,    3962137247/1805957418, 65686358/487910083,   0.0, 0.0],
    [1,                      403863854/491063109,  0.0,      0.0,      -5068492393/434740067,    -411421997/543043805,    652783627/914296604,     11173962825/925320556,  -13158990841/6184727034,  3936647629/1978049680,  -160528059/685178525,   248638103/1413531060, 0.0, 0.0],
    [torch.inf, 13451932/455176623, 0.0, 0.0, 0.0, 0.0, -808719846/976000145, 1757004468/5645159321, 656045339/265891186, -3867574721/1518517206, 465885868/322736535,  53011238/667516719,   2/45,                 0.0],
    [torch.inf, 14005451/335480064, 0.0, 0.0, 0.0, 0.0, -59238493/1068277825, 181606767/758867731,   561292985/797845732, -1041891430/1371343529, 760417239/1151165299, 118820643/751138087, -528747749/2220607170, 1/4]
], dtype=torch.float64), integrator_order=8, integrator_name="AdaptiveRK87Integrator")




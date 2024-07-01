import math
from typing import Type

import torch
import einops
import typing

from neuralode.integrators import classes
from neuralode.integrators import routines
from neuralode.integrators import signatures
from neuralode.integrators import helpers
from neuralode.integrators.classes import Integrator


__all__ = ['get_integrator', 'MidpointIntegrator', 'RK4Integrator',
           'AdaptiveRK45Integrator',]


def get_integrator(integrator_tableau: torch.Tensor, integrator_order: int, use_local_extrapolation: bool = True,
                   integrator_name: str = None) -> Type[Integrator]:
    __integrator_type = classes.create_integrator_class(integrator_tableau, integrator_order,
                                                        use_local_extrapolation, integrator_name)
    
    def __internal_forward(ctx: torch.autograd.function.FunctionCtx,
                           forward_fn: typing.Callable[[torch.Tensor, torch.Tensor, typing.Any], torch.Tensor],
                           x0: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor, dt: torch.Tensor,
                           atol: torch.Tensor, rtol: torch.Tensor, *additional_dynamic_args):
        
        integrator_spec = (
            __integrator_type.integrator_tableau.clone().to(x0.device, x0.dtype),
            __integrator_type.is_adaptive,
            __integrator_type.number_of_stages,
            __integrator_type.integrator_order,
            __integrator_type.use_local_extrapolation
            )
        
        c_state, c_time, intermediate_states, intermediate_times, error_in_state = routines.solve_ivp(
            forward_fn, integrator_spec, x0, t0, t1, dt, atol, rtol, additional_dynamic_args
            )
        
        non_differentiable_parameters = [dt]
        backward_save_variables = [x0, t0, t1, dt, c_state, c_time, intermediate_states, intermediate_times,
                                   *additional_dynamic_args]
        if __integrator_type.is_adaptive:
            non_differentiable_parameters = non_differentiable_parameters + [atol, rtol]
            backward_save_variables = [atol, rtol] + backward_save_variables
        ctx.mark_non_differentiable(*non_differentiable_parameters)
        ctx.save_for_backward(*backward_save_variables)
        
        ctx.__internal_forward = forward_fn
        
        return c_state, c_time, intermediate_states, intermediate_times, error_in_state.detach()
    
    def __internal_backward(ctx: torch.autograd.function.FunctionCtx,
                            d_c_state: torch.Tensor,
                            d_c_time: torch.Tensor,
                            d_intermediate_states: torch.Tensor,
                            d_intermediate_times: torch.Tensor,
                            d_error_in_state: torch.Tensor) -> tuple[torch.Tensor | None]:
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
        forward_fn: signatures.integration_fn_signature = ctx.__internal_forward
        
        # Then we retrieve the input variables
        if __integrator_type.is_adaptive:
            atol, rtol, x0, t0, t1, dt, c_state, c_time, intermediate_states, intermediate_times, *additional_dynamic_args = ctx.saved_tensors
            tol_args = [atol, rtol]
        else:
            x0, t0, t1, dt, c_state, c_time, intermediate_states, intermediate_times, *additional_dynamic_args = ctx.saved_tensors
            atol, rtol = torch.inf, torch.inf
            tol_args = []
        
        inputs = forward_fn, x0, t0, t1, dt, atol, rtol, *additional_dynamic_args
        input_grads = [None for _ in range(len(inputs))]
        
        if any(ctx.needs_input_grad):
            # Construct the adjoint equation
            adjoint_fn = helpers.construct_adjoint_fn(forward_fn, c_state.shape)
            
            # We ensure that gradients are enabled so that autograd tracks the variable operations
            # For pointwise functionals, the initial adjoint state is simply the incoming gradients
            parameter_shapes = [i.shape for i in additional_dynamic_args]
            packed_reverse_state = torch.cat([
                c_state.ravel(),
                (d_c_state + d_intermediate_states[-1]).ravel(),
                ])
            if len(additional_dynamic_args) > 0:
                packed_reverse_state = torch.cat([
                    packed_reverse_state,
                    torch.zeros(sum(map(math.prod, parameter_shapes)), device = c_state.device, dtype = c_state.dtype)
                    ])
            
            current_adj_time = t1
            current_adj_state = packed_reverse_state
            
            if torch.any(d_intermediate_states != 0.0):
                # We only need to account for the incoming gradients if any are non-zero
                for next_adj_time, d_inter_state in zip(intermediate_times[1:-1].flip(dims = [0]),
                                                        d_intermediate_states[1:-1].flip(dims = [0])):
                    # The incoming gradients of the intermediate states are the gradients of the state defined at
                    # various points in time. For each of these incoming gradients, we need to integrate up to that
                    # temporal boundary and add them to adjoint state
                    if torch.all(d_inter_state == 0.0):
                        # No need to integrate up to the boundary if the incoming gradients are zero
                        continue
                    current_adj_state, current_adj_time, _, _, _ = __integrator_type.apply(adjoint_fn,
                                                                                           current_adj_state,
                                                                                           current_adj_time,
                                                                                           next_adj_time, -dt,
                                                                                           *tol_args,
                                                                                           *additional_dynamic_args)
                    packed_reverse_state = torch.cat([
                        torch.zeros_like(c_state.ravel()),
                        d_inter_state.ravel(),
                        ])
                    if len(additional_dynamic_args) > 0:
                        packed_reverse_state = torch.cat([
                            packed_reverse_state,
                            torch.zeros(sum(map(math.prod, parameter_shapes)), device = c_state.device,
                                        dtype = c_state.dtype)
                            ])
                    current_adj_state = current_adj_state + packed_reverse_state
            
            final_adj_state, final_adj_time, _, _, _ = __integrator_type.apply(adjoint_fn, current_adj_state,
                                                                               current_adj_time, t0, -dt, *tol_args,
                                                                               *additional_dynamic_args)
            
            adj_variables = final_adj_state[c_state.numel():2 * c_state.numel()].reshape(c_state.shape)
            adj_parameter_gradients = final_adj_state[2 * c_state.numel():]
            
            # The gradients of the incoming state are equal to the gradients from the first element of the
            # intermediate state plus the lagrange variables
            input_grads[1] = adj_variables + d_intermediate_states[0].ravel().reshape(c_state.shape)
            
            # The gradient of the initial time is equal to the gradient from the first element of the intermediate times
            # minus the product of the lagrange variables and the derivative of the system at the initial time
            input_grads[2] = d_intermediate_times[0].ravel() - einops.einsum(adj_variables.ravel(), forward_fn(
                final_adj_state[:c_state.numel()].reshape(c_state.shape), final_adj_time,
                *additional_dynamic_args).ravel(), "i,i->")
            # The gradient of the final time is equal to the gradient from the gradient in the final state
            # plus the product of the lagrange variables and the derivative of the system at the final time
            input_grads[3] = (d_c_time + d_intermediate_times[-1]) + einops.einsum(
                (d_c_state + d_intermediate_states[-1]).ravel(),
                forward_fn(c_state, c_time, *additional_dynamic_args).ravel(), "i,i->")
            
            parameter_gradients = []
            
            for p_shape, num_elem in zip(parameter_shapes, map(math.prod, parameter_shapes)):
                parameter_gradients.append(None)
                adj_parameter_gradients, parameter_gradients[-1] = adj_parameter_gradients[
                                                                   num_elem:], adj_parameter_gradients[
                                                                               :num_elem].reshape(p_shape)
            
            input_grads[7:] = parameter_gradients
            
            if not __integrator_type.is_adaptive:
                input_grads = input_grads[:5] + input_grads[7:]
        return tuple(input_grads)
    
    classes.finalise_integrator_class(__integrator_type, __internal_forward, __internal_backward)
    
    return __integrator_type


MidpointIntegrator = get_integrator(torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [torch.inf, 0.0, 1.0]
    ], dtype=torch.float64), integrator_order=2, integrator_name = "MidpointIntegrator")


RK4Integrator = get_integrator(torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [torch.inf, 1/6, 2/6, 2/6, 1/6]
    ], dtype=torch.float64), integrator_order = 5, integrator_name = "RK4Integrator")


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

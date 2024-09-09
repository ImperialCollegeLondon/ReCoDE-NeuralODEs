import torch
import abc
import typing

from neuralode.integrators import signatures


# By using a metaclass, we can easily specify an appropriate class interface.
# This enables us to
#   1) Type check statically that everything is well-defined
#   2) Use methods like `isinstance` to distinguish between bare torch.autograd functions and our integrators
class Integrator(
    torch.autograd.Function, metaclass=torch.autograd.function.FunctionMeta
):
    __init__ = torch.autograd.Function.__init__
    __call__ = torch.autograd.Function.__call__

    @property
    @abc.abstractmethod
    def integrator_tableau(self) -> torch.Tensor:
        # Butcher tableau for the integrator
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def integrator_order(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_adaptive(self) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def number_of_stages(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def use_local_extrapolation(self) -> bool:
        raise NotImplementedError

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: typing.Any,
        outputs: typing.Any,
    ) -> None:
        forward_fn, x0, t0, t1, dt, integrator_kwargs, *additional_dynamic_args = inputs
        c_state, c_time, intermediate_states, intermediate_times, error_in_state = (
            outputs
        )

        non_differentiable_parameters = [dt]
        backward_save_variables = [
            x0,
            t0,
            t1,
            dt,
            c_state,
            c_time,
            intermediate_times,
            *additional_dynamic_args,
        ]
        ctx.mark_non_differentiable(*non_differentiable_parameters)
        ctx.save_for_backward(*backward_save_variables)
        ctx.integrator_kwargs = integrator_kwargs
        ctx.forward_fn = forward_fn


def create_integrator_class(
    integrator_tableau: torch.Tensor,
    integrator_order: int,
    use_local_extrapolation: bool = True,
    integrator_name: str = None,
) -> typing.Type[Integrator]:
    # We look at the first column of the last two rows, and if both are `inf`, we know the method is adaptive
    is_adaptive = torch.isinf(integrator_tableau[-1, 0]) and torch.isinf(
        integrator_tableau[-2, 0]
    )
    # The number of stages is the number of rows minus the last row
    # (or last two rows if the method is adaptive)
    number_of_stages = integrator_tableau.shape[0] - 1
    if is_adaptive:
        number_of_stages -= 1
    # The `type` function in this form works to dynamically create a class,
    # the first parameter is the class name, the second are parent classes,
    # and the last are the class attributes. We store the integrator attributes
    # here, and reference them in the integration code.
    # In this way, we can query these parameters at a future point.
    __integrator_type = type(
        integrator_name,
        (Integrator,),
        {
            "integrator_tableau": integrator_tableau.clone(),
            "integrator_order": integrator_order,
            "is_adaptive": is_adaptive,
            "number_of_stages": number_of_stages,
            "use_local_extrapolation": use_local_extrapolation,
        },
    )
    return __integrator_type


def finalise_integrator_class(
    integrator_type: typing.Type[Integrator],
    forward_method: signatures.forward_method_signature,
    backward_method: signatures.backward_method_signature,
    vmap_method,
) -> typing.Type[Integrator]:
    integrator_type.forward = staticmethod(forward_method)
    integrator_type.backward = staticmethod(backward_method)
    integrator_type.vmap = staticmethod(vmap_method)
    return integrator_type

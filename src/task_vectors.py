import abc

import torch

from src.linearize import LinearizedImageEncoder


class _TaskVector(abc.ABC):
    def __init__(
        self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None
    ):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert (
                pretrained_checkpoint is not None and finetuned_checkpoint is not None
            )
            with torch.no_grad():
                pretrained_state_dict = self._load_checkpoint(
                    pretrained_checkpoint
                ).state_dict()
                finetuned_state_dict = self._load_checkpoint(
                    finetuned_checkpoint
                ).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype == torch.int64:
                        continue
                    if pretrained_state_dict[key].dtype == torch.uint8:
                        continue
                    self.vector[key] = (
                        finetuned_state_dict[key] - pretrained_state_dict[key]
                    )

    @abc.abstractmethod
    def _load_checkpoint(self, checkpoint):
        """Load a checkpoint into a model."""
        raise NotImplementedError

    @abc.abstractmethod
    def _cast_to_same_type(self, other):
        raise NotImplementedError

    def __add__(self, other):
        """Add two task vectors together."""
        other = self._cast_to_same_type(other)
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return self.__class__(vector=new_vector)

    def __sub__(self, other):
        """Subtract two task vectors."""
        return self.__add__(-other)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return self.__class__(vector=new_vector)

    def __pow__(self, power):
        """Power of a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] ** power
        return self.__class__(vector=new_vector)

    def __mul__(self, other):
        """Multiply a task vector by a scalar."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = other * self.vector[key]
        return self.__class__(vector=new_vector)

    def dot(self, other):
        """Dot product of two task vectors."""
        other = self._cast_to_same_type(other)
        with torch.no_grad():
            dot_product = 0.0
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                dot_product += torch.sum(self.vector[key] * other.vector[key])
        return dot_product

    def norm(self):
        """Norm of a task vector."""
        return torch.sqrt(self.dot(self))

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = self._load_checkpoint(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(
                        f"Warning: key {key} is present in the pretrained state dict but not in the task vector"  # noqa: E501
                    )
                    continue
                new_state_dict[key] = (
                    pretrained_state_dict[key] + scaling_coef * self.vector[key]
                )
        pretrained_model.load_state_dict(new_state_dict)
        return pretrained_model


class NonLinearTaskVector(_TaskVector):
    """A task vector for nonlinear models."""

    def _load_checkpoint(self, checkpoint):
        """Load a checkpoint into a model."""
        return torch.load(checkpoint, map_location="cpu")

    def apply_to_nonlinear(self, pretrained_nonlinear_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a nonlinear pretrained model."""
        return self.apply_to(pretrained_nonlinear_checkpoint, scaling_coef)

    def apply_to_linear(self, pretrained_linear_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a linear pretrained model."""
        return nonlinear_to_linear(self).apply_to(
            pretrained_linear_checkpoint, scaling_coef
        )

    def _cast_to_same_type(self, other):
        return linear_to_nonlinear(other, self.vector.keys())


class LinearizedTaskVector(_TaskVector):
    """A task vector for linearized models."""

    def _load_checkpoint(self, checkpoint):
        """Load a checkpoint into a model."""
        return LinearizedImageEncoder.load(checkpoint)

    def apply_to_nonlinear(
        self, pretrained_nonlinear_checkpoint, param_names, scaling_coef=1.0
    ):
        """Apply a task vector to a nonlinear pretrained model."""
        return linear_to_nonlinear(self, param_names).apply_to(
            pretrained_nonlinear_checkpoint, scaling_coef
        )

    def apply_to_linear(self, pretrained_linear_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a linear pretrained model."""
        return self.apply_to(pretrained_linear_checkpoint, scaling_coef)

    def get_named_parameters(self, param_names):
        """Get the named parameters of the task vector."""
        params = {k: v for k, v in self.vector.items() if "model.params0" not in k}
        return {k: v for k, v in zip(param_names, params.values())}

    def _cast_to_same_type(self, other):
        return nonlinear_to_linear(other)


def nonlinear_to_linear(nonlinear_task_vector):
    """Convert a nonlinear task vector to a linear task vector."""
    if isinstance(nonlinear_task_vector, LinearizedTaskVector):
        return nonlinear_task_vector
    else:
        linear_params = {
            f"model.params.{i}": v
            for i, v in enumerate(nonlinear_task_vector.vector.values())
        }
        # The diff of the init params of the linearized moodels are all zero.
        linear_params |= {
            f"model.params0.{i}": torch.zeros_like(v)
            for i, v in enumerate(nonlinear_task_vector.vector.values())
        }
        return LinearizedTaskVector(vector=linear_params)


def linear_to_nonlinear(linear_task_vector, param_names):
    """Convert a linear task vector to a nonlinear task vector."""
    if isinstance(linear_task_vector, NonLinearTaskVector):
        return linear_task_vector
    else:
        return NonLinearTaskVector(
            vector=linear_task_vector.get_named_parameters(param_names)
        )

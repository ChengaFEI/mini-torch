from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from typing_extensions import Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    args_right = [val for val in vals]
    args_right[arg] += epsilon / 2

    args_left = [val for val in vals]
    args_left[arg] -= epsilon / 2

    return (f(*args_right) - f(*args_left)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    sorted_variables: List[Variable] = []
    opened_variables: List[int] = []
    visited_variables: List[int] = []

    # Recursive function to sort variables
    def visit_variable(variable: Variable) -> None:
        # Terminating Case
        if variable.is_constant():
            return
        if variable.unique_id in opened_variables:
            raise RuntimeError("Computation graph is invalid: Cyclic")
        if variable.unique_id in visited_variables:
            return

        # Recursive Case
        opened_variables.append(variable.unique_id)

        for parent_variable in variable.parents:
            visit_variable(parent_variable)

        opened_variables.remove(variable.unique_id)
        visited_variables.append(variable.unique_id)
        sorted_variables.insert(0, variable)

    # Run the recursion
    visit_variable(variable)
    return sorted_variables


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sorted_variables: Iterable[Variable] = topological_sort(variable)
    derivs: Dict[int, Any] = {variable.unique_id: deriv}

    for current_variable in sorted_variables:
        current_deriv = derivs[current_variable.unique_id]

        if current_variable.is_leaf():
            current_variable.accumulate_derivative(current_deriv)
        else:
            for parent_variable, parent_deriv in current_variable.chain_rule(
                current_deriv
            ):
                if parent_variable.unique_id not in derivs:
                    derivs[parent_variable.unique_id] = 0.0
                derivs[parent_variable.unique_id] += parent_deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values

import subprocess
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor
from typing import Optional, List, Tuple

def single_dimer_ngspice_loss(p):
    gainr1, gainr2, resc1, resc2, lam, factor = p
    input_args = ['ngspice'] + [str(gainr1), str(gainr2), str(resc1), str(resc2), str(lam), str(factor)]
    result = subprocess.check_output(["julia", "single_dimer.jl"] + input_args)
    result = float(result.decode("utf-8").strip())
    return result

class SingleDimerCMTLoss(SyntheticTestFunction):
    r"""Single Dimer CMT Loss function.

    This function calls an external Julia script `single_dimer.jl` to compute the loss.
    The function takes a 5-dimensional input and returns a scalar loss value.

    The function's parameters are:
    w2, k, n11, n10, n20
    """

    _optimal_value = None  # Set to None if the global minimum is not known
    _check_grad_at_opt: bool = False  # Disable gradient checking since it's an external function

    def __init__(
        self,
        dim: int = 5,
        noise_std: Optional[float] = 1e-6,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension. This should be 5 for your function.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function (useful for maximization problems).
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        if dim != 5:
            raise ValueError("This function is defined for 5-dimensional input only.")
        self.dim = dim
        if bounds is None:
            bounds = [(0.0, 10.0) for _ in range(self.dim)]  # Adjust bounds as needed
        self._optimizers = None  # Set to None since the global optimizer is unknown
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        results = []
        for x in X:
            p = x.tolist()
            result = -self.single_dimer_cmt_loss(p)
            results.append(result)
        return torch.tensor(results, dtype=torch.float32)

    @staticmethod
    def single_dimer_cmt_loss(p: List[float]) -> float:
        w2, k, n11, n10, n20 = p
        input_args = ['cmt'] + [str(w2), str(k), str(n11), str(n10), str(n20), '-1000.0']
        result = subprocess.check_output(["julia", "single_dimer.jl"] + input_args)
        result = float(result.decode("utf-8").strip())
        return result
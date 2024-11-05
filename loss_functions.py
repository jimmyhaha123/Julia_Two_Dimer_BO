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

def single_dimer_cmt_loss(p: List[float]) -> float:
    w2, k, n11, n10, n20 = p
    input_args = ['cmt'] + [str(w2), str(k), str(n11), str(n10), str(n20), '-1000.0']
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
    
    def single_dimer_cmt_loss(self, p: List[float]) -> float:
        w2, k, n11, n10, n20 = p
        input_args = ['cmt'] + [str(w2), str(k), str(n11), str(n10), str(n20), '-1000.0']
        result = subprocess.check_output(["julia", "single_dimer.jl"] + input_args)
        result = float(result.decode("utf-8").strip())
        return result
        


def two_dimer_cmt_loss(p: List[float]) -> float:
    w2, w3, w4, k, an11, an10, an20, bn11, bn10, bn20, nu0 = p
    input_args = ['cmt'] + [str(w2), str(w3), str(w4), str(k), str(an11), str(an10), str(an20), str(bn11), str(bn10), str(bn20), str(nu0)]
    result = subprocess.check_output(["julia", "two_dimer.jl"] + input_args)
    result = float(result.decode("utf-8").strip())
    return result


class TwoDimerCMTLoss(SyntheticTestFunction):
    r"""Two Dimer CMT Loss function.

    This function calls an external Julia script `Two_dimer.jl` to compute the loss.
    The function takes a 11-dimensional input and returns a scalar loss value.

    The function's parameters are:
    w2, w3, w4, k, an11, an10, an20, bn11, bn10, bn20, nu0
    """

    _optimal_value = None  # Set to None if the global minimum is not known
    _check_grad_at_opt: bool = False  # Disable gradient checking since it's an external function

    def __init__(
        self,
        dim: int = 11,
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
        if dim != 11:
            raise ValueError("This function is defined for 11-dimensional input only.")
        self.dim = dim
        if bounds is None:
            bounds = [(0.0, 10.0) for _ in range(self.dim)]  # Adjust bounds as needed
        self._optimizers = None  # Set to None since the global optimizer is unknown
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        results = []
        for x in X:
            p = x.tolist()
            result = -self.two_dimer_cmt_loss(p)
            results.append(result)
        return torch.tensor(results, dtype=torch.float32)
    
    def two_dimer_cmt_loss(self, p: List[float]) -> float:
        w2, w3, w4, k, an11, an10, an20, bn11, bn10, bn20, nu0 = p
        input_args = ['cmt'] + [str(w2), str(w3), str(w4), str(k), str(an11), str(an10), str(an20), str(bn11), str(bn10), str(bn20), str(nu0)]
        result = subprocess.check_output(["julia", "two_dimer.jl"] + input_args)
        result = float(result.decode("utf-8").strip())
        return result
    
# p = [0.9259501468342337, 0.9370383303858767, 0.8556021732489235, 1.147449709932502, -0.6150562496186814, 0.9808652766864474, 0.512521117858977, -0.571913546816477, 1.3492302785758965, 0.0494632343878712, 0.533884893558711]
# print(two_dimer_cmt_loss(p))
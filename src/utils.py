from typing import Union, Dict, List, Tuple
import numpy as np
import torch

class RAvg:
    """
    Running average for array or dict-valued functions.
    """

    def __init__(self, initial_value: Union[float, np.ndarray, Dict]):
        self.count = 1
        self.value = initial_value
        self.sq_value = self._square(initial_value)

    def _square(self, x):
        if isinstance(x, (float, int)):
            return x * x
        elif isinstance(x, np.ndarray):
            return x * x
        elif isinstance(x, dict):
            return {k: self._square(v) for k, v in x.items()}
        else:
            raise TypeError("Unsupported type for RAvg")

    def update(self, new_value: Union[float, np.ndarray, Dict]):
        """
        Update the running average with a new value.

        Args:
            new_value (Union[float, np.ndarray, Dict]): The new value to include in the average.
        """
        self.count += 1
        delta = self._subtract(new_value, self.value)
        self.value = self._add(self.value, self._divide(delta, self.count))
        delta2 = self._subtract(new_value, self.value)
        self.sq_value = self._add(self.sq_value, self._multiply(delta, delta2))

    def _add(self, a, b):
        if isinstance(a, (float, int)):
            return a + b
        elif isinstance(a, np.ndarray):
            return a + b
        elif isinstance(a, dict):
            return {k: self._add(a[k], b[k]) for k in a}

    def _subtract(self, a, b):
        if isinstance(a, (float, int)):
            return a - b
        elif isinstance(a, np.ndarray):
            return a - b
        elif isinstance(a, dict):
            return {k: self._subtract(a[k], b[k]) for k in a}

    def _multiply(self, a, b):
        if isinstance(a, (float, int)):
            return a * b
        elif isinstance(a, np.ndarray):
            return a * b
        elif isinstance(a, dict):
            return {k: self._multiply(a[k], b[k]) for k in a}

    def _divide(self, a, b):
        if isinstance(a, (float, int)):
            return a / b
        elif isinstance(a, np.ndarray):
            return a / b
        elif isinstance(a, dict):
            return {k: self._divide(a[k], b) for k in a}

    def mean(self):
        """
        Get the current mean value.

        Returns:
            Union[float, np.ndarray, Dict]: The mean value.
        """
        return self.value

    def variance(self):
        """
        Get the current variance.

        Returns:
            Union[float, np.ndarray, Dict]: The variance.
        """
        return self._divide(self.sq_value, self.count - 1) if self.count > 1 else None

    def error(self):
        """
        Get the current standard error of the mean.

        Returns:
            Union[float, np.ndarray, Dict]: The standard error of the mean.
        """
        var = self.variance()
        if var is None:
            return None
        return self._divide(self._sqrt(var), np.sqrt(self.count))

    def _sqrt(self, x):
        if isinstance(x, (float, int)):
            return np.sqrt(x)
        elif isinstance(x, np.ndarray):
            return np.sqrt(x)
        elif isinstance(x, dict):
            return {k: self._sqrt(v) for k, v in x.items()}

def to_tensor(x: Union[float, np.ndarray, Dict], device: torch.device) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Convert input to PyTorch tensor(s).

    Args:
        x (Union[float, np.ndarray, Dict]): Input to convert.
        device (torch.device): Device to put the tensor(s) on.

    Returns:
        Union[torch.Tensor, Dict[str, torch.Tensor]]: PyTorch tensor(s) on the specified device.
    """
    if isinstance(x, (float, int)):
        return torch.tensor(x, device=device)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    elif isinstance(x, dict):
        return {k: to_tensor(v, device) for k, v in x.items()}
    else:
        raise TypeError("Unsupported type for tensor conversion")

def from_tensor(x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Convert PyTorch tensor(s) to NumPy array(s).

    Args:
        x (Union[torch.Tensor, Dict[str, torch.Tensor]]): PyTorch tensor(s) to convert.

    Returns:
        Union[np.ndarray, Dict[str, np.ndarray]]: NumPy array(s).
    """
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, dict):
        return {k: from_tensor(v) for k, v in x.items()}
    else:
        raise TypeError("Unsupported type for tensor conversion")
    
def discretize(x: torch.Tensor, discrete_dims: List[int], bounds: Union[List[Tuple[float, float]], np.ndarray]) -> torch.Tensor:
    """
    Discretize the specified dimensions of the input tensor.

    Args:
        x (torch.Tensor): Input tensor.
        discrete_dims (List[int]): List of dimensions to discretize.
        bounds (Union[List[Tuple[float, float]], np.ndarray]): Bounds for each dimension.

    Returns:
        torch.Tensor: Tensor with discretized dimensions.
    """
    result = x.clone()
    for dim in discrete_dims:
        lower, upper = bounds[dim]
        result[..., dim] = torch.floor(result[..., dim] * (upper - lower + 1)) + lower
    return result
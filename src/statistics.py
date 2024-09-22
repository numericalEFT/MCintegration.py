from typing import Union, Dict
import numpy as np

class Statistics:
    """
    Class to store and manage integration statistics.
    """

    def __init__(self):
        self.iterations = 0
        self.total_evaluations = 0
        self.result = None
        self.error = None
        self.chi_square = None

    def update(self, result: Union[float, np.ndarray, Dict], error: Union[float, np.ndarray, Dict], neval: int):
        """
        Update the statistics with new results.

        Args:
            result (Union[float, np.ndarray, Dict]): The new result of the integration.
            error (Union[float, np.ndarray, Dict]): The estimated error of the new result.
            neval (int): Number of function evaluations for this update.
        """
        self.iterations += 1
        self.total_evaluations += neval

        if self.result is None:
            self.result = result
            self.error = error
        else:
            # Combine results and update error estimates
            # This would depend on the type of result (scalar, array, or dict)
            # and might involve weighted averaging
            pass

        # Update chi-square if applicable
        # self.chi_square = ...

    def __str__(self):
        return f"Result: {self.result}\nError: {self.error}\nIterations: {self.iterations}\nTotal evaluations: {self.total_evaluations}"
from mytorch import Tensor
import numpy as np


def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    sub = preds.__sub__(actual)
    pow_two_sub = sub.__pow__(2)
    sum = pow_two_sub.sum()
    denominator = Tensor(np.array([pow_two_sub.data.size], dtype=np.float64))
    error = sum.__mul__(denominator.__pow__(-1))
    return 
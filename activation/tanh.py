import numpy as np
from mytorch import Tensor, Dependency

def tanh(x: Tensor) -> Tensor:
    """
    TODO: (optional) implement tanh function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    exp_x = np.exp(x.data)
    exp_neg_x = np.exp(-x.data)

    data = (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

    req_grad = x.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            tanh_x = data
            return grad * (1 - tanh_x ** 2)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
from typing import List
from mytorch.layer import Layer
from mytorch.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, layers:List[Layer], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
            for layer in self.layers:
                # Some layers don't have weights (e.g., ReLU)
                if hasattr(layer, "weight") and layer.weight is not None:
                    layer.weight.data -= layer.weight.grad.data * self.learning_rate

                if hasattr(layer, "bias") and layer.need_bias:
                    layer.bias.data -= layer.bias.grad.data * self.learning_rate
from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer

import numpy as np


class Linear(Layer):
    def __init__(self, inputs: int, outputs: int, need_bias: bool = False, mode="xavier") -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        y = x @ self.weight
        if self.need_bias:
            y += self.bias
        return y


    def initialize(self):
        "TODO: initialize weight by initializer function (mode)"
        w_data = initializer(shape=(self.inputs, self.outputs), mode=self.initialize_mode)
        self.weight = Tensor(
            data=w_data,
            requires_grad=True
        )

        "TODO: initialize bias by initializer function (zero mode)"
        if self.need_bias:
            b_data = initializer(shape=(1, self.outputs), mode="zero") 
            self.bias = Tensor(
                data=b_data,
                requires_grad=True
            )

    def zero_grad(self):
        "TODO: implement zero grad"
        "it resets the gradient to zero"
        if self.weight is not None:
            self.weight.zero_grad()
        if self.bias is not None and self.need_bias:
            self.bias.zero_grad()
        else:
            self.bias = None

    def parameters(self):
        "TODO: return weights and bias"
        params = [self.weight]
        if self.need_bias:
            params.append(self.bias)
        return params
        

    def __str__(self) -> str:
        return "linear - total param: {} - in: {}, out: {}".format(self.inputs * self.outputs, self.inputs,
                                                                   self.outputs)

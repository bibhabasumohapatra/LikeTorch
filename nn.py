import numpy as np

class Linear:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.uniform(
            low=0.0, high=1.0, size=(n_inputs, n_neurons)
        ).astype(np.float32)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def __call__(self, x):
        return self.forward(x)


class Dropout:
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, x):
        drop = np.random.uniform(low=0.0, high=1.0, size=x.shape) > self.p
        x = np.multiply(drop, x)
        return x

    def __call__(self, x):
        return self.forward(x)


class sigmoid:
    def __init__(
        self,
    ):
        pass

    def forward(self, x):
        return 1 / (1 + np.exp(x))

    def __call__(self, x):
        return self.forward(x)


class ReLu:
    def __init__(
        self,
    ):
        pass

    def forward(self, x):
        return np.maximum(0, x)

    def __call__(self, x):
        return self.forward(x)


class Softmax:
    def __init__(self, axis=0):
        self.axis = axis

    def forward(self, x):
        x = np.exp(x) / np.sum(np.exp(x), self.axis)  ## Row-Wise
        return x

    def __call__(self, x):
        return self.forward(x)


class Conv2d:
    def __init__(self, kernel=(3, 3), stride=(1, 1)):
        self.kernel = np.ones(kernel)
        self.stride = stride

    def pad(self, image, pad_y, pad_x):
        image = np.pad(
            image,
            ((pad_y, pad_y), (pad_x, pad_x)),
            mode="constant",
            constant_values=0,
        )  ## PADDING confirm
        return image

    def forward(self, image):  ## H, W
        Yc, Xc = image.shape[0] // 2, image.shape[1] // 2
        k_len = self.kernel.shape[0]

        padding = k_len // 2
        out_x, out_y = (image.shape[0] + 2 * padding - k_len) + 1, image.shape[
            1
        ] + 2 * padding - k_len + 1
        out = np.zeros((out_y, out_x))

        image = self.pad(image, padding, padding)
        for i in range(0, image.shape[0] - k_len, self.stride[0]):
            for j in range(0, image.shape[1] - k_len, self.stride[1]):
                iteration = np.multiply(
                    image[i : i + k_len, j : j + k_len], self.kernel
                )
                out[i, j] = np.sum(iteration)
        return out
    def __call__(self, x):
        return self.forward(x)

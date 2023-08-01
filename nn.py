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


class Patch_Generator:
    def __init__(self, kernel=(3, 16, 16), stride=(1, 1)):
        self.kernel = np.ones(kernel)
        self.stride = stride

    def pad(self, image, pad_y, pad_x):
        image = np.pad(
            image,
            ((0, 0), (pad_y, pad_y), (pad_x, pad_x)),
            mode="constant",
            constant_values=0,
        )  ## PADDING confirm
        return image

    def forward(self, image):  ## H, W
        Yc, Xc = image.shape[1] // 2, image.shape[2] // 2
        k_len = self.kernel.shape[1]

        padding = k_len // 2
        patches = []
        out = []
        image = self.pad(image, padding, padding)
        for i in range(0, image.shape[1] - k_len, 16):
            for j in range(0, image.shape[2] - k_len, 16):
                iteration = np.multiply(
                    image[:, i : i + k_len, j : j + k_len], self.kernel
                )
                patches.append(iteration)
                out.append(iteration.flatten())
        return {"patch": np.array(patches), "flatten_patches": np.array(out)}

    def __call__(self, x):
        return self.forward(x)


class Projection_Layer:
    def __init__(
        self,
    ):
        self.layer_1 = Linear(768, 512)
        self.layer_2 = Linear(512, 512)
        self.drop = Dropout(0.5)
        self.relu = ReLu()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.drop(x)
        x = self.relu(x)

        return x

    def __call__(self, x):
        return self.forward(x)


class Multiply_k_q:
    def __init__(
        self,
    ):
        pass

    def forward(self, k, q):
        return np.matmul(q, k.T)

    def __call__(self, k, q):
        return self.forward(k, q)


class Multiply_k_q_v:
    def __init__(
        self,
    ):
        pass

    def forward(self, k_q, v):
        return np.matmul(k_q, v)

    def __call__(self, k_q, v):
        return self.forward(k_q, v)


class Simple_Transformer:
    def __init__(
        self,
    ):
        self.patching = Patch_Generator()
        self.projection = Projection_Layer()
        self.init_rand = np.random.uniform(-0.1, 0.1, size=(196, 512)).astype(np.float32)
        self.W_q_1 = Linear(512, 64)
        self.W_k_1 = Linear(512, 64)
        self.W_v_1 = Linear(512, 64)

        self.W_q_2 = Linear(512, 64)
        self.W_k_2 = Linear(512, 64)
        self.W_v_2 = Linear(512, 64)

        self.mul_q_k_1 = Multiply_k_q()
        self.mul_q_k_2 = Multiply_k_q()

        self.mul_q_k_v_1 = Multiply_k_q_v()
        self.mul_q_k_v_2 = Multiply_k_q_v()

        self.softmax = Softmax()

        self.fc = Linear(128, 512)
        self.dropout = Dropout()
        self.relu = ReLu()
        self.final_layer = Linear(512, 128)

    def forward(self, x):
        x = self.patching(x)["flatten_patches"]  ## 196,768
        x = self.projection(x)  ## 196,512
        x = x + self.init_rand  ## 196,512

        k_1 = self.W_k_1(x)  ## 196,64
        k_2 = self.W_k_2(x)  ## 196,64

        q_1 = self.W_q_1(x)  ## 196,64
        q_2 = self.W_q_2(x)  ## 196,64

        v_1 = self.W_v_1(x)  ## 196,64
        v_2 = self.W_v_2(x)  ## 196,64

        k_1_q_1 = self.mul_q_k_1(k_1, q_1)  ## 196,196
        k_2_q_2 = self.mul_q_k_2(k_2, q_2)  ## 196,196

        out_1 = self.softmax(k_1_q_1)  ## 196,196
        out_2 = self.softmax(k_2_q_2)  ## 196,196

        x_1 = self.mul_q_k_v_1(out_1, v_1)  ## 196x64
        x_2 = self.mul_q_k_v_2(out_2, v_2)  ## 196x64

        x = np.concatenate([x_1, x_2], axis=1)  ## 196x128

        x = self.fc(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.final_layer(x)

        return x

    def __call__(self, x):
        return self.forward(x)

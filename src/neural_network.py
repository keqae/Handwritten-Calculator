import numpy as np
from abc import abstractmethod
from scipy.signal import correlate


def he_normal_numpy(shape):
    fan_in = np.prod(shape[:-1])  # exclude output channels
    stddev = np.sqrt(2. / fan_in)
    return np.random.normal(loc=0.0, scale=stddev, size=shape)


class BaseLayer:
    def __init__(self, name:str, id:int, trainable:bool = True):
        # basic attributes
        self.name = name
        self.id = id
        self.trainable = trainable

        # store parameters during forward propagation
        self.input = None
        self.output = None

        # dictionaries
        self.parameters = {}
        self.gradients = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}, ID: {self.id}, trainable: {self.trainable}"

    # limited functionality as designed to be overridden by child classes
    @abstractmethod
    def forward(self, input:np.ndarray) -> np.ndarray:
        pass

    # limited functionality as designed to be overridden by child classes
    @abstractmethod
    def backward(self, upstream:np.ndarray) -> np.ndarray:
        pass


    def update_parameters(self, learning_rate: float = 0.1) -> None:
        pass

    def initialize_parameters(self) -> None:
        pass

    def get_parameters(self) -> dict:
        return self.parameters

    def get_gradients(self) -> dict:
        return self.gradients


class ConvolutionalLayer(BaseLayer):
    def __init__(self,
            name:str, id:int,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            trainable:bool = True
                 ):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.pad = 1

        super().__init__(name, id, trainable)


    def initialize_parameters(self) -> None:
        # safety check
        if not self.trainable:
            raise Exception("Attempted to initialize a layer that is not trainable")

        # weights come in shape eg. (3,3,16,1)
        weight_shape = (self.kernel_size, self.kernel_size, self.in_channels, self.out_channels)

        # randomly initialise weights according to HeNormal distribution
        self.parameters["W"] = he_normal_numpy(weight_shape)
        # randomly initialise biases to zero
        self.parameters["b"] = np.zeros((self.out_channels, 1))


    def forward(self, input:np.ndarray) -> np.ndarray:
        self.input = input

        # fetch dimensions from image , color_channels, and number of images
        num_images, image_height, image_width, image_channels = input.shape

        # fetch kernel dimensions, input channels, and number of kernels
        kernel_height, kernel_width, kernel_in_channels, kernel_out_channels = self.parameters["W"].shape

        # add padding of 1
        input_pad = np.pad(input, pad_width=((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='constant')


        # create blank canvas for the feature map
        output = np.zeros((num_images, image_height, image_width, kernel_out_channels))


        # kernel slide, vectorised operation over image array
        for image in range(num_images):
            for kout in range(kernel_out_channels):
                for kin in range(kernel_in_channels):
                    # convolution computation
                    output[image, :, :, kout] += correlate(input_pad[image, :, :, kin],
                                                           self.parameters["W"][:, :, kin, kout],
                                                           mode="valid")

                # add bias to every kernel
                output[image, :, :, kout] += self.parameters["b"][kout]

        self.output = output
        return output

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        # get shapes
        num_images, input_height, input_width, input_channels = self.input.shape
        kernel_height, kernel_width, _, output_channels = self.parameters["W"].shape


        # create arrays of zeros to store gradients
        input_padded = np.pad(self.input, ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)))
        d_input_padded = np.zeros_like(input_padded)
        dW = np.zeros_like(self.parameters["W"])
        db = np.zeros_like(self.parameters["b"])

        # find gradient accumulation
        for image in range(num_images):
            for out_channel in range(output_channels):
                # accumulate gradients for biases
                db[out_channel] += np.sum(upstream[image, :, :, out_channel])

                for in_channel in range(input_channels):
                    # gradients of weights
                    dW[:, :, in_channel, out_channel] += correlate(
                        input_padded[image, :, :, in_channel],
                        upstream[image, :, :, out_channel],
                        mode="valid"
                    )

                    d_input_padded[image, :, :, in_channel] += correlate(
                        upstream[image, :, :, out_channel],
                        self.parameters["W"][:, :, in_channel, out_channel],
                        mode="full"
                    )

        # remove padding from d_input
        d_input = d_input_padded[:, self.pad:-self.pad, self.pad:-self.pad, :]

        # store gradients
        self.gradients["W"] = dW / num_images
        self.gradients["b"] = db / num_images

        return d_input

    def update_parameters(self, learning_rate: float = 0.1) -> None:
        self.parameters["W"] -= learning_rate * self.gradients["W"]
        self.parameters["b"] -= learning_rate * self.gradients["b"]


    def get_parameters(self) -> dict:
        return self.parameters

    def get_gradients(self) -> dict:
        return self.gradients



class MaxPoolingLayer(BaseLayer):
    def __init__(self, name:str, id:int, trainable:bool = False):
        super().__init__(name, id, trainable)

        self.pool_size = 2
        self.stride = 2

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input

        # unpack input
        num_images, image_height, image_width, image_channels = input.shape

        # define dimensions
        output_height = image_height // self.pool_size
        output_width = image_width // self.pool_size

        if image_height % self.pool_size != 0 or image_width % self.pool_size != 0:
            raise ValueError(
                f"Input shape {input.shape} is not valid for pooling size {self.pool_size}."
                " Height and width must be divisible by pool size.")

        # reshape input
        reshaped = input.reshape(num_images, output_height, self.pool_size, output_width, self.pool_size, image_channels)

        # pool
        output = reshaped.max(axis=(2, 4))

        self.output = output

        return output

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        # collect parameters from upstream layer
        num_images, output_height, output_width, channels = upstream.shape
        pooling_height, pooling_width = self.pool_size, self.pool_size

        d_input = np.zeros_like(self.input)

        for image in range(num_images):
            for row in range(output_height):
                for column in range(output_width):
                    for channel in range(channels):
                        # get pooling window
                        h_start, h_end = row * pooling_height, (row + 1) * pooling_height
                        w_start, w_end = column * pooling_width, (column + 1) * pooling_width
                        window = self.input[image, h_start:h_end, w_start:w_end, channel]

                        # mask where max value occurred
                        mask = (window == np.max(window))
                        d_input[image, h_start:h_end, w_start:w_end, channel] += mask * upstream[image, row, column, channel]

        return d_input


class FlatteningLayer(BaseLayer):
    def __init__(self, name:str, id: int, trainable: bool = False):

        super().__init__(name, id, trainable)
        self.input = []
        self.output = []


    def forward(self, input:np.ndarray) -> np.ndarray:
        self.input = input
        # fetch flattening dimensions
        num_images, image_height, image_width, image_channels = input.shape

        # flatten
        output = np.reshape(input, (num_images, -1))

        self.output = output

        return output


    def backward(self, upstream:np.ndarray) -> np.ndarray:
        return upstream.reshape(self.input.shape)


# classification mechanism
class FullyConnectedLayer(BaseLayer):
    def __init__(self, name:str, id:int,
                 in_channels:int,
                 image_size:int,
                 out_channels:int,
                 trainable:bool = True):
        super().__init__(name, id, trainable)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size

        self.input = None
        self.output = None

    def initialize_parameters(self) -> None:
        # safety check
        if not self.trainable:
            raise Exception("Attempted to initialise a layer that is not trainable")

        # initialise weights and biases according to input size
        std_dev = np.sqrt(2.0/ self.image_size)
        self.parameters["W"] = np.random.normal(0, std_dev, (self.in_channels, self.out_channels))
        self.parameters["b"] = np.zeros((1, self.out_channels))


    def forward(self, input:np.ndarray) -> np.ndarray:
        self.input = input

        # compute weighted sum
        self.output = np.dot(input, self.parameters["W"]) + self.parameters["b"]

        return self.output

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        # upstream = dL/dZ where Z = input @ W + b
        dZ = upstream  # (batch_size, out_channels)
        derivative_weights = np.dot(self.input.T, dZ) / dZ.shape[0]
        db = np.sum(dZ, axis=0, keepdims=True) / dZ.shape[0]

        dInput = np.dot(dZ, self.parameters["W"].T)

        # Store gradients
        self.gradients["W"] = derivative_weights
        self.gradients["b"] = db

        return dInput


    def update_parameters(self, learning_rate: float = 0.1) -> None:
        self.parameters["W"] -= learning_rate * self.gradients["W"]
        self.parameters["b"] -= learning_rate * self.gradients["b"]


    def get_parameters(self) -> dict:
        return self.parameters

    def get_gradients(self) -> dict:
        return self.gradients



#activation layer
class ReLULayer(BaseLayer):
    def __init__(self, name: str, id: int, trainable: bool = False):
        super().__init__(name, id, trainable)
        self.input = None
        self.output = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.output = np.maximum(0, input)
        return self.output

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        return upstream * (self.input > 0)


# classification layer
class SoftmaxLayer(BaseLayer):
    def __init__(self, name:str, id:int, trainable=False):
        super().__init__(name, id, trainable)

    def forward(self, input:np.ndarray) -> np.ndarray:

        # print(f"logits before softmax: {input}")
        # stable softmax
        x_max = np.max(input, axis=1, keepdims=True)
        e_x = np.exp(input - x_max)
        softmax = e_x / np.sum(e_x, axis=1, keepdims=True)

        return softmax

    # ignore gradient and pass it upstream
    def backward(self, upstream:np.ndarray) -> np.ndarray:
        return upstream


# loss function
def cross_entropy_loss(y_pred, y_true):

    print("y_true shape:", y_true.shape)
    print("y_true[0]:", y_true[0])

    # handle one-hot encoded labels
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true

    m = y_pred.shape[0]

    # add small epsilon to prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    log_likelihood = -np.log(y_pred[np.arange(m), y_true_indices])
    return np.sum(log_likelihood) / m


# combine softmax + cross-entropy gradient
def softmax_crossentropy_backward(y_pred, y_true):
    m = y_pred.shape[0]

    # handle one hot encoded labels
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true

    gradient = y_pred.copy()
    gradient[np.arange(m), y_true_indices] -= 1
    gradient /= m
    return gradient


#main architecture
class ConvolutionalNeuralNetwork:
    def __init__(self, train=False):
        self.train = train

        self.layers = [
            ConvolutionalLayer("Conv1", 0, in_channels=1, out_channels=16, kernel_size=3),
            ReLULayer("ReLU1", 1),
            MaxPoolingLayer("Pool1", 2),

            ConvolutionalLayer("Conv2", 3, in_channels=16, out_channels=16, kernel_size=3),
            ReLULayer("ReLU2", 4),
            MaxPoolingLayer("Pool2", 5),

            FlatteningLayer("Flatten", 6),

            FullyConnectedLayer("FC1", 7, in_channels=784, image_size=5*5*16, out_channels=128),
            ReLULayer("ReLU3", 8),
            FullyConnectedLayer("FC2", 9, in_channels=128, image_size=128, out_channels=14),

            SoftmaxLayer("Softmax", 10)
        ]

        # initialise parameters
        for layer in self.layers:
            if layer.trainable:
                layer.initialize_parameters()

    # forward propagate through every single layer
    def forward(self, input: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            input = layer.forward(input)
        return input

    # train
    def backward(self, loss_gradient: np.ndarray, learning_rate) -> None:
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient)
            # update weights and biases if training
            layer.update_parameters(learning_rate)

    def assemble_params(self):
        all_params = {}

        for layer in self.layers:
            if layer.trainable:
                all_params[f"{layer.name}_W"] = layer.get_parameters()["W"]
                all_params[f"{layer.name}_b"] = layer.get_parameters()["b"]

        return all_params

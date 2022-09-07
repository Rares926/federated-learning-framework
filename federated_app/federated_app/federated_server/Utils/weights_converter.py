
from flwr.common import (
    Parameters,
    Weights
)

from ..Utils.array_converter import ArrayConverter


class WeightsConverter:

    @staticmethod
    def weights_to_parameters(weights: Weights) -> Parameters:
        """Convert NumPy weights to parameters object."""
        tensors = [ArrayConverter.ndarray_to_bytes(ndarray)
                   for ndarray in weights]
        return Parameters(tensors=tensors, tensor_type="numpy.nda")

    @staticmethod
    def parameters_to_weights(parameters: Parameters) -> Weights:
        """Convert parameters object to NumPy weights."""
        return [ArrayConverter.bytes_to_ndarray(tensor)
                for tensor in parameters.tensors]

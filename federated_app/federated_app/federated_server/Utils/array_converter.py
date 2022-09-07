import numpy as np
from typing import cast


class ArrayConverter:

    @staticmethod
    def ndarray_to_bytes(ndarray: np.ndarray) -> bytes:
        """Serialize NumPy array to bytes."""
        return cast(bytes, ndarray.tobytes())

    @staticmethod
    def bytes_to_ndarray(tensor: bytes) -> np.ndarray:
        """Deserialize NumPy array from bytes."""
        ndarray_deserialized = np.frombuffer(tensor, dtype=np.float32)
        return cast(np.ndarray, ndarray_deserialized)

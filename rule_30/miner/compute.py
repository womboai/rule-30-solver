from math import log2

import numpy as np
from numpy.typing import NDArray


def compute_response(data: NDArray[np.uint64]) -> NDArray[np.uint64]:
    raw_bytes = data.tobytes()
    combined_int = int.from_bytes(raw_bytes, "little")
    transformed_int = combined_int ^ ((combined_int << 1) | (combined_int << 2))
    
    num_bytes = (transformed_int.bit_length() + 7) // 8
    num_bytes += (8 - num_bytes % 8)   
    
    transformed_bytes = transformed_int.to_bytes(num_bytes, 'little')
    return np.frombuffer(transformed_bytes, dtype=np.uint64)
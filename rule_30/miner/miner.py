import os
import sys

import uvicorn
from fiber.logging_utils import get_logger
from fiber.miner import server
from fiber.miner.middleware import configure_extra_logging_middleware
from numpy.typing import NDArray
import numpy as np

logger = get_logger(__name__)

app = server.factory_app(debug=True)

app.include_router(get_subnet_router())


if os.getenv("ENV", "dev").lower() == "dev":
    configure_extra_logging_middleware(app)


def compute_response(data: NDArray[np.uint64]) -> NDArray[np.uint64]:
    raw_bytes = data.tobytes()
    combined_int = int.from_bytes(raw_bytes, "little")
    transformed_int = combined_int ^ ((combined_int << 1) | (combined_int << 2))
    transformed_bytes = transformed_int.to_bytes(len(raw_bytes), 'little')

    return np.frombuffer(transformed_bytes, dtype=np.uint64)


def main():
    sys.argv.append("rule_30.miner:app")

    uvicorn.main()

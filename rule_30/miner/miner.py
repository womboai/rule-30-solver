import os
import sys
from itertools import chain

import uvicorn
from fiber.logging_utils import get_logger
from fiber.miner import server
from fiber.miner.middleware import configure_extra_logging_middleware

logger = get_logger(__name__)

app = server.factory_app(debug=True)

app.include_router(get_subnet_router())


if os.getenv("ENV", "dev").lower() == "dev":
    configure_extra_logging_middleware(app)
 
def sample_miner(int_list: list[int]) -> list[int]:
    combined_int = int.from_bytes(b''.join(num.to_bytes(8, 'little') for num in int_list), "little")
    transformed_int = combined_int ^ ((combined_int << 1) | (combined_int << 2))
    masked_int = (transformed_int >> 1) & ((1 << (transformed_int.bit_length() - 2)) - 1)
    transformed_bytes = masked_int.to_bytes((masked_int.bit_length() + 7) // 8, 'little')
    return [int.from_bytes(transformed_bytes[i:i+8], "little") for i in range(0, len(transformed_bytes), 8)]

def main():
    sys.argv.append("rule_30.miner:app")

    uvicorn.main()

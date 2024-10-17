import os
import sys

import numpy as np
import uvicorn
from fiber.logging_utils import get_logger
from fiber.miner import server
from fiber.miner.middleware import configure_extra_logging_middleware

from rule_30.miner.compute import compute_response
from rule_30.models import ComputationData

logger = get_logger(__name__)

app = server.factory_app(debug=True)

if os.getenv("ENV", "dev").lower() == "dev":
    configure_extra_logging_middleware(app)


@app.post("/compute")
def compute(body: ComputationData) -> ComputationData:
    return ComputationData(
        parts=compute_response(np.array(body.parts, dtype=np.uint64)).tolist(),
    )


@app.get("/current_step")
def current_step() -> int:
    return 1


def main():
    sys.argv.append("rule_30.miner:app")

    uvicorn.main()

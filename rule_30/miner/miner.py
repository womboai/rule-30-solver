import os
import sys

import uvicorn
from fiber.logging_utils import get_logger
from fiber.miner import server
from fiber.miner.middleware import configure_extra_logging_middleware

logger = get_logger(__name__)

app = server.factory_app(debug=True)

app.include_router(get_subnet_router())


if os.getenv("ENV", "dev").lower() == "dev":
    configure_extra_logging_middleware(app)


def main():
    sys.argv.append("rule_30.miner:app")

    uvicorn.main()

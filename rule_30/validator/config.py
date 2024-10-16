from argparse import ArgumentParser

from fiber.constants import FINNEY_NETWORK


def get_config():
    argument_parser = ArgumentParser()

    argument_parser.add_argument(
        "--subtensor.chain_endpoint",
        type=str,
        required=False,
        help="Chain address",
        default=None,
    )

    argument_parser.add_argument(
        "--subtensor.network",
        type=str,
        required=False,
        help="Chain network",
        default=FINNEY_NETWORK,
    )

    argument_parser.add_argument("--wallet.name", type=str, required=False, help="Wallet name", default="default")
    argument_parser.add_argument("--wallet.hotkey", type=str, required=False, help="Hotkey name", default="default")

    argument_parser.add_argument("--netuid", type=int, required=True, help="Network UID")

    argument_parser.add_argument(
        "--epoch_length",
        type=int,
        help="The default epoch length (how often we pull the metagraph, measured in 12 second blocks).",
        default=100,
    )

    return vars(argument_parser.parse_args())

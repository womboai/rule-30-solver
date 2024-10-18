import os

WALLET_NAME = os.getenv("WALLET_NAME", "default")
HOTKEY_NAME = os.getenv("HOTKEY_NAME", "default")
NETUID = os.getenv("NETUID")
SUBTENSOR_NETWORK = os.getenv("SUBTENSOR_NETWORK")
SUBTENSOR_ADDRESS = os.getenv("SUBTENSOR_ADDRESS")

EPOCH_LENGTH = int(os.getenv("EPOCH_LENGTH", str(100)))

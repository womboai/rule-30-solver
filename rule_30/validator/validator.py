import asyncio
from collections.abc import Iterable
from itertools import islice, chain
from os import makedirs
from os.path import exists
from pathlib import Path
from time import perf_counter
from typing import Any

from miner import sample_miner
import numpy as np
from cryptography.fernet import Fernet
from fiber.chain.chain_utils import load_hotkey_keypair
from fiber.chain.interface import get_substrate
from fiber.chain.metagraph import Metagraph
from fiber.chain.models import Node
from fiber.logging_utils import get_logger
from fiber.validator.client import make_non_streamed_post, make_non_streamed_get
from fiber.validator.handshake import perform_handshake
from httpx import AsyncClient, Response
from numpy.typing import NDArray
from substrateinterface import Keypair, SubstrateInterface

from .config import get_config

logger = get_logger(__name__)


class Validator:
    wallet_name: str
    wallet_hotkey: str

    keypair: Keypair
    substrate: SubstrateInterface
    metagraph: Metagraph
    client: AsyncClient

    step: numpy.uint64
    current_row: NDArray[numpy.uint64]
    center_column: NDArray[numpy.uint64]
    scores: list[float]

    valid_miners: list[int]
    hotkeys: list[str]

    def __init__(self):
        config = get_config()

        self.wallet_name = config["wallet.name"]
        self.wallet_hotkey = config["wallet.hotkey"]

        self.keypair = load_hotkey_keypair(self.wallet_name, self.wallet_hotkey)

        self.substrate = get_substrate(
            subtensor_network=config["subtensor.network"],
            subtensor_address=config["subtensor.chain_endpoint"],
        )

        self.metagraph = Metagraph(
            substrate=self.substrate,
            netuid=config["netuid"],
            load_old_nodes=False
        )

        self.metagraph.sync_nodes()

        self.client = AsyncClient()

        self.step = numpy.uint64(1)
        self.current_row = numpy.array([1], dtype=numpy.uint64)
        self.center_column = numpy.array([1], dtype=numpy.uint64)
        self.valid_miners = []
        self.hotkeys = list(self.metagraph.nodes)

    @property
    def state_path(self):
        full_path = (
            Path.home() /
            ".bittensor" /
            "miners" /
            self.wallet_name /
            self.wallet_hotkey /
            f"netuid{self.metagraph.netuid}" /
            "validator"
        )

        makedirs(full_path, exist_ok=True)

        return full_path / "state.npz"

    def save_state(self):
        logger.log("save_state")

        numpy.savez(
            self.state_path,
            step=self.step,
            current_row=self.current_row,
            center_column=self.center_column,
            valid_miners=self.valid_miners,
            hotkeys=self.hotkeys,
        )

    def load_state(self):
        logger.log("load_state")

        path = self.state_path

        if not exists(path):
            return

        state = numpy.load(path)

        self.step = state["step"]
        self.current_row = state["current_row"]
        self.center_column = state["center_column"]
        self.valid_miners = state["valid_miners"]
        self.hotkeys = state["hotkeys"]

    def sync(self):
        logger.log("sync")

        self.metagraph.sync_nodes()
        self.valid_miners.clear()

        for hotkey, node in self.metagraph.nodes.items():
            if hotkey != self.hotkeys[node.node_id]:
                self.hotkeys[node.node_id] = hotkey

        logger.log("Checking active miners")

        current_steps = asyncio.gather(*(
            self.make_request(node, "current_step")
            for node in self.metagraph.nodes.values()
        ))

        self.valid_miners.extend(*numpy.where(current_steps == self.step))

    async def make_request(self, node: Node, endpoint: str, payload: dict[str, Any] | None = None):
        miner_address = f"http://{node.ip}:{node.port}"

        symmetric_key_str, symmetric_key_uuid = await perform_handshake(
            keypair=self.keypair,
            httpx_client=self.client,
            server_address=miner_address,
            miner_hotkey_ss58_address=node.hotkey,
        )

        if symmetric_key_str is None or symmetric_key_uuid is None:
            return None

        start = perf_counter()

        if payload:
            fernet = Fernet(symmetric_key_str)

            resp = await make_non_streamed_post(
                httpx_client=self.client,
                server_address=miner_address,
                fernet=fernet,
                keypair=self.keypair,
                symmetric_key_uuid=symmetric_key_uuid,
                validator_ss58_address=self.keypair.ss58_address,
                miner_ss58_address=node.hotkey,
                payload=payload,
                endpoint=f"/{endpoint}",
            )
        else:
            resp = await make_non_streamed_get(
                httpx_client=self.client,
                server_address=miner_address,
                symmetric_key_uuid=symmetric_key_uuid,
                validator_ss58_address=self.keypair.ss58_address,
                endpoint=f"/{endpoint}",
            )

        inference_time = perf_counter() - start

        resp.raise_for_status()

        return resp, inference_time

    def nodes_list(self) -> list[Node]:
        nodes: list[Node | None] = len(self.metagraph.nodes) * [None]

        for node in self.metagraph.nodes.values():
            nodes[node.node_id] = node

        return nodes

    async def do_step(self):
        logger.log(f"Evolution step {self.step}")

        iterator = iter(self.current_row)

        chunk_size = numpy.ceil(len(self.current_row) / len(self.valid_miners))

        nodes = self.nodes_list()

        chunks = [
            (nodes[uid], list(islice(iterator, chunk_size)))
            for uid in self.valid_miners
        ]

        responses: Iterable[tuple[Response, float]] = await asyncio.gather(*(
            self.make_request(
                node,
                "compute",
                {
                    "parts": chunk
                }
            )
            for node, chunk in chunks
        ))

        responses: list[tuple[int, list[int] | None, float]] = [
            (uid, response.json()["parts"] if response else None, inference_time)
            for uid, (response, inference_time) in zip(self.valid_miners, responses)
        ]

        for (uid, response, inference_time) in responses:
            if not response:
                self.scores[uid] = 0.0
            else:
                self.scores[uid] = 1 / inference_time

        self.current_row = numpy.array(list(chain(response[1] for response in responses)), dtype=numpy.uint64)

        bit_index = self.step % 64
        current_row_part = self.current_row[self.step / 64]

        self.center_column[-1] = self.center_column[-1] | (current_row_part >> bit_index << bit_index)

        self.step += 1

        self.save_state()

    async def run(self):
        while True:
            try:
                await self.do_step()
            except:
                logger.error(f"Error in evolution step {self.step}", exc_info=True)
    
    async def normalize_scores(self, outputs: np.ndarray) -> np.ndarray:
        def normalize_pair(a: np.uint64, b: np.uint64) -> tuple[np.uint64, np.uint64]:
            carry = a & 1
            a = a >> 1
            b = (carry << (b.bit_length())) | b
            a, b = sample_miner(np.array([a, b], dtype=np.uint64))
            msb = b >> (b.dtype.itemsize * 8 - 1)
            b = b & ((1 << (b.dtype.itemsize * 8 - 1)) - 1)
            a = (a << 1) | msb
            return a, b
        normalized_outputs = []
        for i in range(len(outputs)-2):
            a,b = normalize_pair(outputs[i][0],outputs[i+1][-1])
            normalized_outputs.append(a)
            normalized_outputs.append(b)
        return normalized_outputs

def main():
    asyncio.run(Validator().run())

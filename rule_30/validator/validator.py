import asyncio
from collections.abc import Iterable
from itertools import islice
from math import ceil
from os import makedirs
from os.path import exists
from pathlib import Path
from time import perf_counter, sleep
from typing import TypeAlias

import numpy as np
import wandb
from fiber.chain.chain_utils import load_hotkey_keypair
from fiber.chain.interface import get_substrate
from fiber.chain.metagraph import Metagraph
from fiber.chain.models import Node
from fiber.chain.weights import set_node_weights
from fiber.logging_utils import get_logger
from httpx import AsyncClient, Response
from numpy.typing import NDArray
from pydantic import BaseModel
from substrateinterface import Keypair, SubstrateInterface
from wandb.sdk.wandb_run import Run

from .config import SUBTENSOR_NETWORK, SUBTENSOR_ADDRESS, NETUID, WALLET_NAME, HOTKEY_NAME, EPOCH_LENGTH
from ..models import ComputationData
from ..wandb_config import WANDB_REFRESH_INTERVAL, WANDB_PROJECT, WANDB_ENTITY

logger = get_logger(__name__)


VALIDATOR_VERSION: tuple[int, int, int] = (1, 0, 0)
VALIDATOR_VERSION_STRING = ".".join(map(str, VALIDATOR_VERSION))

WEIGHTS_VERSION = (
    VALIDATOR_VERSION[0] * 10000 +
    VALIDATOR_VERSION[1] * 100 +
    VALIDATOR_VERSION[2]
)


Block: TypeAlias = int
Uid: TypeAlias = int
SS58Key: TypeAlias = str


class UnrecoverableError(Exception):
    pass


class Validator:
    keypair: Keypair
    substrate: SubstrateInterface
    metagraph: Metagraph
    client: AsyncClient

    step: np.uint64
    current_row: NDArray[np.uint64]
    center_column: NDArray[np.uint64]
    scores: list[float]

    valid_miners: list[Uid]
    hotkeys: list[SS58Key]

    last_metagraph_sync: Block

    last_wandb_run: Block
    wandb_run: Run

    def __init__(self):
        self.keypair = load_hotkey_keypair(WALLET_NAME, HOTKEY_NAME)

        self.substrate = get_substrate(
            subtensor_network=SUBTENSOR_NETWORK,
            subtensor_address=SUBTENSOR_ADDRESS,
        )

        self.metagraph = Metagraph(
            substrate=self.substrate,
            netuid=NETUID,
            load_old_nodes=False
        )

        self.metagraph.sync_nodes()

        self.valid_miners = []
        self.hotkeys = list(self.metagraph.nodes)
        self.scores = [0.0] * len(self.metagraph.nodes)

        self.client = AsyncClient()

        self.step = np.uint64(1)
        self.current_row = np.array([1], dtype=np.uint64)
        self.center_column = np.array([1], dtype=np.uint64)

        self.load_state()

    def check_registered(self):
        if self.keypair.ss58_address not in self.metagraph.nodes:
            raise UnrecoverableError(f"Hotkey {self.keypair.ss58_address} is not registered in {self.metagraph.netuid}")

    @property
    def state_path(self):
        full_path = (
            Path.home() /
            ".bittensor" /
            "miners" /
            WALLET_NAME /
            HOTKEY_NAME /
            f"netuid{self.metagraph.netuid}" /
            "validator"
        )

        makedirs(full_path, exist_ok=True)

        return full_path / "state.npz"

    def save_state(self):
        logger.info("save_state")

        np.savez(
            self.state_path,
            step=self.step,
            current_row=self.current_row,
            center_column=self.center_column,
            valid_miners=self.valid_miners,
            hotkeys=self.hotkeys,
        )

    def load_state(self):
        logger.info("load_state")

        path = self.state_path

        if not exists(path):
            return

        state = np.load(path)

        self.step = state["step"]
        self.current_row = state["current_row"]
        self.center_column = state["center_column"]
        self.valid_miners = list(state["valid_miners"])
        self.hotkeys = state["hotkeys"]

    async def sync(self, block: Block | None = None):
        logger.info("sync")

        self.metagraph.sync_nodes()

        if not block:
            block = self.substrate.get_block_number(None)

        self.last_metagraph_sync = block

        self.check_registered()

        self.valid_miners.clear()

        if len(self.hotkeys) != len(self.metagraph.nodes):
            new_scores = [0.0] * len(self.metagraph.nodes)
            length = len(self.hotkeys)
            new_scores[:length] = self.scores[:length]

            self.scores = new_scores

        nodes = self.nodes_list()

        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey not in self.metagraph.nodes:
                self.scores[uid] = 0.0

        self.hotkeys = [node.hotkey for node in nodes]

        logger.info("Checking active miners")

        current_steps = await asyncio.gather(*(
            self.make_request(node, "current_step")
            for node in self.metagraph.nodes.values()
        ))

        current_steps = [
            step_response.json() if step_response else 0
            for step_response, _ in current_steps
        ]

        # TODO implement system where validator keeps track of miner steps
        self.valid_miners.extend(*np.where(current_steps == np.uint64(1)))

        if block - self.metagraph.nodes[self.keypair.ss58_address].last_updated >= EPOCH_LENGTH:
            self.set_weights()

    def set_weights(self):
        logger.info("set_weights")

        if sum(self.scores) <= 0.0:
            weights = [1.0] * len(self.metagraph.nodes)
        else:
            weights = self.scores

        set_node_weights(
            self.substrate,
            self.keypair,
            node_ids=list(range(len(self.metagraph.nodes))),
            node_weights=weights,
            netuid=self.metagraph.netuid,
            validator_node_id=self.hotkeys.index(self.keypair.ss58_address),
            version_key=WEIGHTS_VERSION,
        )

    def new_wandb_run(self, block: Block | None = None):
        if not block:
            block = self.substrate.get_block_number(None)  # type: ignore

        self.wandb_run = wandb.init(
            name=f"validator-{self.metagraph.netuid}-{self.step}-{block}",
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            tags=[
                f"version_{VALIDATOR_VERSION_STRING}",
            ],
            config={
                "version": VALIDATOR_VERSION_STRING,
                "hotkey": self.keypair.ss58_address,
                "uid": self.metagraph.nodes[self.keypair.ss58_address].node_id,
            },
        )

        self.last_wandb_run = block

    def check_wandb_run(self):
        if not self.wandb_run:
            return

        self.wandb_run.log(
            {
                "center_column": self.center_column.tolist(),
                "current_evolution": self.current_row.tolist(),
            },
            step=self.step.item(),
        )

        current_block = self.substrate.get_block_number(None)  # type: ignore

        if current_block - self.last_wandb_run > WANDB_REFRESH_INTERVAL:
            self.new_wandb_run(current_block)

    async def make_request(self, node: Node, endpoint: str, payload: BaseModel | None = None):
        miner_address = f"http://{node.ip}:{node.port}"

        try:
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
                payload=payload.model_dump(),
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
        logger.info(f"Evolution step {self.step}")

        current_block = self.substrate.get_block_number(None)  # type: ignore

        elapsed_blocks = current_block - self.last_metagraph_sync

        if elapsed_blocks >= EPOCH_LENGTH:
            await self.sync(current_block)

            elapsed_blocks = 0

        while not len(self.valid_miners):
            logger.info(f"Not enough miners to compute step, waiting until next sync")

            sleep(EPOCH_LENGTH - elapsed_blocks)

            current_block = self.substrate.get_block_number(None)  # type: ignore

            await self.sync(current_block)

        iterator = iter(self.current_row)

        chunk_size = ceil(len(self.current_row) // len(self.valid_miners))

        nodes = self.nodes_list()

        chunks = [
            (nodes[uid], list(islice(iterator, chunk_size)))
            for uid in self.valid_miners
        ]

        responses: Iterable[tuple[Response, float]] = await asyncio.gather(*(
            self.make_request(
                node,
                "compute",
                ComputationData(parts=chunk),
            )
            for node, chunk in chunks
        ))

        responses: list[tuple[int, ComputationData | None, float]] = [
            (uid, ComputationData.model_validate_json(response.text) if response else None, inference_time)
            for uid, (response, inference_time) in zip(self.valid_miners, responses)
        ]

        for (uid, response, inference_time) in responses:
            if not response:
                self.scores[uid] = 0.0
            else:
                self.scores[uid] = 1 / inference_time

        data = [
            np.array(response.parts, dtype=np.uint64)
            for _, response, _ in responses
        ]

        self.current_row = self.normalize_response_data(data)

        bit_index = self.step % 64
        current_row_part = self.current_row[self.step // 64]

        self.center_column[-1] |= current_row_part >> bit_index << bit_index

        self.step += 1

        self.save_state()

        self.check_wandb_run()

    async def run(self):
        await self.sync()

        self.new_wandb_run()

        while True:
            try:
                await self.do_step()
            except (UnrecoverableError, KeyboardInterrupt):
                raise
            except:
                logger.error(f"Error in evolution step {self.step}", exc_info=True)

    def normalize_response_data(self, outputs: list[NDArray[np.uint64]]) -> NDArray[np.uint64]:
        def rule_30(a: np.uint64) -> np.uint64:
            return np.uint64(a ^ (a << np.uint64(1) | a << np.uint64(2)))

        def normalize_pair(a: np.uint64, b: np.uint64) -> tuple[np.uint64, np.uint64]:
            carry = np.uint64(a & np.uint64(1))

            a = np.uint64(a >> np.uint64(1))
            b = np.uint64((carry << np.uint64(63)) | b)

            a = rule_30(a)
            b = rule_30(b)

            msb = np.uint64(b >> np.uint64(63))

            b = np.uint64(b & np.uint64((1 << 63) - 1))

            a = np.uint64((a << np.uint64(1)) | msb)

            return a, b

        normalized_outputs = []

        i = 0

        for i in range(len(outputs) - 1): 
            a = np.uint64(outputs[i][-1])
            b = np.uint64(outputs[i + 1][0])

            a, b = normalize_pair(a, b)

            outputs[i][-1] = a
            outputs[i + 1][0] = b

        normalized_outputs.extend(outputs[i].astype(np.uint64))

        if i:
            normalized_outputs.extend(outputs[-1].astype(np.uint64))

        return np.array(normalized_outputs, dtype=np.uint64)


async def run():
    await Validator().run()


def main():
    asyncio.run(run())

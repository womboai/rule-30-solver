from pydantic import BaseModel


class ComputationData(BaseModel):
    parts: list[int]

import json
from typing import List
from .struct import Struct


class Tag(Struct):

    def __init__(self, tag: str) -> None:
        super().__init__()
        self._tag = tag

    def to_json(self) -> str:
        return json.dumps({'tags': self._tag}, ensure_ascii=False)

    @classmethod
    def parse(cls, json_str: str) -> "Tag":
        json_obj = json.loads(json_str)
        return cls(json_obj['tags'])

    @classmethod
    def load(cls, file_path: str) -> List["Tag"]:
        return super().load(file_path)

    def get_label(self) -> str:
        return self._tag
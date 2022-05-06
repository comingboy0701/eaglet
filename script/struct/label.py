import json
from typing import List, Union
from .struct import Struct


class Label(Struct):

    def __init__(self, label: Union[str, dict]) -> None:
        super().__init__()
        self._label = label

    def to_json(self) -> str:
        return json.dumps({'label': self._label}, ensure_ascii=False)

    @classmethod
    def parse(cls, json_str: str) -> "Label":
        json_obj = json.loads(json_str)
        return cls(json_obj['label'])

    @classmethod
    def load(cls, file_path: str) -> List["Label"]:
        return super().load(file_path)

    def get_label(self) -> Union[str, dict]:
        return self._label

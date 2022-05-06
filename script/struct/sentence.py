from typing import List
import json
from .struct import Struct


class Sentence(Struct):

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def to_json(self) -> str:
        return json.dumps({'text': self._text}, ensure_ascii=False)

    @classmethod
    def parse(cls, json_str: str) -> "Sentence":
        json_dict = json.loads(json_str)
        return cls(json_dict['text'])

    @classmethod
    def load(cls, file_path: str) -> List["Sentence"]:
        return super().load(file_path)

    def get_text(self) -> List[str]:
        return self._text

    @property
    def text(self):
        return self._text

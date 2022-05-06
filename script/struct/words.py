from typing import List
import json
from .struct import Struct


class Words(Struct):

    def __init__(self, words: List[str]) -> None:
        super().__init__()
        self._words = words

    def to_json(self) -> str:
        return json.dumps({'words': self._words}, ensure_ascii=False)

    @classmethod
    def parse(cls, json_str: str) -> "Word":
        json_dict = json.loads(json_str)
        return cls(json_dict['words'])

    @classmethod
    def load(cls, file_path: str) -> List["Word"]:
        return super().load(file_path)

    def get_words(self) -> List[str]:
        return self._words

    @property
    def words(self):
        return self._words

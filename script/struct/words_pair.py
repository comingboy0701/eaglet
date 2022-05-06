import json
from .words import Words
from .struct import Struct


class WordsPair(Struct):

    def __init__(self, words1: Words, words2: Words):
        super().__init__()
        self._words1 = words1
        self._words2 = words2

    def to_json(self) -> str:
        return json.dumps({
            "words_1": self._words1.get_words(),
            "words_2": self._words2.get_words()
        }, ensure_ascii=False)

    @classmethod
    def parse(cls, json_str: str) -> "WordsPair":
        json_obj = json.loads(json_str)
        s1 = Words(json_obj['words_1'])
        s2 = Words(json_obj['words_2'])
        return WordsPair(s1, s2)

    def get_words1(self) -> Words:
        return self._words1

    def get_words2(self) -> Words:
        return self._words2

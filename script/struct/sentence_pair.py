import json
from .sentence import Sentence
from .struct import Struct


class SentencePair(Struct):

    def __init__(self, sentence1: Sentence, sentence2: Sentence):
        super().__init__()
        self._sentence1 = sentence1
        self._sentence2 = sentence2

    def to_json(self) -> str:
        return json.dumps({
            "text_1": self._sentence1.get_text(),
            "text_2": self._sentence2.get_text()
        }, ensure_ascii=False)

    @classmethod
    def parse(cls, json_str: str) -> "SentencePair":
        json_obj = json.loads(json_str)
        s1 = Sentence(json_obj['text_1'])
        s2 = Sentence(json_obj['text_2'])
        return SentencePair(s1, s2)

    def get_sentence1(self) -> Sentence:
        return self._sentence1

    def get_sentence2(self) -> Sentence:
        return self._sentence2

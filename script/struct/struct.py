from typing import List, Any


class Struct:

    def to_json(self) -> str:
        raise NotImplementedError()

    @classmethod
    def parse(cls, json_str: str) -> "Struct":
        raise NotImplementedError()

    @classmethod
    def load(cls, file_path: str) -> List[Any]:
        result = []
        sentence_file = open(file_path, encoding='utf-8')
        for line in sentence_file:
            line = line.strip()
            result.append(cls.parse(line))
        sentence_file.close()
        return result


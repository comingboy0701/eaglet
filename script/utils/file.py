import json
import pickle
from typing import Any, List


def save_json(obj: Any, file_path: str, **kwargs):
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=kwargs.get('ensure_ascii', False), indent=kwargs.get('indent', 2))


def load_json(file_path: str) -> Any:
    with open(file_path, "r", encoding='utf-8') as f:
        return json.load(f)


def save_pickle(obj: Any, file_path: str) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_path: str) -> Any:
    with open(file_path, "rb") as f:
        resource = pickle.load(f)
    return resource


def load_line_json(file_path: str) -> List[dict]:
    result = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            result.append(json.loads(line))
    return result


def persist_line_json(obj: List[dict], file_path: str, **kwargs) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        for s in obj:
            s = json.dumps(s, ensure_ascii=kwargs.get('ensure_ascii', False), indent=kwargs.get('indent')) + "\n"
            f.write(s)
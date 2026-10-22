from pathlib import Path
import json


def get_index_dir(project_id, document_id, index_version):
    return Path("data") / "indexes" / project_id / document_id / index_version


def save_project(project):
    project_dir = Path("data") / "projects" / project["project_id"]
    project_dir.mkdir(parents=True, exist_ok=True)

    with (project_dir / "project.json").open(
        "w", encoding="utf-8"
    ) as file:
        # 使用 json.dump，把 project 写入 file
        # 设置 ensure_ascii=False、indent=2
        json.dump(project, file, ensure_ascii=False, indent=2)

def load_project(project_id):
    project_path = Path("data") / "projects" / project_id / "project.json"

    with project_path.open("r", encoding="utf-8") as file:
        # 读取 file 中的 JSON，并返回解析后的 Python 对象
        return json.load(file)
from pathlib import Path
from uuid import uuid4
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

def list_projects():
    project_root = Path("data") / "projects"
    projects = []

    for project_path in sorted(project_root.glob("*/project.json")):
        project_id = project_path.parent.name

        # 调用已有的 load_project() 读取这个项目
        # 把读到的项目字典加入 projects 列表
        project = load_project(project_id)
        projects.append(project)

    return projects



def create_project(name):
    name = name.strip()

    if not name:
        raise ValueError("项目名称不能为空")

    project = {
        "project_id": f"project_{uuid4().hex}",
        "name": name,
        # 补上 documents 字段，新项目还没有任何文档
        "documents": [],
    }

    # 调用已有的 save_project() 保存项目
    save_project(project)

    # 返回新建的项目字典
    return project

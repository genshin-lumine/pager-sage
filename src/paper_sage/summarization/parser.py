"""总结结果的结构与来源标识校验；不判断原文是否支持要点。"""

import json


SUMMARY_FIELDS = ("research_question", "method", "results", "limitations")
SUMMARY_LABELS = {
    "research_question": "研究问题", "method": "方法",
    "results": "结果", "limitations": "局限",
}

# 只拦截已经观察到的占位正文，不把它当成通用的事实核验器。
KNOWN_PLACEHOLDERS = {"中文要点", "用本批次原文支持的方法要点替换此占位文字"}


def reject_summary_placeholders(summary):
    """接收已通过结构校验的结果；真实要点是否有原文支持仍需另行核对。"""
    for field in SUMMARY_FIELDS:
        for index, point in enumerate(summary[field]):
            text = point["text"].strip().rstrip("。.!！ ")
            if text in KNOWN_PLACEHOLDERS:
                raise ValueError(f"{field}[{index}].text 是占位文字，不能作为论文要点：{point['text']}")


def parse_batch_summary(raw_response, batch):
    """模型只返回要点和引用标识，页码、来源、原文由程序保留。"""
    if not isinstance(raw_response, str):
        raise ValueError("批次总结必须是 JSON 字符串")
    summary = json.loads(raw_response)
    if not isinstance(summary, dict) or set(summary) != set(SUMMARY_FIELDS):
        raise ValueError("批次总结必须且只能包含 research_question、method、results、limitations")
    if not isinstance(batch, dict) or not isinstance(batch.get("parts"), list) or not batch["parts"]:
        raise ValueError("需要本批次的原始文字段")
    allowed_part_ids = set()
    for part in batch["parts"]:
        if not isinstance(part, dict) or not isinstance(part.get("part_id"), str) or not part["part_id"].strip():
            raise ValueError("原始文字段缺少有效标识")
        if part["part_id"] in allowed_part_ids:
            raise ValueError("原始文字段标识重复")
        allowed_part_ids.add(part["part_id"])

    for field in SUMMARY_FIELDS:
        points = summary[field]
        if not isinstance(points, list):
            raise ValueError(f"{field} 必须是列表")
        for point_index, point in enumerate(points):
            location = f"{field}[{point_index}]"
            if not isinstance(point, dict):
                raise ValueError(f"{location} 必须是包含 text 和 part_ids 的对象，实际类型：{type(point).__name__}")
            if set(point) != {"text", "part_ids"}:
                missing = sorted({"text", "part_ids"} - set(point))
                extra = sorted(set(point) - {"text", "part_ids"})
                raise ValueError(
                    f"{location} 每条要点必须且只能包含 text 和 part_ids；"
                    f"缺少字段：{missing}；多余字段：{extra}；实际字段：{sorted(point)}"
                )
            if not isinstance(point["text"], str) or not point["text"].strip():
                raise ValueError(f"{location}.text 要点正文必须是非空字符串")
            part_ids = point["part_ids"]
            if not isinstance(part_ids, list) or not part_ids:
                raise ValueError(f"{location} 每条要点必须提供非空的 part_ids 列表")
            for part_id in part_ids:
                if not isinstance(part_id, str) or not part_id.strip():
                    raise ValueError("引用的文字段标识必须是非空字符串")
                # 判断 part_id 是否属于当前批次允许引用的 allowed_part_ids。
                part_is_allowed = True if part_id in allowed_part_ids else False
                if type(part_is_allowed) is not bool:
                    raise ValueError("文字段归属判断必须返回布尔值")
                if not part_is_allowed:
                    raise ValueError(f"引用的文字段不属于当前批次：{part_id}")
            if len(set(part_ids)) != len(part_ids):
                raise ValueError("同一条要点不能重复引用同一文字段")
    return summary

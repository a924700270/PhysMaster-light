"""Generate pipeline visualization HTML for the single-branch PhysMaster workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TEMPLATE_FILE = Path(__file__).resolve().parent.parent / "utils/visualization_template.html"
INJECT_MARKER = "<!-- __DATA_INJECT__ -->"


def _safe_short(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return text[:half] + "\n... [truncated] ...\n" + text[-half:]


def _serialize_trajectory(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the linear trajectory list into visualization node dicts with parent/child chain."""
    nodes_payload: list[dict[str, Any]] = []

    for idx, node in enumerate(trajectory):
        node_id = int(node.get("node_id", idx + 1))
        parent_id = int(trajectory[idx - 1]["node_id"]) if idx > 0 else None
        children = [int(trajectory[idx + 1]["node_id"])] if idx + 1 < len(trajectory) else []

        subtask = node.get("subtask")
        if subtask is None:
            subtask = {
                "id": node.get("subtask_id"),
                "description": node.get("description", ""),
            }

        evaluation = node.get("evaluation") or node.get("critic_feedback") or {}
        reward = float(node.get("reward", 0.0) or 0.0)

        nodes_payload.append(
            {
                "node_id": node_id,
                "parent_id": parent_id,
                "children": children,
                "depth": idx,
                "node_type": node.get("node_type", "draft"),
                "subtask": subtask,
                "description": _safe_short(node.get("description", ""), 4000),
                "reward": reward,
                "visits": 1,
                "status": "completed",
                "created_by": "supervisor",
                "memory": _safe_short(node.get("memory", ""), 4000),
                "theoretician_output": _safe_short(node.get("theoretician_output", ""), 48000),
                "supervisor_dispatch": node.get("supervisor_dispatch") or {},
                "critic_feedback": evaluation,
                "supervisor_feedback": node.get("supervisor_feedback") or {},
                "selected_round": node.get("selected_round"),
            }
        )
    return nodes_payload


def _compute_chain_layout(nodes: list[dict[str, Any]]) -> dict[int, tuple[float, float]]:
    """Compute layout for a single-branch chain: all nodes centered horizontally."""
    if not nodes:
        return {}
    total = len(nodes)
    coords: dict[int, tuple[float, float]] = {}
    for i, node in enumerate(nodes):
        x = 0.5
        y = 0.08 + (0.84 * (i / max(1, total - 1))) if total > 1 else 0.5
        coords[int(node["node_id"])] = (x, y)
    return coords


def build_payload(
    *,
    nodes: list[dict[str, Any]],
    task_description: str,
    subtasks: Any = None,
    summary: str = "",
) -> dict[str, Any]:
    coords = _compute_chain_layout(nodes)
    root_id = int(nodes[0]["node_id"]) if nodes else 0

    edges: list[list[int]] = []
    for node in nodes:
        for child_id in node.get("children", []):
            if child_id in coords and node["node_id"] in coords:
                edges.append([int(node["node_id"]), int(child_id)])

    payload_nodes = []
    for node in nodes:
        x, y = coords.get(node["node_id"], (0.5, 0.5))
        copied = dict(node)
        copied["viz_x"] = x
        copied["viz_y"] = y
        payload_nodes.append(copied)

    return {
        "task_description": task_description,
        "subtasks": subtasks if subtasks is not None else [],
        "summary": summary or "",
        "root_id": root_id,
        "nodes": payload_nodes,
        "edges": edges,
    }


def build_html(
    *,
    nodes: list[dict[str, Any]],
    task_description: str,
    subtasks: Any = None,
    summary: str = "",
) -> str:
    payload = build_payload(
        nodes=nodes,
        task_description=task_description,
        subtasks=subtasks,
        summary=summary,
    )
    payload_json = json.dumps(payload, ensure_ascii=False)

    template = TEMPLATE_FILE.read_text(encoding="utf-8")
    inject = f"<script>window.__PHY_MCTS_DATA__ = {payload_json};</script>"
    return template.replace(INJECT_MARKER, inject)


def write_html(
    output_path: str | Path,
    *,
    nodes: list[dict[str, Any]],
    task_description: str = "",
    subtasks: Any = None,
    summary: str = "",
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(
        nodes=nodes,
        task_description=task_description,
        subtasks=subtasks,
        summary=summary,
    )
    out.write_text(html, encoding="utf-8")
    return out


def generate_vis(
    output_path: str | Path,
    trajectory: list[dict[str, Any]],
    task_description: str = "",
    subtasks: Any = None,
    summary: str = "",
) -> Path:
    """Generate visualization HTML from a linear trajectory list.

    Args:
        output_path: Where to write the HTML file.
        trajectory: List of node dicts from SupervisorOrchestrator.run()["trajectory"].
        task_description: The task description text.
        subtasks: List of subtask dicts.
        summary: The summary markdown text.
    """
    nodes = _serialize_trajectory(trajectory)
    return write_html(
        output_path=output_path,
        nodes=nodes,
        task_description=task_description,
        subtasks=subtasks if subtasks is not None else [],
        summary=summary,
    )

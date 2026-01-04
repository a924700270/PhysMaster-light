import os
import re
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable
from pathlib import Path

from .mcts_tree import MCTSTree
from .mcts_node import MCTSNode
from .theoretician import run_theo_node

from utils.gpt5_utils import call_model
from utils.save_utils import MarkdownWriter

try:
    from knowledge_base.global_store import GlobalKnowledgeBase
    from knowledge_base.local_store import LocalKnowledgeBase
except Exception:
    GlobalKnowledgeBase = object
    LocalKnowledgeBase = object

_GLOBAL_POOL: ProcessPoolExecutor | None = None

SEARCH_KB_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": (
                "Search across FOUR knowledge sources and return short snippets:\n"
                "1) knowledge_base/local_knowledge_base (core knowledge JSON)\n"
                "2) knowledge_base/global_knowledge_base (core knowledge JSON)\n"
                "3) knowledge_base/global_prior (prior outputs)\n"
                "4) knowledge_base/global_methodology (manual methodology notes, .md/.txt)"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (keywords / question / subtask description)."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": 5
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Subset of sources to search. Allowed values: "
                            "local, global, prior, methodology. "
                            "If omitted, search all."
                        )
                    },
                    "refresh": {
                        "type": "boolean",
                        "description": "Force rebuild disk index before searching (use when files changed).",
                        "default": False
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]




def _init_worker():
    """Worker initializer: 每个子进程只初始化一次"""
    global _WORKER_INIT
    if "_WORKER_INIT" in globals():
        return
    _WORKER_INIT = True


_DISK_KB_INDEX = None  # lazy-built: list[dict]

def _kb_roots() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    kb_root = repo_root() / "knowledge_base"
    return {
        "local": kb_root / "local_library",
        "global": kb_root / "global_library",
        "prior": kb_root / "global_prior",
        "methodology": kb_root / "global_methodology"
    }


def _tokenize_query(q: str) -> list[str]:
    q = (q or "").strip().lower()
    if not q:
        return []
    # English tokens + Chinese blocks
    toks = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", q)
    # drop ultra-short latin tokens to reduce noise
    cleaned = []
    for t in toks:
        if re.fullmatch(r"[a-z0-9_]+", t) and len(t) <= 2:
            continue
        cleaned.append(t)
    return cleaned


def _extract_search_text(obj, *, max_chars: int = 20000) -> str:
    """
    Best-effort extraction:
    - prioritize title/abstract/core_knowledge(qualitative/quantitative)
    - recursively collect strings, but skip obviously huge id lists
    """
    parts: list[str] = []

    def add(s: str):
        if not s:
            return
        s = str(s).strip()
        if not s:
            return
        parts.append(s)

    def walk(x, depth: int = 0):
        if len(" ".join(parts)) > max_chars:
            return
        if depth > 8:
            return
        if isinstance(x, str):
            add(x)
            return
        if isinstance(x, (int, float, bool)) or x is None:
            return
        if isinstance(x, list):
            # skip huge numeric/id lists
            if len(x) > 500 and all(isinstance(v, (int, float, str)) for v in x):
                return
            for v in x[:200]:
                walk(v, depth + 1)
            return
        if isinstance(x, dict):
            # priority keys first
            for k in ["title", "abstract", "core_knowledge", "qualitative", "quantitative", "summary", "takeaways"]:
                if k in x:
                    walk(x.get(k), depth + 1)
            # then everything else except obvious junk
            for k, v in x.items():
                if k in ("touch_ids", "ids", "references", "bib", "citations_list"):
                    continue
                if k in ("title", "abstract", "core_knowledge", "qualitative", "quantitative", "summary", "takeaways"):
                    continue
                walk(v, depth + 1)

    walk(obj, 0)
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def _iter_files(root: Path, exts: tuple[str, ...]) -> Iterable[Path]:
    if not root or not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _build_disk_index(force: bool = False) -> list[dict]:
    global _DISK_KB_INDEX
    if _DISK_KB_INDEX is not None and not force:
        return _DISK_KB_INDEX

    roots = _kb_roots()
    index: list[dict] = []

    # 1) local/global/prior: JSON
    for source in ("local", "global", "prior"):
        root = roots.get(source)
        if not root or not root.exists():
            continue
        for fp in _iter_files(root, (".json",)):
            try:
                with fp.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            # If it contains extra.recall_papers/top_papers, index each candidate for finer granularity
            candidates = []
            if isinstance(data, dict):
                extra = data.get("extra") or {}
                if isinstance(extra, dict):
                    rp = extra.get("recall_papers") or extra.get("top_papers")
                    if isinstance(rp, list):
                        candidates = [c for c in rp if isinstance(c, dict)]

            if candidates:
                for i, c in enumerate(candidates[:200]):
                    title = c.get("title", "") or data.get("title", "")
                    text = _extract_search_text(c)
                    if not text and title:
                        text = title
                    index.append(
                        {
                            "source": source,
                            "path": f"{fp.as_posix()}#cand{i}",
                            "title": title,
                            "text": text,
                            "text_lower": (text or "").lower(),
                        }
                    )
            else:
                title = data.get("title", "") if isinstance(data, dict) else ""
                text = _extract_search_text(data)
                if not text and title:
                    text = title
                index.append(
                    {
                        "source": source,
                        "path": fp.as_posix(),
                        "title": title,
                        "text": text,
                        "text_lower": (text or "").lower(),
                    }
                )

    # 2) methodology: manual notes (.md/.txt), fallback legacy global_case
    meth_root = roots.get("methodology")
    legacy_root = roots.get("case_legacy")
    chosen = meth_root if (meth_root and meth_root.exists()) else legacy_root

    if chosen and chosen.exists():
        for fp in _iter_files(chosen, (".md", ".txt", ".json")):
            try:
                if fp.suffix.lower() == ".json":
                    with fp.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    title = (data.get("title") if isinstance(data, dict) else "") or fp.stem
                    text = _extract_search_text(data)
                else:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                    title = fp.stem
                text = (text or "")[:20000]
            except Exception:
                continue

            index.append(
                {
                    "source": "methodology",
                    "path": fp.as_posix(),
                    "title": title,
                    "text": text,
                    "text_lower": (text or "").lower(),
                }
            )

    _DISK_KB_INDEX = index
    return _DISK_KB_INDEX


def _make_excerpt(text: str, q_lower: str, tokens: list[str], *, window: int = 220) -> str:
    if not text:
        return ""
    t_lower = text.lower()
    pos = -1
    if q_lower:
        pos = t_lower.find(q_lower)
    if pos < 0:
        for tok in tokens:
            pos = t_lower.find(tok)
            if pos >= 0:
                break
    if pos < 0:
        return (text[:window] + "...") if len(text) > window else text

    start = max(0, pos - window)
    end = min(len(text), pos + window)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def _disk_search(query: str, top_k: int, sources: list[str] | None, refresh: bool) -> list[str]:
    q = (query or "").strip()
    if not q:
        return []

    allowed = {"local", "global", "prior", "methodology"}
    if sources:
        scope = [s for s in sources if s in allowed]
        if not scope:
            scope = list(allowed)
    else:
        scope = list(allowed)

    index = _build_disk_index(force=bool(refresh))

    q_lower = q.lower()
    tokens = _tokenize_query(q)

    scored = []
    for item in index:
        if item.get("source") not in scope:
            continue
        text_lower = item.get("text_lower") or ""
        if not text_lower:
            continue
        phrase = text_lower.count(q_lower) if q_lower else 0
        token_hits = sum(text_lower.count(t) for t in tokens) if tokens else 0
        score = phrase * 10 + token_hits
        if score <= 0:
            continue
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, item in scored[: max(1, top_k)]:
        title = item.get("title") or ""
        path = item.get("path") or ""
        excerpt = _make_excerpt(item.get("text") or "", q_lower, tokens)
        out.append(f"[{item.get('source')}] {title}\n[path] {path}\n[score] {score}\n{excerpt}")
    return out



class SupervisorOrchestrator:
    """MCTS Supervisor，负责完整的树搜索流程"""

    def __init__(
        self,
        structured_problem,
        local_kb: LocalKnowledgeBase,
        global_kb: GlobalKnowledgeBase,
        task_dir: str,
        processes: int = 2,
        max_nodes: int = 4,
        scheduler_path: str = "prompts/supervisor-scheduler_prompt.txt",
        critic_path: str = "prompts/supervisor-critic_prompt.txt",

        draft_expansion: int = 2,
        revise_expansion: int = 2,
        improve_expansion: int = 1,
        exploration_constant: float = 1.414,
        complete_score_threshold: float = 0.9,
    ):
        self.structured_problem = structured_problem
        self.local_kb = local_kb
        self.global_kb = global_kb
        self.task_dir = task_dir
        self.processes = max(1, processes)
        self.max_nodes = max_nodes
        self.scheduler_path = scheduler_path
        self.critic_path = critic_path
        self.draft_expansion = draft_expansion
        self.revise_expansion = revise_expansion
        self.improve_expansion = improve_expansion
        self.exploration_constant = exploration_constant
        self.complete_score_threshold = float(complete_score_threshold)

        try:
            with open(scheduler_path, "r", encoding="utf-8") as f:
                self.scheduler_prompt = f.read()
        except Exception:
            self.scheduler_prompt = ""

        try:
            with open(critic_path, "r", encoding="utf-8") as f:
                self.critic_prompt = f.read()
        except Exception:
            self.critic_prompt = ""

        self.subtasks = self._build_subtasks()

        self.tree = MCTSTree(
            root_subtask_id=0,
            root_description="Virtual Root",
        )
        self.tree.root.node_type = "virtual"
        self.tree.root.subtask_description = "Virtual Root"
        self.tree.root.subtask_type = "virtual"
        self.tree.root.status = "completed"
        self.tree.root.evaluation = {"decision": "complete", "score": 0}
        self.tree.root.total_reward = 0
        self.tree.root.average_reward = 0
        self.tree.root.visits = 1

        self.node_counter = 1

        global _GLOBAL_POOL
        if _GLOBAL_POOL is None:
            mp.set_start_method("spawn", force=True)
            _GLOBAL_POOL = ProcessPoolExecutor(
                max_workers=self.processes,
                initializer=_init_worker,
            )

    def _search_kb(self, query: str, top_k: int = 5, sources=None, refresh: bool = False) -> str:
        """
        Search KB across:
        - disk local/global/prior/methodology (primary)
        - legacy self.local_kb/self.global_kb interfaces (fallback/extra)
        """
        top_k = int(top_k or 5)
        if top_k <= 0:
            top_k = 5

        # 1) Primary: disk search across four KBs
        disk_results = []
        try:
            disk_results = _disk_search(query=query, top_k=top_k, sources=sources, refresh=refresh) or []
        except Exception as e:
            disk_results = [f"[disk_search error: {e}]"]

        # 2) Extra: legacy KB interfaces (your current local_store/global_store)
        legacy_results = []
        if hasattr(self.local_kb, "search"):
            try:
                legacy_results.extend(self.local_kb.search(query, top_k=top_k) or [])
            except Exception as e:
                legacy_results.append(f"[local_kb.search error: {e}]")
        elif hasattr(self.local_kb, "to_brief"):
            try:
                legacy_results.extend(self.local_kb.to_brief(n=top_k) or [])
            except Exception as e:
                legacy_results.append(f"[local_kb.to_brief error: {e}]")

        if hasattr(self.global_kb, "search"):
            try:
                legacy_results.extend(self.global_kb.search(query, top_k=top_k) or [])
            except Exception as e:
                legacy_results.append(f"[global_kb.search error: {e}]")
        elif hasattr(self.global_kb, "to_brief"):
            try:
                legacy_results.extend(self.global_kb.to_brief(n=top_k) or [])
            except Exception as e:
                legacy_results.append(f"[global_kb.to_brief error: {e}]")

        # Merge + de-dup (disk first)
        merged = []
        seen = set()

        for r in (disk_results + legacy_results):
            if not r:
                continue
            s = str(r)
            if s in seen:
                continue
            seen.add(s)
            merged.append(s)

        if not merged:
            return f"[KB] No entries found for query: {query}"

        lines = [f"[KB] Search results for: {query}"]
        for i, txt in enumerate(merged[:top_k], 1):
            lines.append(f"{i}. {txt}")
        return "\n".join(lines)


    def run(self) -> Dict[str, Any]:

        total_nodes_created = 1 
        completed_subtasks: List[Dict[str, Any]] = []

        while total_nodes_created < self.max_nodes:
            selected_node = self._select_leaf_node()
            if selected_node is None:
                break
            
            scheduler_output: str = None
            try:
                scheduler_output = self._call_scheduler(selected_node) # [subtask_description, background_knowledge]
            except Exception:
                scheduler_output = ""
                print("Scheduler call failed.")

            evaluation = getattr(selected_node, "evaluation", "")
            decision = (evaluation.get("decision") or "to_revise").lower()
            score = float(evaluation.get("score", 0.0) or 0.0)

            expansion_count = self._get_expansion_count(decision)
            child_node_type = self._decision_to_node_type(decision)
            next_subtask_id = selected_node.subtask_id
            is_complete = (decision == "complete") and (score >= self.complete_score_threshold)

            if selected_node.node_type == "virtual":
                next_subtask_id = self.subtasks[0]["id"] if self.subtasks else 1
                child_node_type = "draft"
                expansion_count = self.draft_expansion

            elif is_complete:
                completed_subtasks.append(
                    {
                        "subtask_id": selected_node.subtask_id,
                        "description": selected_node.subtask_description,
                        "result": getattr(selected_node, "result", None),
                        "score": score,
                        "log_path": getattr(selected_node, "log_path", None),
                        "best_node_index": selected_node.node_index,
                    }
                )
                next_subtask = self._get_next_subtask(selected_node.subtask_id)
                if next_subtask:
                    next_subtask_id = next_subtask["id"]
                    child_node_type = "draft"
                    expansion_count = self.draft_expansion
                else:
                    break

            elif decision == "to_redraft":
                child_node_type = "draft"
            elif decision == "to_revise":
                child_node_type = "revise"
            elif decision == "to_improve":
                child_node_type = "improve"
            else:
                child_node_type = "revise"

            new_nodes = self._expand_and_simulate_nodes(
                parent=selected_node,
                node_type=child_node_type,
                count=expansion_count,
                subtask_id=next_subtask_id,
                augmented_description=scheduler_output,
            )
            total_nodes_created += len(new_nodes)

            if total_nodes_created >= self.max_nodes:
                break

        best_trajectory = self._find_best_trajectory()

        summary = {
            "completed_subtasks": completed_subtasks,
            "total_nodes": total_nodes_created,
            "tree_stats": self.tree.get_tree_stats(),
            "best_trajectory": best_trajectory,
        }

        return summary


    def _call_scheduler(self, node: MCTSNode) -> str:
        """调用 Supervisor-Scheduler，根据 structured_problem 给出subtasks及补充信息，并返回调度结果。"""
        if not self.scheduler_prompt:
            return None

        try:
            if node is not None:
                node_info = {
                    "node_index": node.node_index,
                    "subtask_id": node.subtask_id,
                    "node_type": node.node_type,
                    "subtask_description": getattr(node, "subtask_description", None),
                    "evaluation": getattr(node, "evaluation", None),
                    "result": getattr(node, "result", None),
                }
            else:
                node_info = None
        except Exception:
            node_info = None

        prompt = self.scheduler_prompt.format(
            structured=json.dumps(self.structured_problem, ensure_ascii=False, indent=2),
            node=json.dumps(node_info, ensure_ascii=False, indent=2),
        )

        tools = SEARCH_KB_TOOL
        #tool_functions = {
        #    "search_kb": lambda query, top_k=5: self._search_kb(query, top_k)
        #}

        tool_functions = {
            "search_kb": lambda query, top_k=5, sources=None, refresh=False: self._search_kb(
                query, top_k=top_k, sources=sources, refresh=refresh
            )
        }


        response = call_model(
            system_prompt=(
                "You are a theoretical physicist and project supervisor. "
                "Your task is to schedule the research problem into several well-defined sub-tasks and provide necessary background knowledge. "
                "If you require background knowledge, call the `search_kb` tool to retrieve relevant information."
                "Only call ONE tool at a time."
            ),
            user_prompt=prompt,
            tools=tools,
            tool_functions=tool_functions,
            model_name="gpt-5",
        )


        # markdown_writer.write_to_markdown( response + "\n", mode="supervisor_scheduler" )
        print("========== Supervisor-Scheduler ========== \n" + response + "\n")

        return response

    def _call_critic(self, node: MCTSNode) -> Dict[str, Any]:
        """
        Supervisor-Critic 评价节点结果
        Output: decision, score, summary, code
        """
        import json

        result_data = node.result or ""

        if isinstance(result_data, str):
            try:
                node_output = json.loads(result_data)
            except Exception:
                node_output = {}
        else:
            node_output = result_data or {}

        core_results = ""
        analysis = ""
        code = ""
        files: list[Any] = []

        if isinstance(node_output, dict):
            core_results = (
                node_output.get("core_results")
                or node_output.get("core_result")
                or ""
            )
            analysis = node_output.get("analysis") or ""
            code = node_output.get("code") or ""
            files = node_output.get("files") or []

        context_str = json.dumps(
            {
                "analysis": analysis,
                "code": code,
                "files": files,
            },
            ensure_ascii=False,
            indent=2,
        )

        prompt = self.critic_prompt.format(
            result=core_results,
            context=context_str,
        )

        markdown_writer = MarkdownWriter(
            problem=self.structured_problem.get("task_description", ""),
            topic=self.structured_problem.get("topic", ""),
            log_dir=Path(self.task_dir),
            depth=node.get_depth() if node else 0,
            node_index=node.node_index if node else 0,
            file_prefix=self._get_safe_name(),
        )

        system_prompt = (
            """
            As the supervisor, your task is to critically evaluate the node like a peer reviewer:
            1. Check whether the numerical results are reasonable and consistent with physical intuition and related knowledge.
            2. Check whether the proposed physical model and and mathmatical methodology are reasonable. Rigorously confirm the correctness of the equations, derivations, and approximations used. 
            3. Check whether the numerical methods and algorithms are appropriate and correctly implemented with code.
            4. Point out potential errors and gaps.
            5. Summrize the key points of the current solution as well as provide summmary and constructive opinion.

            When needed, you may call the `search_kb` tool to search for domain knowledge or relative papersthat are stored in the knowledge base to support your critique.
            Only call ONE tool at a time.
            """
        )

        tools = SEARCH_KB_TOOL

        tool_functions = {
            "search_kb": lambda query, top_k=5, sources=None, refresh=False: self._search_kb(
                query, top_k=top_k, sources=sources, refresh=refresh
            )
        }       



        response = call_model(
            system_prompt=system_prompt,
            user_prompt=prompt,
            tools=tools,
            tool_functions=tool_functions,
            markdown_writer=markdown_writer,
        )

        markdown_writer.write_to_markdown(
            response + "\n",
            mode="supervisor_critic",
        )

        print("========== Supervisor-Critic ========== \n" + response + "\n")

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return {
                "decision": "to_revise",
                "score": 0.05,
                "summary": "Critic failed to parse model output.",
                "opinion": "Invalid JSON from critic.",
                "code": code,
            }

        decision = str(parsed.get("decision", "to_revise")).strip().lower()
        score = parsed.get("score", 0.0)
        summary = parsed.get("summary") or ""
        opinion = parsed.get("opinion") or ""

        return {
            "decision": decision or "to_revise",
            "score": float(score) if score is not None else 0.0,
            "summary": summary,
            "opinion": opinion,
            "code": code,
        }

    

    def _expand_and_simulate_nodes(
        self,
        parent: MCTSNode,
        node_type: str,
        count: int,
        subtask_id: int,
        augmented_description: str = None,
        ) -> List[MCTSNode]:
        """
        扩展 & 并行模拟多个子节点：
        - 使用Scheduler 由structured_problem生成自然语言任务subtask_description
        - 批量调用 Critic 进行评分，并进行Backpropagation
        """
        new_nodes: List[MCTSNode] = []

        subtask_info = next(subtusk for subtusk in self.subtasks if subtusk["id"] == subtask_id)

        futures = []
        indices: List[int] = []
        global _GLOBAL_POOL

        for _ in range(count):
            node_index = self.node_counter
            self.node_counter += 1
            indices.append(node_index)

            parent_memory = ""
            if parent and getattr(parent, "evaluation", None):
                pe = parent.evaluation or {}
                parts = []
                if pe.get("summary"):
                    parts.append(pe["summary"])
                if pe.get("opinion"):
                    parts.append("opinion: " + str(pe["opinion"]))
                if pe.get("core_results"):
                    parts.append("Core results: " + str(pe["core_results"]))
                if pe.get("code"):
                    parts.append("Code: " + str(pe["code"]))
                parent_memory = "\n".join(parts)

            child_depth = (parent.get_depth() + 1) if parent else 0

            payload = {
                "depth": child_depth,
                "node_index": node_index,
                "node_type": node_type,
                "structured_problem": self.structured_problem,
                "subtask": {
                    "id": subtask_id,
                    "description": augmented_description,
                    "subtask_type": subtask_info.get("subtask_type", "reasoning"),
                    "input": subtask_info.get("input"),
                    "expected_output": subtask_info.get("expected_output"),
                },
                "task_dir": self.task_dir,
                "parent_memory": parent_memory,
            }

            futures.append(_GLOBAL_POOL.submit(run_theo_node, payload))

        wait(futures)

        outputs: List[Tuple[MCTSNode, Dict[str, Any]]] = []

        for idx, f in enumerate(futures):
            node_index = indices[idx]

            try:
                node_output = f.result()
            except Exception as e:
                failed_node = MCTSNode(
                    subtask_id=subtask_id,
                    node_index=node_index,
                    node_type=node_type,
                    subtask_description=augmented_description,
                    status="failed",
                )
                failed_node.result = {"error": str(e)}
                failed_node.evaluation = {"decision": "to_revise", "score": 0.0}
                self.tree.add_node(failed_node)
                parent.add_child(failed_node)
                try:
                    failed_node.backpropagate(0.0)
                except Exception:
                    failed_node.update_stats(0.0)
                new_nodes.append(failed_node)
                continue

            child_node = MCTSNode(
                subtask_id=subtask_id,
                node_index=node_index,
                node_type=node_type,
                subtask_description=augmented_description,
                status="completed",
            )
            child_node.result = node_output.get("result")
            child_node.log_path = node_output.get("log_path")
            child_node.subtask_type = subtask_info.get("subtask_type", "reasoning")
            child_node.input = subtask_info.get("input")
            child_node.expected_output = subtask_info.get("expected_output")

            self.tree.add_node(child_node)
            parent.add_child(child_node)

            new_nodes.append(child_node)
            outputs.append((child_node, node_output))

        for child_node, node_output in outputs:
            evaluation = self._call_critic(child_node)
            reward = evaluation.get("score", evaluation.get("reward", 0.0)) or 0.0
            child_node.evaluation = evaluation
            try:
                child_node.backpropagate(reward)
            except Exception:
                child_node.update_stats(reward)

        return new_nodes

    # Tools
    def _get_safe_name(self) -> str:
        instr_name = (
            self.structured_problem.get("instruction_filename")
            or self.structured_problem.get("topic")
        )
        try:
            instr_stem = Path(str(instr_name)).stem
        except Exception:
            instr_stem = str(instr_name)
        safe_name = "".join(
            c if (c.isalnum() or c in "._-") else "_" for c in str(instr_stem)
        )
        return safe_name
    
    def _get_expansion_count(self, decision: str) -> int:
        d = (decision or "").lower()
        if d in ("to_redraft", "complete"):
            return self.draft_expansion
        elif d == "to_revise":
            return self.revise_expansion
        elif d == "to_improve":
            return self.improve_expansion
        else:
            return 1

    def _decision_to_node_type(self, decision: str) -> str:
        d = (decision or "").lower()
        if d in ("to_redraft", "complete"):
            return "draft"
        if d == "to_revise":
            return "revise"
        if d == "to_improve":
            return "improve"
        return "revise"

    def _select_leaf_node(self) -> Optional[MCTSNode]:
        """使用 UCB1 策略从树中选择一个要扩展的节点。"""
        
        selected = self.tree.selection(self.exploration_constant)

        while selected.status == "completed" and selected.children:
            best_child = selected.select_best_child(self.exploration_constant)
            if best_child is None:
                break
            selected = best_child

        return selected

    def _build_subtasks(self) -> List[Dict[str, Any]]:
        """ 直接读取 structured_problem 中已有的 sub-tasks """
        subtasks_payload = self.structured_problem.get("sub-tasks", [])

        if not subtasks_payload:
            subtasks_payload = [
                {
                    "id": 1,
                    "description": self.structured_problem.get("task_description", ""),
                    "subtask_type": "reasoning",
                    "input": self.structured_problem.get("input", ""),
                    "expected_output": self.structured_problem.get("expected_output", ""),
                }
            ]
        return subtasks_payload

    def _get_next_subtask(self, current_subtask_id: int) -> Optional[Dict[str, Any]]:
        """获取下一个 subtask"""
        for subtask in self.subtasks:
            if subtask["id"] > current_subtask_id:
                return subtask
        return None

    def _find_best_trajectory(self) -> List[Dict[str, Any]]:
        """
        找到覆盖所有 subtasks 的一条最优路径，从虚拟根往下
        """
        best_path: List[MCTSNode] = []
        best_avg_reward = -float("inf")

        def dfs(node: MCTSNode, current_path: List[MCTSNode], current_subtask_id: int):
            nonlocal best_path, best_avg_reward

            if node.node_type != "virtual":
                current_path.append(node)

            if node.evaluation and node.evaluation.get("decision") == "complete":
                rewards = [n.average_reward for n in current_path if n.visits > 0]
                avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_path = list(current_path)

            for child in node.children:
                if child.subtask_id >= current_subtask_id:
                    dfs(child, current_path, max(current_subtask_id, child.subtask_id))

            if node.node_type != "virtual":
                current_path.pop()

        dfs(self.tree.root, [], self.tree.root.subtask_id)

        return [
            {
                "node_index": n.node_index,
                "subtask_id": n.subtask_id,
                "node_type": n.node_type,
                "description": n.subtask_description,
                "result": getattr(n, "result", None),
                "score": n.evaluation.get("score") if n.evaluation else None,
                "log_path": getattr(n, "log_path", None),
            }
            for n in best_path
        ]

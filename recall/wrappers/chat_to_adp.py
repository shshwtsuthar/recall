"""
Helpers for converting chat-and-tool trajectory datasets into Recall ADP.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("recall.chat_to_adp")

_UPLOADED_FILES_RE = re.compile(
    r"<uploaded_files>\s*(.*?)\s*</uploaded_files>",
    re.DOTALL | re.IGNORECASE,
)
_EXIT_CODE_RE = re.compile(r"\[exit_code=(-?\d+)\]")
_FILE_EDITOR_COMMANDS = {"view", "edit", "create", "undo_edit"}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            return str(value)
    return value


def deterministic_trajectory_id(*parts: str) -> str:
    joined = "\x1f".join(parts)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def _message_action(content: str, description: str | None = None) -> dict[str, Any]:
    return {
        "class_": "message_action",
        "content": content,
        "description": description,
    }


def _code_action(
    language: str,
    content: str,
    description: str | None = None,
) -> dict[str, Any]:
    return {
        "class_": "code_action",
        "language": language,
        "content": content,
        "description": description,
    }


def _api_action(
    function: str,
    kwargs: dict[str, Any],
    description: str | None = None,
) -> dict[str, Any]:
    return {
        "class_": "api_action",
        "function": function,
        "kwargs": kwargs,
        "description": description,
    }


def _text_observation(
    content: str,
    source: str = "environment",
    name: str | None = None,
) -> dict[str, Any]:
    return {
        "class_": "text_observation",
        "content": content,
        "source": source,
        "name": name,
    }


def _load_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, dict):
        return _json_safe(raw_arguments)
    if isinstance(raw_arguments, str):
        text = raw_arguments.strip()
        if not text:
            return {}
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return {"raw_arguments": text}
        if isinstance(loaded, dict):
            return _json_safe(loaded)
        return {"value": _json_safe(loaded)}
    return {"value": _json_safe(raw_arguments)}


def _description_from_kwargs(kwargs: dict[str, Any]) -> str | None:
    thought = _clean_text(kwargs.get("thought"))
    if thought:
        return thought
    return None


def _uploaded_paths(text: str) -> list[str]:
    paths: list[str] = []
    for block in _UPLOADED_FILES_RE.findall(text):
        for line in block.splitlines():
            line = line.strip()
            if line:
                paths.append(line)
    return paths


def _strip_uploaded_files(text: str) -> str:
    return _UPLOADED_FILES_RE.sub("", text).strip()


def _extract_task_goal(messages: Iterable[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        text = _clean_text(msg.get("content"))
        if not text:
            continue
        stripped = _strip_uploaded_files(text)
        return stripped or text
    return ""


def _extract_working_dir(
    messages: Iterable[dict[str, Any]],
    fallback: str = "",
) -> str:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        text = _clean_text(msg.get("content"))
        paths = _uploaded_paths(text)
        if paths:
            return paths[0]
    return fallback


def _extract_tool_names(tools: Any) -> list[str]:
    names: list[str] = []
    for tool in tools or []:
        function = (tool or {}).get("function") or {}
        name = function.get("name")
        if name:
            names.append(name)
    return names


def _tool_call_to_adp(
    tool_call: dict[str, Any],
    assistant_content: str,
) -> tuple[dict[str, Any] | None, str | None]:
    function = (tool_call.get("function") or {})
    name = _clean_text(function.get("name"))
    kwargs = _load_tool_arguments(function.get("arguments"))
    description = _description_from_kwargs(kwargs)

    if not name:
        return None, None

    if name == "finish":
        finish_text = assistant_content or _clean_text(kwargs.get("message"))
        finish_text = finish_text or _clean_text(kwargs.get("response"))
        if finish_text:
            return _message_action(f"<finish> {finish_text}".strip()), "finish"
        return _message_action("<finish>"), "finish"

    if name == "execute_bash":
        command = _clean_text(kwargs.get("command"))
        if command and not kwargs.get("is_input"):
            return _code_action(
                language="bash",
                content=command,
                description=description,
            ), "execute_bash"
        return _api_action(
            function="execute_bash",
            kwargs=kwargs,
            description=description,
        ), "execute_bash"

    if name == "str_replace_editor":
        command = _clean_text(kwargs.get("command")) or "view"
        return _api_action(
            function=f"file_editor_{command}",
            kwargs=kwargs,
            description=description,
        ), f"file_editor_{command}"

    if name in _FILE_EDITOR_COMMANDS:
        return _api_action(
            function=f"file_editor_{name}",
            kwargs=kwargs,
            description=description,
        ), f"file_editor_{name}"

    if name in {"browse_url", "browse_interactive"}:
        return _api_action(
            function="browse",
            kwargs=kwargs,
            description=description,
        ), "browse"

    return _api_action(
        function=name,
        kwargs=kwargs,
        description=description,
    ), name


def _assistant_items(
    message: dict[str, Any],
    action_counts: Counter[str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    content = _clean_text(message.get("content"))
    tool_calls = message.get("tool_calls") or []
    has_finish = any(
        ((tool_call.get("function") or {}).get("name") == "finish")
        for tool_call in tool_calls
    )

    if content and not has_finish:
        items.append(_message_action(content))
        action_counts["assistant_message"] += 1

    for tool_call in tool_calls:
        item, label = _tool_call_to_adp(tool_call, assistant_content=content)
        if item is None:
            continue
        items.append(item)
        if label:
            action_counts[label] += 1

    return items


def _tool_observation(message: dict[str, Any]) -> dict[str, Any] | None:
    content = _clean_text(message.get("content"))
    if not content:
        return None
    if content.startswith("OBSERVATION:"):
        content = content[len("OBSERVATION:"):].lstrip()
    return _text_observation(
        content=content,
        source="environment",
        name=_clean_text(message.get("name")) or None,
    )


def _count_errors(content: Iterable[dict[str, Any]]) -> int:
    errors = 0
    for item in content:
        if item.get("class_") != "text_observation":
            continue
        text = _clean_text(item.get("content"))
        for match in _EXIT_CODE_RE.finditer(text):
            if int(match.group(1)) != 0:
                errors += 1
                break
    return errors


def convert_chat_trajectory(
    *,
    messages: list[dict[str, Any]],
    trajectory_id: str,
    source_format: str,
    tools: list[dict[str, Any]] | None = None,
    extra_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    system_prompt = ""

    for message in messages:
        role = _clean_text(message.get("role"))

        if role == "system":
            if not system_prompt:
                system_prompt = _clean_text(message.get("content"))
            continue

        if role == "user":
            text = _clean_text(message.get("content"))
            if text:
                content.append(_text_observation(text, source="user"))
                action_counts["user_message"] += 1
            continue

        if role == "assistant":
            content.extend(_assistant_items(message, action_counts))
            continue

        if role == "tool":
            observation = _tool_observation(message)
            if observation is not None:
                content.append(observation)
                label = observation.get("name") or "tool_observation"
                action_counts[f"observation:{label}"] += 1
            continue

        logger.debug("Skipping unsupported message role: %s", role)

    details: dict[str, Any] = {
        "task_goal": _extract_task_goal(messages),
        "tools_available": _extract_tool_names(tools),
        "working_dir": _extract_working_dir(
            messages,
            fallback=_clean_text((extra_details or {}).get("repo")),
        ),
        "error_count": _count_errors(content),
        "total_events": len(messages),
        "action_counts": dict(action_counts),
        "source_format": source_format,
    }

    if system_prompt:
        details["system_prompt"] = system_prompt

    if extra_details:
        details.update(_json_safe(extra_details))

    return {
        "id": trajectory_id,
        "content": content,
        "details": details,
    }


def output_path_for(output_dir: str | Path, trajectory_id: str) -> Path:
    return Path(output_dir) / f"{trajectory_id}.json"


def write_adp_trajectory(
    trajectory: dict[str, Any],
    output_dir: str | Path,
    *,
    pretty: bool = False,
    overwrite: bool = False,
) -> Path | None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_path_for(output_dir, trajectory["id"])
    if path.exists() and not overwrite:
        return None

    path.write_text(
        json.dumps(trajectory, indent=2 if pretty else None, default=str),
        encoding="utf-8",
    )
    return path

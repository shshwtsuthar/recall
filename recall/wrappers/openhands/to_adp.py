"""
OpenHands V1 → ADP Converter
=============================

Reads OpenHands V1 SDK event JSON files from a session's events/ directory
and converts them into an ADP Trajectory object ready for Qdrant storage.

Input:  .openhands/conversations/<session_id>/events/event-*.json
Output: ADP Trajectory dict (JSON-serializable)

The ADP schema (from the ADP paper, ICLR 2026):
  Trajectory:
    id: str
    content: list[Action | Observation]   # alternating sequence
    details: dict                          # flexible metadata

  Actions:
    CodeAction:    {class_: "code_action",    language, content, description}
    APIAction:     {class_: "api_action",     function, kwargs, description}
    MessageAction: {class_: "message_action", content, description}

  Observations:
    TextObservation: {class_: "text_observation", content, source}
    WebObservation:  {class_: "web_observation",  url, html, axtree, ...}

Usage:
    from openhands_to_adp import convert_session

    trajectory = convert_session("/path/to/.openhands/conversations/<id>/events")
    # trajectory is a dict ready for JSON serialization / Qdrant storage
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("recall.converter")


# ---------------------------------------------------------------------------
# ADP object builders
# ---------------------------------------------------------------------------

def _code_action(language: str, content: str, description: str | None = None) -> dict:
    return {
        "class_": "code_action",
        "language": language,
        "content": content,
        "description": description,
    }


def _api_action(function: str, kwargs: dict, description: str | None = None) -> dict:
    return {
        "class_": "api_action",
        "function": function,
        "kwargs": kwargs,
        "description": description,
    }


def _message_action(content: str, description: str | None = None) -> dict:
    return {
        "class_": "message_action",
        "content": content,
        "description": description,
    }


def _text_observation(content: str, source: str = "environment") -> dict:
    return {
        "class_": "text_observation",
        "content": content,
        "source": source,
        "name": None,
    }


# ---------------------------------------------------------------------------
# Field extractors for the OpenHands V1 event schema
# ---------------------------------------------------------------------------

def _extract_message_text(event: dict) -> str:
    """Extract the text content from a MessageEvent's llm_message."""
    llm_msg = event.get("llm_message", {})
    content_blocks = llm_msg.get("content", [])
    texts = []
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(block.get("text", ""))
        elif isinstance(block, str):
            texts.append(block)
    return "\n".join(texts).strip()


def _extract_reasoning(event: dict) -> str | None:
    """Extract reasoning/thinking content from an ActionEvent or MessageEvent."""
    reasoning = event.get("reasoning_content")
    if reasoning and isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    # Fallback: check thinking_blocks
    blocks = event.get("thinking_blocks", [])
    if blocks:
        parts = [b.get("thinking", "") for b in blocks if isinstance(b, dict)]
        joined = "\n".join(p for p in parts if p).strip()
        if joined:
            return joined

    return None


def _extract_observation_text(event: dict) -> str:
    """Extract text content from an ObservationEvent's observation field."""
    obs = event.get("observation", {})
    content_blocks = obs.get("content", [])
    texts = []
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(block.get("text", ""))
        elif isinstance(block, str):
            texts.append(block)
    return "\n".join(texts).strip()


def _extract_obs_metadata(event: dict) -> dict:
    """Extract useful metadata from an ObservationEvent (exit_code, error, etc.)."""
    obs = event.get("observation", {})
    meta = {}
    if "exit_code" in obs:
        meta["exit_code"] = obs["exit_code"]
    if obs.get("is_error"):
        meta["is_error"] = True
    if obs.get("timeout"):
        meta["timeout"] = True
    if obs.get("command"):
        meta["command"] = obs["command"]
    obs_meta = obs.get("metadata", {})
    if obs_meta.get("working_dir"):
        meta["working_dir"] = obs_meta["working_dir"]
    return meta


# ---------------------------------------------------------------------------
# Event → ADP conversion
# ---------------------------------------------------------------------------

def _convert_event(event: dict) -> list[dict]:
    """
    Convert a single OpenHands V1 event into zero or more ADP content items.

    Returns a list because some events map to nothing (SystemPromptEvent)
    and most map to exactly one ADP item.
    """
    kind = event.get("kind", "")
    source = event.get("source", "")

    # --- SystemPromptEvent: skip, metadata only ---
    if kind == "SystemPromptEvent":
        return []

    # --- MessageEvent ---
    if kind == "MessageEvent":
        text = _extract_message_text(event)
        if not text:
            return []

        reasoning = _extract_reasoning(event)

        if source == "user":
            return [_text_observation(text, source="user")]
        elif source == "agent":
            return [_message_action(text, description=reasoning)]
        return []

    # --- ActionEvent ---
    if kind == "ActionEvent":
        action = event.get("action", {})
        action_kind = action.get("kind", "")
        reasoning = _extract_reasoning(event)

        # TerminalAction → CodeAction(language=bash)
        if action_kind == "TerminalAction":
            command = action.get("command", "")
            return [_code_action(
                language="bash",
                content=command,
                description=reasoning,
            )]

        # FileEditorAction → APIAction
        if action_kind == "FileEditorAction":
            cmd = action.get("command", "view")  # view, edit, create, undo_edit
            kwargs = {}
            if action.get("path"):
                kwargs["path"] = action["path"]
            if action.get("old_str") is not None:
                kwargs["old_str"] = action["old_str"]
            if action.get("new_str") is not None:
                kwargs["new_str"] = action["new_str"]
            if action.get("file_text") is not None:
                kwargs["file_text"] = action["file_text"]
            if action.get("view_range") is not None:
                kwargs["view_range"] = action["view_range"]
            if action.get("insert_line") is not None:
                kwargs["insert_line"] = action["insert_line"]
            return [_api_action(
                function=f"file_editor_{cmd}",
                kwargs=kwargs,
                description=reasoning,
            )]

        # ThinkAction → skip (reasoning is metadata, not a trajectory step)
        if action_kind == "ThinkAction":
            return []

        # FinishAction → MessageAction
        if action_kind == "FinishAction":
            message = action.get("message", "")
            if not message:
                outputs = action.get("outputs", {})
                message = outputs.get("message", "") if isinstance(outputs, dict) else str(outputs)
            return [_message_action(
                content=f"<finish> {message}".strip(),
                description=reasoning,
            )]

        # DelegateAction → APIAction
        if action_kind == "DelegateAction":
            return [_api_action(
                function="delegate",
                kwargs={
                    "command": action.get("command", ""),
                    "ids": action.get("ids", []),
                    "tasks": action.get("tasks", {}),
                },
                description=reasoning,
            )]

        # TaskTrackerAction → APIAction
        if action_kind == "TaskTrackerAction":
            return [_api_action(
                function="task_tracker",
                kwargs=action,
                description=reasoning,
            )]

        # BrowseInteractiveAction / BrowseURLAction → APIAction
        if action_kind in ("BrowseInteractiveAction", "BrowseURLAction"):
            return [_api_action(
                function="browse",
                kwargs={k: v for k, v in action.items() if k != "kind"},
                description=reasoning,
            )]

        # Unknown action kind → generic APIAction
        logger.warning("Unknown action kind: %s", action_kind)
        return [_api_action(
            function=action_kind,
            kwargs={k: v for k, v in action.items() if k != "kind"},
            description=reasoning,
        )]

    # --- ObservationEvent ---
    if kind == "ObservationEvent":
        obs = event.get("observation", {})
        obs_kind = obs.get("kind", "")
        text = _extract_observation_text(event)

        # Append metadata to text for richer observations
        meta = _extract_obs_metadata(event)
        suffix_parts = []
        if "exit_code" in meta and meta["exit_code"] != 0:
            suffix_parts.append(f"[exit_code={meta['exit_code']}]")
        if meta.get("is_error"):
            suffix_parts.append("[ERROR]")
        if meta.get("timeout"):
            suffix_parts.append("[TIMEOUT]")

        if suffix_parts:
            text = text + "\n" + " ".join(suffix_parts)

        # WebObservation would map to ADP WebObservation, but not in this session
        # For now, all observations are TextObservation
        return [_text_observation(text, source="environment")]

    # Unknown event kind
    logger.debug("Skipping unknown event kind: %s", kind)
    return []


# ---------------------------------------------------------------------------
# Session-level conversion
# ---------------------------------------------------------------------------

def _extract_session_metadata(events: list[dict]) -> dict[str, Any]:
    """Extract session-level metadata from the raw events."""
    details: dict[str, Any] = {}

    # From SystemPromptEvent (event-00000): extract tools list
    for ev in events:
        if ev.get("kind") == "SystemPromptEvent":
            tools = ev.get("tools", [])
            details["tools_available"] = [t.get("title", t.get("kind", "")) for t in tools]
            # Extract working directory from dynamic_context
            dynamic = ev.get("dynamic_context", {})
            if isinstance(dynamic, dict):
                text = dynamic.get("text", "")
                if "working directory is:" in text:
                    idx = text.index("working directory is:")
                    line = text[idx:].split("\n")[0]
                    details["working_dir"] = line.replace("working directory is:", "").strip()
            break

    # Count actions by type
    action_counts: dict[str, int] = {}
    error_count = 0
    for ev in events:
        if ev.get("kind") == "ActionEvent":
            ak = ev.get("action", {}).get("kind", "unknown")
            action_counts[ak] = action_counts.get(ak, 0) + 1
        if ev.get("kind") == "ObservationEvent":
            obs = ev.get("observation", {})
            if obs.get("is_error") or (obs.get("exit_code", 0) != 0):
                error_count += 1

    details["action_counts"] = action_counts
    details["error_count"] = error_count
    details["total_events"] = len(events)

    # Timestamps
    timestamps = [ev.get("timestamp", "") for ev in events if ev.get("timestamp")]
    if timestamps:
        details["started_at"] = timestamps[0]
        details["ended_at"] = timestamps[-1]

    return details


def _extract_task_goal(events: list[dict]) -> str:
    """Find the user's initial task/instruction from the events."""
    for ev in events:
        if ev.get("kind") == "MessageEvent" and ev.get("source") == "user":
            text = _extract_message_text(ev)
            if text:
                return text
    return ""


def convert_session(
    events_dir: str | Path,
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Convert an OpenHands V1 session's event files into an ADP Trajectory.

    Parameters
    ----------
    events_dir
        Path to the events/ directory containing event-*.json files.
    session_id
        Optional session ID. If None, inferred from the parent directory name.

    Returns
    -------
    ADP Trajectory dict with keys: id, content, details
    """
    events_dir = Path(events_dir)

    if session_id is None:
        # Infer from path: .../conversations/<session_id>/events
        session_id = events_dir.parent.name

    # Read and sort all event files
    event_files = sorted(events_dir.glob("event-*.json"))
    if not event_files:
        raise FileNotFoundError(f"No event-*.json files found in {events_dir}")

    raw_events = []
    for ef in event_files:
        try:
            with open(ef) as f:
                raw_events.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", ef.name, e)

    # Build ADP content sequence
    adp_content = []
    for ev in raw_events:
        items = _convert_event(ev)
        adp_content.extend(items)

    # Build metadata
    details = _extract_session_metadata(raw_events)
    details["task_goal"] = _extract_task_goal(raw_events)
    details["source_format"] = "openhands_v1"
    details["session_id"] = session_id

    trajectory = {
        "id": session_id,
        "content": adp_content,
        "details": details,
    }

    return trajectory


def convert_session_dir(conversation_dir: str | Path) -> dict[str, Any]:
    """
    Convenience: convert from the conversation directory level.

    Parameters
    ----------
    conversation_dir
        Path to .openhands/conversations/<session_id>/ (the dir containing events/)
    """
    conversation_dir = Path(conversation_dir)
    events_dir = conversation_dir / "events"
    if not events_dir.exists():
        raise FileNotFoundError(f"No events/ subdirectory in {conversation_dir}")
    return convert_session(events_dir, session_id=conversation_dir.name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Convert OpenHands V1 session events to ADP Trajectory"
    )
    parser.add_argument(
        "events_dir",
        help="Path to the events/ directory (or conversation directory containing events/)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path (default: stdout)",
        default=None,
    )
    parser.add_argument(
        "--session-id",
        help="Override session ID (default: inferred from directory name)",
        default=None,
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output",
    )
    args = parser.parse_args()

    events_path = Path(args.events_dir)

    # Auto-detect: if the path has events/ inside, use it directly
    # Otherwise check if the path IS the events dir
    if (events_path / "events").exists():
        trajectory = convert_session(events_path / "events", args.session_id)
    elif events_path.name == "events":
        trajectory = convert_session(events_path, args.session_id)
    else:
        parser.error(
            f"Cannot find event files. Provide path to events/ dir or its parent.\n"
            f"  Got: {events_path}"
        )
        return

    output_json = json.dumps(trajectory, indent=2 if args.pretty else None, default=str)

    if args.output:
        Path(args.output).write_text(output_json)
        print(f"Wrote ADP trajectory to {args.output}")
    else:
        print(output_json)

    # Summary
    content = trajectory["content"]
    details = trajectory["details"]
    n_actions = sum(1 for c in content if c["class_"] in ("code_action", "api_action", "message_action"))
    n_obs = sum(1 for c in content if c["class_"] == "text_observation")

    import sys
    print(f"\n--- ADP Trajectory Summary ---", file=sys.stderr)
    print(f"Session:     {trajectory['id']}", file=sys.stderr)
    print(f"Task:        {details.get('task_goal', '?')[:80]}", file=sys.stderr)
    print(f"ADP items:   {len(content)} ({n_actions} actions, {n_obs} observations)", file=sys.stderr)
    print(f"Raw events:  {details.get('total_events', '?')}", file=sys.stderr)
    print(f"Errors:      {details.get('error_count', 0)}", file=sys.stderr)
    print(f"Time range:  {details.get('started_at', '?')} → {details.get('ended_at', '?')}", file=sys.stderr)


if __name__ == "__main__":
    main()

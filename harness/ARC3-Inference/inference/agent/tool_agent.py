"""Direct OpenAI-compatible tool-calling analyzer for ARC puzzle runs."""
from __future__ import annotations

import base64
import heapq
import threading
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse, urlunparse

import requests

from inference.agent.action_observation import verdict as observation_verdict, clock_status
from inference.agent.priority_scheduler import (
    ProgressPace, PrioritySnapshot, parse_total_levels, priority_value,
)
from inference.utils.animation import (
    _transient_cells as _animation_transient_cells,
    build_animation_frames,
    build_animation_view,
    describe_animation,
)
from inference.agent.action_names import reset_exposed, to_engine_action, to_model_action, undo_exposure_mode
from inference.utils.retained_functions import function_signature
from inference.agent.prompts import (
    build_system_prompt,
    COMPACTION_NOTE_HEADER,
    COMPACTION_REQUEST_PROMPT,
    GAME_OVER_REPLAY_WARNING,
    GAME_OVER_USER_PROMPT,
    LEVEL_START_USER_PROMPT,
    MALFORMED_TOOL_CALL_PROMPT,
    MARKUP_TOOL_CALL_PROMPT,
    NO_TOOL_CALL_PROMPT,
    PYTHON_CODE_PARAMETER_DESCRIPTION,
    PYTHON_TOOL_DESCRIPTION,
    STALL_NOTE,
    TRUNCATED_REPLY_PROMPT,
)

from inference.agent.vision_context import (
    animation_composite_part,
    animation_death_image_mode,
    animation_image_mode,
    animation_peak_part,
    current_grid_image_part,
)

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.runtime_state import Frame, HistoryEntry, RUNTIME_STATE_FILENAME, load_runtime_state
from inference.utils.openai_compat import (
    assemble_streamed_chat_response,
    build_chat_payload,
    build_headers,
)

log = logging.getLogger(__name__)

_LOCAL_ANALYZER_MODEL_ID = os.environ.get("LOCAL_ANALYZER_MODEL_ID", "")
_LOCAL_ANALYZER_BASE_URL = os.environ.get("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1")
_DEFAULT_ANALYZER_MODEL = os.environ.get(
    "INFERENCE_ANALYZER_MODEL",
    _LOCAL_ANALYZER_MODEL_ID,
)
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    flags=re.DOTALL | re.IGNORECASE,
)
_TOOL_CALL_PARAMETER_RE = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
    flags=re.DOTALL | re.IGNORECASE,
)
_THINK_TAG_RE = re.compile(r"</?think>", flags=re.IGNORECASE)


def _get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default



def _contains_tool_call_markup(*chunks: str) -> bool:
    for chunk in chunks:
        lowered = chunk.lower()
        if "<tool_call" in lowered or "<function=" in lowered:
            return True
    return False


def _strip_tool_call_markup(text: str) -> str:
    if not text.strip():
        return ""
    stripped = _TOOL_CALL_BLOCK_RE.sub("", text)
    return stripped.strip()


def _recover_tool_calls_from_markup(*chunks: str) -> list[dict[str, Any]]:
    recovered: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        if not chunk.strip():
            continue
        for match in _TOOL_CALL_BLOCK_RE.finditer(chunk):
            tool_name = str(match.group(1) or "").strip()
            if not tool_name:
                continue
            raw_body = str(match.group(2) or "")
            arguments = {
                str(parameter_name).strip(): value
                for parameter_name, value in _TOOL_CALL_PARAMETER_RE.findall(raw_body)
                if str(parameter_name).strip()
            }
            cache_key = (
                tool_name,
                json.dumps(arguments, ensure_ascii=True, sort_keys=True),
            )
            if cache_key in seen:
                continue
            seen.add(cache_key)
            recovered.append(
                {
                    "id": f"markup-call-{len(recovered) + 1}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments, ensure_ascii=True),
                    },
                }
            )
    return recovered


def _get_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


_LOCAL_ANALYZER_MAX_OUTPUT = _get_env_int("LOCAL_ANALYZER_MAX_OUTPUT", 0)
_LOCAL_ANALYZER_CONTEXT_WINDOW = _get_env_int("LOCAL_ANALYZER_CONTEXT_WINDOW", 32768)
_LOCAL_ANALYZER_TIMEOUT = _get_env_float("LOCAL_ANALYZER_TIMEOUT", 0.0)
_LOCAL_ANALYZER_TOOL_STEPS = _get_env_int("LOCAL_ANALYZER_TOOL_STEPS", 12)
_LOCAL_ANALYZER_TOOL_TIMEOUT = _get_env_int("LOCAL_ANALYZER_TOOL_TIMEOUT", 30)
_LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS = _get_env_int("LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS", 1024)
_LOCAL_ANALYZER_YIELD_SECONDS = _get_env_float("LOCAL_ANALYZER_YIELD_SECONDS", 0.0)
# A wall-clock budget is really a token budget with the throughput baked in:
# the same 45 seconds bought 2,140 generated tokens on one backend and 3,829 on
# a faster one, so every serving change silently retuned it. Measured in
# tokens the setting means the same thing whatever the hardware does. Zero
# disables, as with the seconds; if both are set, whichever comes first ends
# the turn.
_LOCAL_ANALYZER_YIELD_TOKENS = _get_env_int("LOCAL_ANALYZER_YIELD_TOKENS", 0)
_LOCAL_ANALYZER_ENABLE_THINKING = _get_env_bool("LOCAL_ANALYZER_ENABLE_THINKING", True)
_LOCAL_ANALYZER_TEMPERATURE = _get_env_float("LOCAL_ANALYZER_TEMPERATURE", 0.6)
_LOCAL_ANALYZER_TOP_P = _get_env_float("LOCAL_ANALYZER_TOP_P", 0.95)
_LOCAL_ANALYZER_TOP_K = _get_env_int("LOCAL_ANALYZER_TOP_K", 20)
_LOCAL_ANALYZER_SEED = _get_env_int("LOCAL_ANALYZER_SEED", -1)
_REQUEST_SAFETY_MARGIN_TOKENS = 512
_CONTEXT_OVERFLOW_RETRY_TRIM_TOKENS = 512
def _context_drain_floor_mode() -> bool:
    """Whether the drain stops ABOVE the target rather than below it.

    History is dropped in whole blocks, so trimming "until under the target"
    normally overshoots by most of a block. With a 40k budget and a 12k drain
    the band is nominally 28k-40k, but simulation with ~6k blocks gives an
    effective 14k-38k: the average sits ~3k below the nominal midpoint and the
    minimum is far below the target.

    Floor mode drops the mandatory blocks to get under budget, then stops
    before a drop would cross below the target - so the retained content keeps
    a real floor and the band means what it says. The cost is trim frequency:
    the overshoot is what lengthens the interval between trims, so removing it
    trims about half again as often, which shortens the stable prefix and cuts
    the prefix-cache hit rate. Off by default for that reason; the same average
    depth is available by raising the budget, which does not touch trim
    frequency."""
    return _get_env_bool("ARC3_CONTEXT_DRAIN_FLOOR", False)


def _context_drain_tokens() -> int:
    """Hysteresis for context trimming, in tokens.

    Without it, trimming drops the oldest exchange the moment the estimate
    exceeds the budget and stops the instant it fits - so once history reaches
    the ceiling, nearly every turn evicts a little. Each eviction is cheap in
    tokens but total in cache cost: prefix caching keys on the token sequence
    from position zero, so removing anything from the FRONT shifts every
    surviving token and invalidates the whole cached prefix, not just the part
    removed. Frequent small evictions are therefore the worst case.

    With a drain set, trimming still TRIGGERS at the budget but keeps dropping
    until it is `drain` tokens below it, so evictions become rare and large:
    the same tokens discarded overall, a fraction of the invalidations. The
    cost is average history depth - the window sits below the budget most of
    the time rather than pressed against it.

    Size it against tokens per turn: at ~4-5k per turn including the opener,
    8-12k buys two or three clean turns between evictions. 0 (default) keeps
    the original drop-until-it-fits behavior."""
    return max(0, _get_env_int("ARC3_CONTEXT_DRAIN_TOKENS", 0))


def _compaction_enabled() -> bool:
    """r17: before a drain removes old turns, the model writes notes that replace them (compaction instead of
    silent truncation). Off by default; the notebook turns it on."""
    return _get_env_bool("ARC3_CONTEXT_COMPACTION", False)


def _compaction_margin_tokens() -> int:
    """Compaction starts this far below the context budget, so the notes request (the full context plus the
    request, plus the notes) still fits the server's context length."""
    return max(0, _get_env_int("ARC3_COMPACTION_MARGIN_TOKENS", 16384))


def _compaction_max_tokens() -> int:
    """Output cap of the notes request; it includes the model's thinking, which stays on (prefix cache)."""
    return max(256, _get_env_int("ARC3_COMPACTION_MAX_TOKENS", 12288))


def _history_turn_drain() -> int:
    """Hysteresis for the assistant-turn cap, in assistant messages.

    The token trimmer got hysteresis in ARC3_CONTEXT_DRAIN_TOKENS, but the
    turn cap runs AFTER it and had none: once history sits at the cap, every
    commit shaves back to exactly N, so the prefix is invalidated on every
    single turn. With a 46k window and a cap of 30 assistant messages the cap
    is the binding limit, which is why a token drain alone changed nothing.

    Set this and the cap triggers at N but drains to N - drain, so evictions
    become periodic instead of continuous."""
    return max(0, _get_env_int("ARC3_HISTORY_TURN_DRAIN", 0))


def _history_drain_coalesce() -> bool:
    """Take the deep turn cut whenever the token trimmer already dropped.

    Invalidation is all-or-nothing: removing anything from the front of
    history shifts every surviving token, so a commit that has already
    dropped blocks for token reasons has already paid the full cache cost.
    Dropping turns in that same commit is therefore free, whereas letting the
    cap fire on its own a turn later costs a second invalidation. This makes
    the two mechanisms evict together rather than alternately."""
    return _get_env_bool("ARC3_HISTORY_DRAIN_COALESCE", False)


def _persistent_history_assistant_turns() -> int:
    """Cap on assistant messages kept in persistent history.

    Counts ASSISTANT MESSAGES, not analyzer turns: a turn that inspects, then
    computes, then acts spends three of them, so 30 is closer to ten or
    fifteen turns of real history than to thirty. The cap is independent of
    the token budget, so with a large context window it can be the binding
    constraint while thousands of tokens sit unused - which is what the
    default 30 does at 98k, having been chosen when the window was 32k.

    0 or negative disables the cap entirely, leaving the token budget as the
    only limit."""
    return _get_env_int("ARC3_HISTORY_ASSISTANT_TURNS", 30)
_DIFF_IMAGE = _get_env_bool("ARC3_DIFF_IMAGE", False)
_RESPONSE_META_MAX_CHARS = 4000

_TRUNCATED_TAIL_CHARS = 1500


def _truncation_recovery_enabled() -> bool:
    """Replace a reply cut off at the output limit by its tail and ask for a short hypothesis + tool call."""
    return _get_env_bool("ARC3_TRUNCATION_RECOVERY", True)


def _motion_summary_enabled() -> bool:
    """r10: object motion phrases per action and animation tracks in the opener (and sandbox helpers)."""
    return _get_env_bool("ARC3_MOTION_SUMMARY", True)


def _endgame_seconds() -> float:
    """Below this much remaining game time the time line says to keep acting (0 disables)."""
    return max(0.0, _get_env_float("ARC3_ENDGAME_SECONDS", 300.0))


def _stall_seconds() -> float:
    """Seconds a turn may deliberate without acting before the anti-stall note (0 disables)."""
    return max(0.0, _get_env_float("ARC3_STALL_SECONDS", 150.0))


def _with_harness_note(content: str, note: str) -> str:
    """Attach a harness note to a JSON tool result (or append to plain text)."""
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return f"{content}\n[harness note] {note}"
    if not isinstance(payload, dict):
        return f"{content}\n[harness note] {note}"
    payload["harness_note"] = note
    return json.dumps(payload, indent=2)


def _time_cues_enabled() -> bool:
    """Remaining game time and time spent on the level, in every opener."""
    return _get_env_bool("ARC3_TIME_CUES", True)


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{int(round(seconds))} s"
    return f"{seconds / 60.0:.0f} min"


def _normalize_valid_actions(valid_actions: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in valid_actions or []:
        engine_name = to_engine_action(value)
        name = to_model_action(engine_name or value)
        if name and name not in names:
            names.append(name)
    return names


def _format_valid_action_line(valid_actions: list[str] | None) -> str:
    names = _normalize_valid_actions(valid_actions)
    if not names:
        return "unknown"
    return ", ".join(names)


def _terminal_action_reason(result: dict[str, Any]) -> str | None:
    if result.get("run_complete"):
        return "run_complete"
    if result.get("game_over"):
        return "game_over"
    if result.get("level_completed"):
        return "level_completed"
    if result.get("done"):
        return "done"
    return None


def _terminal_action_stop_detail(reason: str | None) -> str:
    if reason == "run_complete":
        return "No further actions were executed because the run is already complete."
    if reason == "game_over":
        return (
            "No further actions were executed because the previous action reached GAME_OVER "
            "(attempt failed; the run is NOT finished). The runner will auto-reset the level "
            "before your next turn. Cause is either the action itself (hazard) or a depleted "
            "step/time budget (usually a shrinking bar at the grid border)."
        )
    if reason == "level_completed":
        return (
            "No further actions were executed because the previous action completed a level; "
            "re-ground on the new scene before acting again."
        )
    if reason == "done":
        return "No further actions were executed because the environment reported done."
    return "No further actions were executed because the previous action reached a terminal state."


def _display_action_number(action_num: int) -> int:
    return max(1, int(action_num) + 1)


def _stale_summary_lines(summary: dict[str, Any]) -> list[str]:
    """Opener for a turn that follows one which executed nothing: say so, then
    restate the last executed outcome as a reminder rather than as news."""
    if summary.get("stale_reason") == "no_tool_call":
        lines = ["Your previous turn produced no tool call and executed no actions."]
    else:
        lines = ["Your previous turn executed no actions (inspection only)."]
    executed_actions = summary.get("executed_actions")
    rendered = ""
    if isinstance(executed_actions, list):
        rendered = ", ".join(
            str(name).strip() for name in executed_actions[:10] if str(name).strip()
        )
    if summary.get("game_over"):
        fatal = str(summary.get("fatal_action") or "")
        lines.append(
            "Reminder: the last executed sequence ("
            + (rendered or "see earlier messages")
            + ") ended in GAME OVER"
            + (f" after '{fatal}'" if fatal else "")
            + "; the level was reset and that sequence must not be replayed."
        )
    elif summary.get("level_transition"):
        lines.append("Reminder: the last executed sequence completed the previous level; you are on the new level.")
    elif rendered:
        lines.append(f"For reference, the last executed sequence was: {rendered}.")
    return lines



_LEAKED_TOOLCALL_RE = None


def _extract_leaked_tool_calls(message: dict[str, Any]) -> dict[str, Any]:
    """Fallback for servers without a tool-call parser for the model's
    dialect: when a response carries no structured tool_calls but the content
    contains Nemotron-style <TOOLCALL>[{...}]</TOOLCALL> markup, parse it and
    synthesize the OpenAI-shaped tool_calls the harness expects. Malformed
    payloads leave the message untouched. Disable with
    ARC3_TOOLCALL_TEXT_FALLBACK=0."""
    global _LEAKED_TOOLCALL_RE
    if not _get_env_bool("ARC3_TOOLCALL_TEXT_FALLBACK", True):
        return message
    if not isinstance(message, dict) or message.get("tool_calls"):
        return message
    content = message.get("content")
    if not isinstance(content, str) or "<TOOLCALL>" not in content:
        return message
    import json as _json
    import re as _re
    import uuid as _uuid

    if _LEAKED_TOOLCALL_RE is None:
        _LEAKED_TOOLCALL_RE = _re.compile(
            r"<TOOLCALL>\s*(\[.*?\])\s*</TOOLCALL>", _re.DOTALL
        )
    match = _LEAKED_TOOLCALL_RE.search(content)
    if match is None:
        return message
    try:
        parsed = _json.loads(match.group(1))
    except _json.JSONDecodeError:
        return message
    if not isinstance(parsed, list):
        parsed = [parsed]
    calls: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        arguments = item.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = _json.dumps(arguments, ensure_ascii=False)
        calls.append(
            {
                "id": f"call_{_uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": str(item["name"]),
                    "arguments": arguments,
                },
            }
        )
    if not calls:
        return message
    remainder = (content[: match.start()] + content[match.end():]).strip()
    updated = dict(message)
    updated["tool_calls"] = calls
    updated["content"] = remainder or None
    return updated


def _env_int(name: str, default: int) -> int:
    """A numeric override that cannot take the run down on a typo."""
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


_RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504, 520, 522, 524}
# Establishing a TCP connection is fast or it is not happening: a server that
# has exited refuses instantly, and one that is unreachable will not become
# reachable inside the read budget. Separate from the read timeout so a dead
# endpoint fails in seconds rather than waiting out the generation budget.
_CONNECT_TIMEOUT_SECONDS = 10.0


def _http_retry_delay_seconds(
    attempt: int,
    retry_after: str | None,
    base_seconds: float,
    max_seconds: float = 60.0,
) -> float:
    """Exponential backoff with jitter, or the server's Retry-After when it
    parses as a number and asks for longer. Capped at 60s per wait.

    Retry-After raises the wait, never lowers it. A provider under load can
    answer `Retry-After: 1` to every request - observed at 100 consecutive
    retries roughly 1.3s apart - and honouring that literally means hammering
    an endpoint that has already said no, which neither clears the rate limit
    nor helps the run. The configured base is the floor; the server can still
    ask for more.
    """
    backoff = min(max_seconds, base_seconds * (2 ** attempt)) + random.uniform(0.0, 1.0)
    if retry_after:
        try:
            # max_seconds caps the exponential, not an explicit request from the
            # server: a Retry-After of 30 against a 5s ceiling still waits 30,
            # because the endpoint said so and hammering it earlier helps no one
            return max(backoff, min(60.0, float(retry_after)))
        except ValueError:
            pass
    return backoff


def _post_with_retries(
    post_fn,
    *,
    retries: int,
    base_seconds: float,
    max_seconds: float = 60.0,
    initial_seconds: float = 0.0,
    sleep=time.sleep,
    monotonic=time.monotonic,
):
    """Retry transient HTTP failures (rate limits, gateway errors) in place,
    BEFORE they escalate to turn abort + history rollback. Returns the final
    response either way; non-retryable statuses return immediately and flow
    into the existing error handling. Rate-limit retries here preserve the
    turn's partial tool work, which the abort path deliberately discards."""
    # A negative count means keep trying. Zero keeps its obvious meaning - do
    # not retry - so a typo disables retries rather than hanging the run.
    unlimited = retries < 0
    budget = max(0, retries)
    # A floor on total retry time, used once. On Kaggle the benchmark is
    # released before the server has finished loading, so the first request of
    # every game meets a refused connection and has to wait it out - minutes,
    # not the seconds the ordinary budget allows. Expressed as time rather than
    # a count because the count that covers a 15-minute load would be absurd at
    # any sensible delay, and would then apply to every later failure too.
    started_at = monotonic()

    def _budget_label(attempt_index: int) -> str:
        """What the log should call the limit. Only reached outside a grace
        period now - inside one the retry lines are suppressed entirely, since
        "1/3" describes a budget that is not the one in force."""
        if unlimited:
            return "unlimited"
        if attempt_index >= budget:
            return f"{initial_seconds:.0f}s grace"
        return str(budget)

    def _exhausted(attempt_index: int) -> bool:
        if unlimited:
            return False
        if attempt_index < budget:
            return False
        return (monotonic() - started_at) >= initial_seconds

    def _announce_grace(reason: str) -> None:
        """Said once, on the first failure, so the grace is visible from the
        start. Without it the log shows three ordinary-looking retries before
        the "Ns grace" label appears, and a reader watching a cold start cannot
        tell a long wait from a broken one."""
        nonlocal announced
        if announced or initial_seconds <= 0:
            return
        announced = True
        log.warning(
            "first analyzer request failed (%s); retrying for up to %.0fs while "
            "the endpoint comes up, then falling back to %d retries per request",
            reason, initial_seconds, budget,
        )

    announced = False
    response = None
    attempt = -1
    while True:
        attempt += 1
        try:
            response = post_fn()
        except requests.exceptions.RequestException as exc:
            # The server was never reached - refused, DNS, a socket error, or a
            # connect timeout. Without this the exception leaves the turn
            # entirely and the analyzer-level retry starts it over, discarding
            # whatever tool work it had already done; retrying the request in
            # place keeps it. A ReadTimeout is deliberately not caught: the
            # response body stalled, which has its own path that can yield with
            # history preserved.
            if _is_read_timeout(exc) or _exhausted(attempt):
                raise
            _announce_grace(type(exc).__name__)
            delay = _http_retry_delay_seconds(attempt, None, base_seconds, max_seconds)
            # Silent while the grace is carrying the loop: at 14 games polling a
            # cold endpoint every 5s for 15 minutes this line alone would be
            # ~2,500 entries, burying everything else. The announcement above
            # already said a wait is in progress, and the outcome is logged when
            # the loop ends either way.
            if initial_seconds <= 0:
                log.warning(
                    "analyzer endpoint unreachable (%s); retry %d/%s in %.1fs",
                    type(exc).__name__, attempt + 1, _budget_label(attempt), delay,
                )
            sleep(delay)
            continue
        status = getattr(response, "status_code", 200)
        if status not in _RETRYABLE_HTTP_STATUSES or _exhausted(attempt):
            return response
        _announce_grace(f"HTTP {status}")
        retry_after = None
        headers = getattr(response, "headers", None)
        if headers is not None:
            retry_after = headers.get("Retry-After")
        delay = _http_retry_delay_seconds(attempt, retry_after, base_seconds, max_seconds)
        if initial_seconds <= 0:
            log.warning(
                "analyzer endpoint returned HTTP %s; retry %d/%s in %.1fs",
                status, attempt + 1, _budget_label(attempt), delay,
            )
        sleep(delay)
    return response


_CONTROL_MESSAGE_KEY = "_arc3_control"










_DEGENERATE_CONTENT_STUB = (
    "(previous reply was highly repetitive and has been omitted; do not "
    "continue that pattern - write concise plain prose)"
)








def _yield_on_timeout_enabled() -> bool:
    """Treat an analyzer read timeout as a yield instead of a failed turn.

    The default path clears preserve_history, so the whole invocation's tool
    calls and results are rolled back and the retry re-derives them from
    scratch - on Kaggle that is a ~900s round trip discarded. Nothing can have
    executed when a timeout fires (an executing dispatch breaks the loop before
    another request is issued), so preserving the exchanges is safe: history
    ends on a tool message and the resumed invocation appends a user message,
    exactly the shape every yielded turn already produces.

    Scoped to read timeouts only. Connection errors and HTTP failures keep the
    rollback path, and context-overflow handling is untouched - preserving an
    over-long context would just reproduce the failure."""
    return _get_env_bool("ARC3_YIELD_ON_TIMEOUT", True)


def _is_read_timeout(exc: BaseException) -> bool:
    """requests raises ReadTimeout for a stalled response body; ConnectTimeout
    (a sibling of the same base) means the server was never reached, so it is
    deliberately excluded."""
    return isinstance(exc, requests.exceptions.ReadTimeout)



_HISTORY_IMAGE_PLACEHOLDER = "[grid image omitted from history]"


def _history_image_keep() -> int | None:
    """How many of the most recent image-bearing history MESSAGES keep their
    images. None (unset) preserves today's behavior - images accumulate for as
    long as their message is retained, ~400 tokens each at
    MULTIMODAL_UPSCALE=10. 0 strips every image from history, so only the
    current turn is visual. Counting per message rather than per image keeps a
    multi-image turn (game-over visuals) together in one slot."""
    raw = os.environ.get("ARC3_HISTORY_IMAGE_KEEP", "").strip()
    if not raw:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return None



def _message_has_image(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for part in content
    )


def _strip_message_images(message: dict[str, Any]) -> dict[str, Any]:
    """Replace image parts with a short text marker. The marker keeps the
    message well formed (a message whose only part was an image would
    otherwise be empty) and tells the model a visual was there rather than
    silently showing a turn that never had one."""
    content = message.get("content")
    if not isinstance(content, list):
        return message
    parts: list[Any] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            parts.append({"type": "text", "text": _HISTORY_IMAGE_PLACEHOLDER})
        else:
            parts.append(part)
    return {**message, "content": parts}


def _apply_history_image_window(
    messages: list[dict[str, Any]], keep: int
) -> list[dict[str, Any]]:
    """Keep images on the `keep` most recent image-bearing messages, strip the
    rest. Walks backwards so recency wins; originals are never mutated."""
    seen = 0
    result: list[dict[str, Any]] = []
    for message in reversed(messages):
        if _message_has_image(message):
            seen += 1
            if seen > keep:
                message = _strip_message_images(message)
        result.append(message)
    result.reverse()
    return result


# Tag for the pair of messages a rolling summary is made of: the request that
# asks for it and the reply that is it. Deliberately absent from
# _CONTROL_KIND_ENV, so no pruning knob can remove them - the trimmer's
# summary-aware drain navigates by these, and losing one would silently return
# it to ordinary block dropping.
# Below this a reply that also called a tool is a preamble rather than a
# summary. Real ones measured 4,385 to 6,248 characters; an introduction to a
# tool call measured nine.




_CONTROL_KIND_ENV = {
    "nudge": "ARC3_PRUNE_NUDGES",
    "resume": "ARC3_PRUNE_RESUME_PROMPTS",
    "stub": "ARC3_PRUNE_DEAD_STUBS",
}



def _mark_control_message(message: dict[str, Any], kind: str = "resume") -> dict[str, Any]:
    message[_CONTROL_MESSAGE_KEY] = kind
    return message


def _strip_control_keys(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wire-safe copy: the private tag never leaves the process."""
    cleaned: list[dict[str, Any]] = []
    for message in messages:
        if _CONTROL_MESSAGE_KEY in message:
            copy = dict(message)
            copy.pop(_CONTROL_MESSAGE_KEY, None)
            cleaned.append(copy)
        else:
            cleaned.append(message)
    return cleaned



def _reasoning_history_keys() -> tuple[str, ...]:
    """Message key(s) under which prior reasoning is sent back to the server.

    Backends disagree: OpenRouter returns and reads `reasoning`, while
    llama.cpp returns `reasoning_content` and its chat template reads ONLY
    that - so reasoning stored under `reasoning` is silently discarded there
    (measured: 69 prompt tokens vs 3670 for the same text under the two keys).
    A dropped reasoning block is invisible except as prompt growth that falls
    short of the tokens generated.

    Default `reasoning` preserves existing behaviour. Set to
    `reasoning_content` for llama.cpp, or to a comma list to send both - which
    is backend-agnostic but risks a template that renders both keys duplicating
    the text."""
    raw = os.environ.get("ARC3_REASONING_HISTORY_KEY", "").strip()
    keys = tuple(k.strip() for k in raw.split(",") if k.strip())
    return keys or ("reasoning",)



def _preserve_thinking_kwarg() -> bool | None:
    """Value for the `preserve_thinking` chat-template kwarg, or None to omit.

    The payload builder sets chat_template_kwargs wholesale for vLLM, which
    replaces rather than merges - so a server-side
    --default-chat-template-kwargs carrying preserve_thinking can be silently
    dropped by a request. Defaults to on so the request asserts it rather than
    depending on server-side merge semantics; set to an empty string to omit
    the key entirely and leave the server default in charge."""
    raw = os.environ.get("ARC3_PRESERVE_THINKING", "1").strip().lower()
    if not raw:
        return None
    return raw in ("1", "true", "yes", "on")


def _openrouter_provider_prefs(provider: str) -> dict[str, Any] | None:
    """OpenRouter request-level provider pinning. ARC3_OPENROUTER_PROVIDER
    holds a provider name (or a comma-separated preference order); routing is
    then restricted to that list with fallbacks disabled, so every request in
    a run is served by a known engine. Under :nitro-style routing, providers
    differ in media handling (some silently drop video/image parts),
    quantization, and sampling behavior - a silent confound for any controlled
    comparison. Unset or non-OpenRouter: no change to the request."""
    if str(provider or "").strip().lower() != "openrouter":
        return None
    raw = os.environ.get("ARC3_OPENROUTER_PROVIDER", "").strip()
    if not raw:
        return None
    order = [name.strip() for name in raw.split(",") if name.strip()]
    if not order:
        return None
    return {"order": order, "allow_fallbacks": False}














def _message_display_text(message: dict[str, Any]) -> str:
    """Transcript rendering of a built user message. Content-parts messages
    carry their text in parts; images are noted rather than dumped."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    pieces: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "image_url":
            pieces.append("[image attached]")
        else:
            text = str(part.get("text") or "").strip()
            if text:
                pieces.append(text)
    return "\n\n".join(pieces)



def _degeneracy_ratio_threshold() -> float:
    """Compression ratio below which text is treated as degenerate.

    Measured on representative text: varied prose 0.20-0.28, a coordinate
    list 0.28, an ASCII board map 0.016, one sentence repeated 300 times
    0.008, the field mojibake spiral 0.0016. The default sits below the board
    map so a pasted grid is not falsely rejected, while still catching genuine
    repetition loops by a wide margin. Rejecting on compressibility catches
    those generally rather than one encoding artifact, and does not penalise
    legitimate unicode. 0 disables the check."""
    try:
        return max(0.0, float(os.environ.get("ARC3_DEGENERACY_RATIO", "0.01")))
    except ValueError:
        return 0.01


def _degeneracy_min_chars() -> int:
    """Length floor for the degeneracy check: short strings compress badly for
    uninteresting reasons, so testing them would produce false positives."""
    return max(1, _get_env_int("ARC3_DEGENERACY_MIN_CHARS", 500))


def _is_degenerate_text(value: str) -> bool:
    threshold = _degeneracy_ratio_threshold()
    if threshold <= 0:
        return False
    text = str(value or "")
    if len(text) < _degeneracy_min_chars():
        return False
    try:
        import zlib

        encoded = text.encode("utf-8", "replace")
        return len(zlib.compress(encoded, 6)) / max(1, len(encoded)) < threshold
    except Exception:
        return False



# Only SHORT decorations are stripped. A long parenthetical or emphasised aside
# means the line is a sentence rather than a header, and removing it could turn
# prose into a spurious match - so the header is left intact and simply fails to
# match, which loses the section rather than mis-capturing it.



def _animation_timeline_enabled() -> bool:
    """Whether the diff timeline is offered alongside the raw frames.

    The timeline is a cheap first look, but a summary that reads coherently
    invites stopping at it: on one level it produced a confident and wrong
    mechanic that the model only corrected after going to the frames. Off, the
    model computes its own comparison from `animation_frames` - a few lines it
    writes readily - at the cost of a snippet before it learns anything, which
    is wasted on animations that turn out to be irrelevant.

    Only an experiment settles which costs more.
    """
    return _get_env_bool("ARC3_ANIMATION_TIMELINE", True)



# Three bands, a million apart, so no calculated value can cross between them.
# The maximum _game_priority is about 1,325.
#
#   ~2,000,000   never trimmed: the original prefix is whole and this game has
#                never paid a prefill, so it is the cheapest of all to continue
#   ~1,000,000   warm again: trimmed once, prefix rebuilt. Ordered within the
#                band by the estimate at that trim, so a game that was doing
#                well sits above one that was stalling
#      0..1,325  awaiting prefill: the trim invalidated the prefix and the next
#                request has to pay for it anyway, which is the one moment
#                yielding the slot is free
#
# The old scheme sent a flat 999999 while the prefix held, so every game tied at
# startup and the server broke ties by arrival. Subtracting the queue position
# gives a deterministic order instead.
# ---------------------------------------------------------------- TEMPORARY
# Diagnostic for "the backend never shows more than 13 concurrent requests with
# 14 games". Records where each game thread is and prints a census once a
# second, so the missing one can be located rather than guessed at.
# Remove once the cause is known.
_GATE_STATS = {"enqueued": 0, "blocked": 0, "blocked_ms": 0, "handover_ms": 0}
_DIAG_LOCK = threading.Lock()
_DIAG_STATE: dict[str, str] = {}
_DIAG_LAST = [0.0]


def _diag(agent: Any, where: str) -> None:
    if not _get_env_bool("ARC3_DIAG_CONCURRENCY", False):
        return
    name = getattr(agent, "_diag_name", "?")
    now = time.monotonic()
    with _DIAG_LOCK:
        _DIAG_STATE[name] = where
        if now - _DIAG_LAST[0] < 1.0:
            return
        _DIAG_LAST[0] = now
        counts: dict[str, int] = {}
        for value in _DIAG_STATE.values():
            counts[value] = counts.get(value, 0) + 1
        total = len(_DIAG_STATE)
        inflight = counts.get("request", 0)
        summary = "  ".join(
            f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
        )
        log.warning(
            "DIAG %d games, %d in flight | %s | gate: %d handovers, %d blocked, "
            "%dms blocked, %dms total",
            total, inflight, summary,
            _GATE_STATS["enqueued"], _GATE_STATS["blocked"],
            _GATE_STATS["blocked_ms"], _GATE_STATS["handover_ms"],
        )
        if inflight < total:
            others = sorted(
                f"{g}:{w}" for g, w in _DIAG_STATE.items() if w != "request"
            )
            log.warning("DIAG   elsewhere: %s", ", ".join(others))


def _priority_refresh_enabled() -> bool:
    return any(_get_env_bool(name, False) for name in (
        "ARC3_PRIORITY_REFRESH_QUEUE", "ARC3_PRIORITY_PACE", "ARC3_PRIORITY_TAIL_FADE",
    ))


def _priority_action_count(summary: dict[str, Any]) -> int:
    return int(summary.get("end_action_num") or 0)






class _PriorityGate:
    """Harness-side admission control, for when the server ignores `priority`.

    The request field is a hint a backend may drop; this enforces the same
    ordering locally by limiting how many games hold a slot at once and
    choosing who gets a freed one.

    A game keeps its slot while its prefix is intact, because interrupting it
    there throws away a cache entry about to be reused. When the trimmer drops
    a message the prefix is invalid anyway and the next request must be
    prefilled from the divergence point - the one moment when yielding is free -
    so that is where the handover happens.

    Release and re-enqueue are one critical section. Releasing first would leave
    a window in which a lower-priority waiter takes the slot the caller was
    about to compete for.
    """

    def __init__(self, slots: int) -> None:
        self._cond = threading.Condition()
        self._free = max(1, int(slots))
        self._waiting: list[tuple[int, int]] = []
        self._seq = 0
        self._admitted: set[int] = set()
        self._snapshots: dict[int, PrioritySnapshot] = {}
        self._clock_start: float | None = None
        self._clock_deadline: float | None = None
        self._tail_fade_announced = False

    def configure_clock(self, started_at: float, deadline: float | None) -> None:
        """One shared monotonic clock, configured before initial admission.

        Earliest session start/deadline wins; appropriate for simultaneously
        launched sessions. No clock is inferred from the last action payload.
        """
        if _get_env_bool("ARC3_PRIORITY_TAIL_FADE", False) and deadline is None:
            raise ValueError("ARC3_PRIORITY_TAIL_FADE requires a finite runtime or soft deadline")
        with self._cond:
            self._clock_start = min(self._clock_start, started_at) if self._clock_start is not None else started_at
            if deadline is not None:
                self._clock_deadline = min(self._clock_deadline, deadline) if self._clock_deadline is not None else deadline

    def _snapshot_priority(self, snapshot: PrioritySnapshot, now: float) -> int:
        elapsed = max(0.0, now - self._clock_start) if self._clock_start is not None else 0.0
        endgame = False
        tail_fraction = None
        if _get_env_bool("ARC3_PRIORITY_TAIL_FADE", False):
            if self._clock_deadline is None or self._clock_start is None:
                raise ValueError("ARC3_PRIORITY_TAIL_FADE requires a finite runtime or soft deadline")
            fraction_setting = os.environ.get("ARC3_PRIORITY_TAIL_FADE_FRACTION", "").strip()
            if fraction_setting:
                try:
                    fraction = float(fraction_setting)
                except ValueError:
                    raise ValueError("ARC3_PRIORITY_TAIL_FADE_FRACTION must be > 0 and <= 1") from None
                if not 0.0 < fraction <= 1.0:
                    raise ValueError("ARC3_PRIORITY_TAIL_FADE_FRACTION must be > 0 and <= 1")
                # Use the full shared run duration, not the shrinking time left.
                window = fraction * max(0.0, self._clock_deadline - self._clock_start)
            else:
                window = max(1.0, 60 * _get_env_float("ARC3_PRIORITY_TAIL_FADE_MINUTES", 90.0))
            remaining = max(0.0, self._clock_deadline - now)
            tail_fraction = remaining / window if window > 0.0 else 0.0
            if remaining <= window:
                # Snapshot scoring also happens outside _pump's lock. Announce
                # once per shared gate, even when several games trim together.
                with self._cond:
                    if not self._tail_fade_announced:
                        self._tail_fade_announced = True
                        log.warning(
                            "tail fade phase after %.1f minutes: %.1f minutes "
                            "remaining; fading tail B to zero over the final "
                            "%.1f minutes, retaining hazard C",
                            elapsed / 60.0, remaining / 60.0, window / 60.0,
                        )
        return priority_value(
            snapshot, endgame=endgame, tail_fraction=tail_fraction,
            human_actions=_priority_human_actions(), action_scale=_priority_action_scale(),
            token_scale=_priority_token_scale(), action_weight=_priority_action_weight(),
            tail_base=_priority_tail_base(), tail_cap=_priority_tail_cap(),
            **_priority_level_options(),
        )

    def acquire(self, priority: int) -> None:
        with self._cond:
            self._enqueue_and_wait(priority)

    def release(self) -> None:
        with self._cond:
            self._free += 1
            self._pump()

    def handover(self, priority: int, snapshot: PrioritySnapshot | None = None) -> None:  # noqa: D401
        """Give up the slot and immediately compete for it again.

        The caller may win it straight back, which is the right outcome when
        nothing else is waiting or everything else is further along.
        """
        _t0 = time.monotonic()
        with self._cond:
            self._free += 1
            self._enqueue_and_wait(priority, snapshot)
        _GATE_STATS["handover_ms"] += int(1000 * (time.monotonic() - _t0))

    def _enqueue_and_wait(self, priority: int, snapshot: PrioritySnapshot | None = None) -> None:
        # TEMPORARY: with slots == games nobody should ever block here
        _GATE_STATS["enqueued"] += 1
        self._seq += 1
        token = self._seq
        if snapshot is not None:
            self._snapshots[token] = snapshot
        # negated so the heap pops the HIGHEST priority; the sequence number
        # breaks ties by arrival
        heapq.heappush(self._waiting, (-int(priority), token))
        self._pump()
        if token not in self._admitted:
            _GATE_STATS["blocked"] += 1
            t0 = time.monotonic()
            while token not in self._admitted:
                self._cond.wait()
            _GATE_STATS["blocked_ms"] += int(1000 * (time.monotonic() - t0))
        self._admitted.discard(token)

    def _pump(self) -> None:
        if _priority_refresh_enabled() and self._free > 0:
            now = time.monotonic()
            self._waiting = [
                (-self._snapshot_priority(self._snapshots[token], now), token)
                if token in self._snapshots else (priority, token)
                for priority, token in self._waiting
            ]
            heapq.heapify(self._waiting)
        woke = False
        while self._free > 0 and self._waiting:
            _, token = heapq.heappop(self._waiting)
            self._snapshots.pop(token, None)
            self._admitted.add(token)
            self._free -= 1
            woke = True
        if woke:
            self._cond.notify_all()


def _max_active_streams() -> int:
    """Games allowed to hold a slot at once. 0 disables the gate entirely.

    Below the game count this staggers the start - a slot only frees when a
    game's context is first trimmed, which measurements put 30 to 50 turns in -
    so early games run alone for a while and late ones begin much later. That is
    the point: a prefix is reused for as long as it lasts, instead of fourteen
    games each progressing at a fourteenth of the rate.
    """
    return max(0, _get_env_int("ARC3_MAX_ACTIVE_STREAMS", 0))


_PRIORITY_GATE: _PriorityGate | None = None
_PRIORITY_GATE_LOCK = threading.Lock()


def _priority_gate() -> _PriorityGate | None:
    """Process-wide, built on first use. Games run as threads in one process,
    so a module-level gate needs no plumbing and nothing to pickle."""
    global _PRIORITY_GATE
    slots = _max_active_streams()
    if slots <= 0:
        return None
    with _PRIORITY_GATE_LOCK:
        if _PRIORITY_GATE is None:
            _PRIORITY_GATE = _PriorityGate(slots)
            log.warning("priority gate active: %d concurrent streams", slots)
        return _PRIORITY_GATE


_PRIORITY_BAND = 1_000_000
_PRIORITY_UNTRIMMED_BASE = 2_000_000


def _priority_level_options() -> dict[str, Any]:
    # Keep boolean lookup settings backward compatible. The explicit
    # "remaining" mode selects the 8/7/5/0 tail, already at its final scale.
    lookup = os.environ.get("ARC3_PRIORITY_TAIL_LOOKUP", "").strip().lower()
    return {
        "normalize_score": _get_env_bool("ARC3_PRIORITY_SCORE_NORMALIZATION", False),
        "tail_lookup": (
            "remaining" if lookup == "remaining"
            else _get_env_bool("ARC3_PRIORITY_TAIL_LOOKUP", False)
        ),
        "tail_efficiency": _get_env_float("ARC3_PRIORITY_TAIL_EFFICIENCY", 0.8),
    }


def _priority_action_scale() -> float:
    return max(1.0, _get_env_float("ARC3_PRIORITY_ACTION_SCALE", 115.0))


def _priority_token_scale() -> float:
    return max(1.0, _get_env_float("ARC3_PRIORITY_TOKEN_SCALE", 62000.0))


def _priority_action_weight() -> float:
    """How much of `c` comes from the action curve rather than the token one.

    A blend rather than max(actions, tokens): max sits below both curves, so it
    demotes harder than either measure alone warrants. Replaying four runs
    against their own level-up costs, max was the weakest of every combination
    tried, while the weight itself barely mattered - 0.25 to 1.0 all landed
    within a few percent of each other.
    """
    return min(1.0, max(0.0, _get_env_float("ARC3_PRIORITY_ACTION_WEIGHT", 0.25)))


def _priority_tail_cap() -> float:
    """Ceiling on `b`, the value of every level after the current one.

    min(cap, level + 1) rather than a constant. Chaining the measured
    completion rates makes `b` nearly flat, which is why a constant looked
    right - but that chain prices expected future score, and what the scheduler
    needs is the term that lets depth outrank freshness. Replayed, a flat tail
    was the worst option tested at every constrained budget; the cap matters
    more than the slope, and a lower cap is better because it leaves more of
    the ordering to `a`.
    """
    return max(0.0, _get_env_float("ARC3_PRIORITY_TAIL_CAP", 4.0))


def _priority_tail_base() -> float:
    """Value of `b` at level 1, before the per-level rise.

    b = min(cap, base + level - 1). Base 2 with cap 4 gives the fitted form,
    2 3 4 4 4...; base 6 with cap 6 gives a flat tail, which is what this
    replaced and is worth being able to get back for a comparison.
    """
    return max(0.0, _get_env_float("ARC3_PRIORITY_TAIL_BASE", 2.0))


def _priority_human_actions() -> float:
    """Denominator offset in the efficiency term.

    The benchmark scores a level by min(1, (human/used)^2). The level is not
    finished yet, so the term belongs at the expected completion point rather
    than the current count - hence an offset instead of a bare division.
    """
    return max(1.0, _get_env_float("ARC3_PRIORITY_HUMAN_ACTIONS", 25.0))



# Endgame priorities sit in the upper half of whichever warmth band the game is
# in: the cache argument for the bands is orthogonal to the phase, so a game
# about to pay a prefill still yields to one that has a warm prefix. The widest
# value either phase can produce is under 2,000 even at level 15, so half a
# band is ample.
_PRIORITY_ENDGAME_OFFSET = 500_000


def _game_priority(
    level: int,
    actions_on_level: int,
    tokens_on_level: int = 0,
    endgame: bool = False,
    total_levels: int | None = None,
) -> int:
    """Expected remaining score for a game on `level` after this much spend.

    Two phases. In the main one, (a + b) * c:

      a  what finishing THIS level banks - the score becomes `level` levels
         completed, discounted by the benchmark's efficiency term
      b  every level after it, min(cap, base + level - 1)
      c  whether this level still yields anything, as a blend of two measured
         decay curves

    In the endgame there is no time to reach a further level, so the tail and
    the hazard stop meaning anything and what remains is simply what completing
    this level would bank. The 0.25 floor lets efficiency separate games at the
    same depth without letting a shallower one overtake a deeper one.

    Replaying four runs against their own level-up costs, the endgame switch was
    worth 8 to 12% of total score at constrained budgets and nothing at all once
    the budget covered what the games actually spent - which is where real runs
    sit, so this matters only if the competition budget is tighter.
    """
    level = max(1, int(level))
    actions = max(0, int(actions_on_level))
    tokens = max(0, int(tokens_on_level))
    options = _priority_level_options()
    if options["normalize_score"] or options["tail_lookup"]:
        # Same scoring as the refreshed queue, retaining this legacy path's
        # phase offset. With both flags off, execute the original path below.
        value = priority_value(
            PrioritySnapshot(level, actions, tokens, total_levels=total_levels),
            endgame=endgame, human_actions=_priority_human_actions(),
            action_scale=_priority_action_scale(), token_scale=_priority_token_scale(),
            action_weight=_priority_action_weight(), tail_base=_priority_tail_base(),
            tail_cap=_priority_tail_cap(), **options,
        )
        return value + (_PRIORITY_ENDGAME_OFFSET if endgame else 0)
    human = _priority_human_actions()
    efficiency = min(1.0, (human / (human + actions)) ** 2)

    if endgame:
        value = level * (0.25 + efficiency)
        return _PRIORITY_ENDGAME_OFFSET + max(1, int(value * 100))

    a = level * efficiency
    b = min(_priority_tail_cap(), _priority_tail_base() + level - 1.0)
    weight = _priority_action_weight()
    by_actions = actions / _priority_action_scale()
    by_tokens = tokens / _priority_token_scale()
    c = (
        weight * 0.5 ** (by_actions * by_actions)
        + (1.0 - weight) * 0.5 ** (by_tokens * by_tokens)
    )
    return max(1, int((a + b) * c * 100))


def _persistent_functions_scope() -> str:
    """How long a kept helper lives: 'levelup' (default), 'turn', or 'game'.

    'turn' clears at every analyzer invocation that is not a resumption, so a
    turn that inspects, acts and yields loses its toolkit even though the
    investigation continues. That is conservative but costly: the helpers the
    keepability check lets through are pure transforms taking the board as an
    argument, and those do not care that an action happened.

    'levelup' keeps them until the level changes, which is where a helper can
    genuinely go wrong - not by holding stale data, which the check already
    prevents, but by encoding an assumption in its code, like a glyph parser
    with row offsets from the previous layout.

    'game' keeps helpers across turns and level transitions. The model is
    reminded to recheck level-specific assumptions. New game sessions still
    start with an empty pool; snippet variables remain ephemeral in every scope.
    """
    scope = os.environ.get("ARC3_PERSISTENT_FUNCTIONS_SCOPE", "").strip().lower()
    return scope if scope in ("turn", "levelup", "game") else "levelup"



def _guards_active(level: Any) -> bool:
    """Whether the per-action guards apply at this level.

    Level 1 pays the smallest share of the score - one part in N - and it is
    where the model knows least, so it probes hardest: repeating an action to
    see whether the repeat differs, retrying what killed it to learn what
    killed it. The guards read that as waste. ARC3_GUARDS_FROM_LEVEL=2 lets
    level 1 run unguarded.

    Covers the stale block and both repeat guards. NOT the batch no-op stop,
    which costs one action of a planned sequence rather than refusing a
    deliberate probe, and not the terminal latch, which stops a snippet acting
    against a board that no longer exists.
    """
    try:
        floor = max(1, int(os.environ.get("ARC3_GUARDS_FROM_LEVEL", "1") or 1))
    except ValueError:
        floor = 1
    try:
        return int(level) >= floor
    except (TypeError, ValueError):
        return True  # unknown level: guard rather than not


def _sim_guard_enabled() -> bool:
    """r15: stop a batch at the first action the model's own observe/step mispredicts."""
    from inference.agent import sim_check

    return sim_check.guard_enabled()


def _persistent_functions() -> bool:
    """Retain eligible function sources across calls; off by default.

    Scope is controlled by ARC3_PERSISTENT_FUNCTIONS_SCOPE. State and variables
    are never serialized. Definitions are validated and recreated in each fresh
    sandbox, against its current harness globals.
    """
    return _get_env_bool("ARC3_PERSISTENT_FUNCTIONS", False)


def _animation_enabled() -> bool:
    """Whether intermediate animation frames are captured and offered.

    Off by default: this is new information for the model rather than a fix to
    something broken, so it needs to be measurable on its own."""
    return _get_env_bool("ARC3_ANIMATION", False)






def _request_tool_choice(tools: list[dict[str, Any]] | None) -> str | None:
    """Tool-choice sent with tool-bearing requests. LOCAL_ANALYZER_TOOL_CHOICE
    overrides the default "auto": set "omit"/"none" to drop the field for
    servers whose validation rejects "auto" without a tool-call parser (tools
    are still sent, so chat templates still render tool instructions); any
    other value (e.g. "required") passes through verbatim."""
    if not tools:
        return None
    choice = os.environ.get("LOCAL_ANALYZER_TOOL_CHOICE", "auto").strip().lower()
    if choice in ("", "omit", "none"):
        return None
    return choice


def _trim_log_text(text: str, *, max_chars: int = _RESPONSE_META_MAX_CHARS) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    omitted = len(stripped) - max_chars
    return f"{stripped[:max_chars].rstrip()}\n... [truncated {omitted} chars]"


def _format_model_response_meta(
    *,
    finish_reason: str,
    served_by: str,
    reasoning: str,
    content: str,
    tool_calls: list[dict[str, Any]],
    tool_call_markup_in_text: bool,
    recovered_tool_calls_from_markup: bool,
    malformed_argument_errors: list[str],
) -> str:
    lines = [
        f"finish_reason: {finish_reason or '(empty)'}",
        f"served_by: {served_by or '(unknown)'}",
        f"tool_call_count: {len(tool_calls)}",
        f"content_chars: {len(content)}",
        f"reasoning_chars: {len(reasoning)}",
        f"tool_call_markup_in_text: {'yes' if tool_call_markup_in_text else 'no'}",
        f"tool_calls_recovered_from_markup: {'yes' if recovered_tool_calls_from_markup else 'no'}",
    ]
    if malformed_argument_errors:
        lines.append("tool_call_argument_issues:")
        lines.extend(f"- {issue}" for issue in malformed_argument_errors)
    if tool_calls:
        lines.append("raw_tool_calls:")
        lines.append(_trim_log_text(json.dumps(tool_calls, indent=2, ensure_ascii=True)))
    return "\n".join(lines)






def _build_system_prompt(*, tool_output_chars: int) -> str:
    """The system prompt: every rule stated once (see inference.agent.prompts)."""
    prompt = build_system_prompt(
        output_chars=tool_output_chars,
        border=_get_env_int("ARC3_NOOP_GUARD_BORDER", 4),
        undo=undo_exposure_mode() == "on",
        reset=reset_exposed(),
    )
    # ARC3_SYSTEM_PROMPT_PREFIX: model-specific control directives (e.g. "/think").
    prefix = os.environ.get("ARC3_SYSTEM_PROMPT_PREFIX", "").strip()
    return f"{prefix}\n\n{prompt}" if prefix else prompt


def _tool_output_chars() -> int:
    return max(256, max(64, _LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS) * 4)


def effective_system_prompt() -> str:
    """The exact system prompt every analyzer builds under the current environment."""
    return _build_system_prompt(tool_output_chars=_tool_output_chars())


@dataclass(frozen=True)
class AnalyzerModelConfig:
    provider: str
    base_url: str
    model_id: str


@dataclass(frozen=True)
class AnalyzerTurnResult:
    step_executed: bool
    retryable_failure: bool = False
    reasoning: str = ""
    yielded_control: bool = False


@dataclass(frozen=True)
class _ToolDispatchResult:
    content: str
    step_executed: bool = False
    # True when an executed action changed the board outside the HUD border, or
    # ended the level or the attempt. "Executed" alone is too weak a signal for
    # the reasoning-effort ladder: bumping a wall executes and achieves nothing,
    # and resetting on it lets the model oscillate between truncating at full
    # effort and acting inertly at reduced effort, never settling.
    made_progress: bool = False


@dataclass(frozen=True)
class _AsciiFrameView:
    ascii: str
    step: int
    level: int
    shape: tuple[int, int]

    def __str__(self) -> str:
        rows, cols = self.shape
        return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

    __repr__ = __str__




def _to_ascii_frame_view(frame: Frame | None) -> _AsciiFrameView | None:
    if frame is None:
        return None
    return _AsciiFrameView(
        ascii=frame.ascii,
        step=frame.step,
        level=frame.level,
        shape=frame.shape,
    )




def _ascii_frame_view_payload(frame: Frame | None) -> dict[str, Any] | None:
    view = _to_ascii_frame_view(frame)
    if view is None:
        return None
    return {
        "ascii": view.ascii,
        "step": view.step,
        "level": view.level,
        "shape": [int(view.shape[0]), int(view.shape[1])],
        "grid": [list(row) for row in frame.grid],
    }


def _ascii_history_view_payload(history_entries: list[HistoryEntry]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for entry in history_entries:
        frame_payload = _ascii_frame_view_payload(entry.frame)
        if frame_payload is None:
            continue
        payload.append({
            "action": entry.action, "frame": frame_payload,
            "result": dict(getattr(entry, "result", {}) or {}),
        })
    return payload


_VISION_PATCH_PIXELS = 32          # Qwen3.6: 16px patches, 2x2 merge
_VISION_SENTINEL_TOKENS = 2        # vision start/end markers
_IMAGE_TOKENS_FALLBACK = 402       # 64-cell board at MULTIMODAL_UPSCALE=10
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _text_token_chars() -> int:
    """Characters per token for the text heuristic - the SEED value only.

    Measured against a server-reported prompt_tokens on a real request: 25,189
    characters of system prompt plus opener tokenised to 5,892, i.e. about 4.3
    characters per token. The 3 default therefore over-counts text by roughly
    40%, so the trimmer evicts history the budget could have held. This value
    is used until the first response supplies a measurement; after that
    ARC3_CALIBRATE_TEXT_TOKENS drives the divisor from the server's own
    figures."""
    return max(1, _get_env_int("ARC3_TEXT_TOKEN_CHARS", 3))


def _calibrate_text_tokens() -> bool:
    """Derive the characters-per-token divisor from the server's reported
    prompt_tokens rather than a fixed guess.

    Every response carries prompt_tokens for the request that produced it - an
    exact count including the chat template. Subtracting the vision cost of the
    request's images leaves the text, and dividing by the text characters gives
    the divisor for the next estimate. No extra requests, no tokenizer, and it
    tracks content: code-heavy turns tokenise differently from prose.

    The last measurement is used rather than an average: history carries
    forward, so the prompt just measured is mostly the prompt about to be
    built, and averaging would blend in content already evicted."""
    return _get_env_bool("ARC3_CALIBRATE_TEXT_TOKENS", True)


# Asymmetric on purpose. Over-counting trims history and degrades one game;
# under-counting overflows the context, and an overflowing request holds a
# slot and can pressure the others. Measured across 22 games: 2.75-3.69,
# mean 3.12. The 3.3 ceiling therefore binds on the sparsest games by
# design - a deliberate haircut - while the floor is set low enough that
# only an arithmetic failure reaches it, and such a game is likely lost
# anyway.
def _text_token_chars_min() -> float:
    """Floor for the calibrated divisor. Low enough that only an arithmetic
    failure reaches it, and a game in that state is likely lost anyway."""
    # a divisor of zero would make every estimate infinite and trim the whole
    # history on the first request
    return max(0.1, _get_env_float("ARC3_TEXT_TOKEN_CHARS_MIN", 1.0))


def _text_token_chars_max() -> float:
    """Ceiling for the calibrated divisor.

    Exposed because 3.3 is a fitted constant, not a property of the tokeniser:
    it came from 22 games measuring 2.75-3.69, so it binds on the sparsest
    boards by design. A run whose prompts are denser than that sample - a
    different game set, a longer system prompt, another model - is being
    trimmed harder than its own measurements justify, and raising this is how
    that is tested.
    """
    # never below the floor: an inverted pair would clamp every measurement to
    # the ceiling and silently invert the guard's meaning
    return max(_text_token_chars_min(),
               _get_env_float("ARC3_TEXT_TOKEN_CHARS_MAX", 3.3))


def _image_token_estimate_enabled() -> bool:
    """Count image parts by their real vision-token cost instead of by the
    length of their base64 payload. Without this an image is charged roughly
    one token per three base64 characters - thousands for a board render whose
    true cost is a few hundred - so enabling images silently evicts real
    history."""
    return _get_env_bool("ARC3_IMAGE_TOKEN_ESTIMATE", True)


def _png_dimensions(data_url: str) -> tuple[int, int] | None:
    """Width/height from a base64 PNG data URL, read from the IHDR header
    without decoding the image. Only the first 24 bytes are needed."""
    marker = "base64,"
    index = data_url.find(marker)
    if index < 0:
        return None
    head = data_url[index + len(marker):index + len(marker) + 32]
    if len(head) < 32:
        return None
    try:
        raw = base64.b64decode(head, validate=True)
    except Exception:
        return None
    if len(raw) < 24 or not raw.startswith(_PNG_SIGNATURE) or raw[12:16] != b"IHDR":
        return None
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    if width <= 0 or height <= 0:
        return None
    return width, height


def _image_part_tokens(part: dict[str, Any]) -> int:
    """Vision tokens for one image content part: one token per merged patch
    plus the two sentinel tokens. ARC3_IMAGE_TOKENS_FLAT overrides the
    computation (useful for non-Qwen backends) and is also the fallback when
    the header cannot be read."""
    flat = _get_env_int("ARC3_IMAGE_TOKENS_FLAT", 0)
    if flat > 0:
        return flat
    url = ""
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        url = str(image_url.get("url") or "")
    dims = _png_dimensions(url) if url else None
    if dims is None:
        return _IMAGE_TOKENS_FALLBACK
    width, height = dims
    cells = -(-width // _VISION_PATCH_PIXELS) * -(-height // _VISION_PATCH_PIXELS)
    return cells + _VISION_SENTINEL_TOKENS


def _split_messages_for_estimate(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Return (messages with image payloads replaced by a short placeholder,
    total vision tokens). Originals are never mutated."""
    image_tokens = 0
    rewritten: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            rewritten.append(message)
            continue
        parts: list[Any] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                image_tokens += _image_part_tokens(part)
                parts.append({"type": "image_url", "image_url": {"url": "<image>"}})
            else:
                parts.append(part)
        rewritten.append({**message, "content": parts})
    return rewritten, image_tokens


def _estimate_tokens(value: Any, chars_per_token: float | None = None) -> int:
    try:
        # ensure_ascii=False is deliberate: with escaping ON, every non-ASCII
        # character renders as a six-character \uXXXX sequence, so any prompt
        # containing arrows, box-drawing glyphs or a multiplication sign is
        # counted at up to 6x its real size. The trimmer decides when to evict
        # in these units, so the inflation makes it discard history it did not
        # need to - and the error grows with the share of non-ASCII text.
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        rendered = str(value)
    chars = chars_per_token or _text_token_chars()
    return max(1, int(len(rendered) / chars) + (1 if len(rendered) % chars else 0))


def _host_accessible_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").strip().lower()
    if hostname != "host.docker.internal":
        return base_url
    netloc = "127.0.0.1"
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


def _resolve_analyzer_model(model: str) -> AnalyzerModelConfig:
    requested = (model or "").strip()
    lowered = requested.lower()
    if lowered in {"local", "local-qwen", "qwen-local", "qwen"}:
        configured_base_url = os.environ.get("LOCAL_ANALYZER_BASE_URL", _LOCAL_ANALYZER_BASE_URL).strip()
        if not configured_base_url:
            raise ValueError("LOCAL_ANALYZER_BASE_URL must be set for the local analyzer preset.")

        provider = os.environ.get("LOCAL_ANALYZER_PROVIDER", os.environ.get("OPENAI_PROVIDER", "vllm")).strip().lower()
        if not provider:
            provider = "vllm"
        model_id = os.environ.get("LOCAL_ANALYZER_MODEL_ID", "").strip() or _LOCAL_ANALYZER_MODEL_ID.strip()
        if not model_id:
            raise ValueError("LOCAL_ANALYZER_MODEL_ID must be set for the local analyzer preset.")
        return AnalyzerModelConfig(
            provider=provider,
            base_url=_host_accessible_base_url(configured_base_url),
            model_id=model_id,
        )

    if not requested:
        requested = _LOCAL_ANALYZER_MODEL_ID.strip()
    if not requested:
        raise ValueError(
            "Analyzer model id is required. Set analyzer.model_id in config, pass --model, "
            "or set LOCAL_ANALYZER_MODEL_ID / INFERENCE_ANALYZER_MODEL."
        )

    provider = os.environ.get("OPENAI_PROVIDER", os.environ.get("LOCAL_ANALYZER_PROVIDER", "vllm")).strip().lower()
    if not provider:
        provider = "vllm"
    base_url = _host_accessible_base_url(
        os.environ.get("OPENAI_BASE_URL", os.environ.get("LOCAL_ANALYZER_BASE_URL", _LOCAL_ANALYZER_BASE_URL)).strip()
    )
    if not base_url:
        raise ValueError("OPENAI_BASE_URL or LOCAL_ANALYZER_BASE_URL must be set for direct model ids.")
    return AnalyzerModelConfig(provider=provider, base_url=base_url, model_id=requested)


def _append_transcript_section(log_path: Path, label: str, content: str) -> None:
    rendered_content = content.strip()
    if not rendered_content:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{label}]\n")
        f.write(rendered_content)
        f.write("\n\n")


def _render_transcript_section(label: str, content: str) -> str:
    rendered_content = content.strip()
    if not rendered_content:
        return ""
    return f"[{label}]\n{rendered_content}\n\n"


def _json_like_payload(value: Any) -> Any | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        return json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _render_scalar_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True)


def _render_human_readable_lines(value: Any, *, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]
        lines: list[str] = []
        for key, item in value.items():
            key_text = str(key)
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key_text}:")
                lines.extend(_render_human_readable_lines(item, indent=indent + 2))
                continue
            if isinstance(item, str) and "\n" in item:
                multiline = item.splitlines() or [""]
                lines.append(f"{prefix}{key_text}: |")
                lines.extend(f"{prefix}  {line}" for line in multiline)
                continue
            lines.append(f"{prefix}{key_text}: {_render_scalar_value(item)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_render_human_readable_lines(item, indent=indent + 2))
                continue
            if isinstance(item, str) and "\n" in item:
                multiline = item.splitlines() or [""]
                lines.append(f"{prefix}- |")
                lines.extend(f"{prefix}  {line}" for line in multiline)
                continue
            lines.append(f"{prefix}- {_render_scalar_value(item)}")
        return lines
    if isinstance(value, str):
        if "\n" in value:
            multiline = value.splitlines() or [""]
            return [f"{prefix}|", *(f"{prefix}  {line}" for line in multiline)]
        return [f"{prefix}{value}"]
    return [f"{prefix}{_render_scalar_value(value)}"]


def _render_human_readable_value(value: Any) -> str:
    return "\n".join(_render_human_readable_lines(value))


def _render_jsonish_text(value: Any) -> str:
    parsed = _json_like_payload(value)
    if parsed is not None:
        return _render_human_readable_value(parsed)
    return _normalize_message_content(value) if not isinstance(value, str) else value.strip()


def _render_tool_parameter_text(value: Any) -> str:
    if isinstance(value, str):
        return value.rstrip("\n")
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=True)
    return str(value)


def _normalize_tool_call_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return json.loads(json.dumps(arguments))
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        if stripped.startswith("<tool_call>"):
            recovered_tool_calls = _recover_tool_calls_from_markup(stripped)
            if recovered_tool_calls:
                recovered_arguments = recovered_tool_calls[0].get("function", {}).get("arguments", "{}")
                return json.loads(str(recovered_arguments))
            return {}
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("tool call arguments must decode to a JSON object")
    raise ValueError("tool call arguments must be a JSON object or JSON object string")


def _render_tool_call_markup(tool_name: str, arguments: Any) -> str:
    name = str(tool_name or "").strip()
    if not name:
        return ""
    try:
        parsed_arguments = _normalize_tool_call_arguments(arguments)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""

    lines = ["<tool_call>", f"<function={name}>"]
    for parameter_name, parameter_value in parsed_arguments.items():
        lines.append(f"<parameter={parameter_name}>")
        rendered_value = _render_tool_parameter_text(parameter_value)
        if rendered_value:
            lines.extend(rendered_value.splitlines())
        lines.append("</parameter>")
    lines.append("</function>")
    lines.append("</tool_call>")
    return "\n".join(lines)


def _render_tool_result_display(content: Any) -> str:
    parsed = _json_like_payload(content) if isinstance(content, str) else (content if isinstance(content, dict) else None)
    if isinstance(parsed, dict):
        stdout = str(parsed.get("stdout", "") or "").rstrip("\n")
        error = str(parsed.get("error", "") or "").rstrip("\n")
        result = parsed.get("result")
        has_result = result not in (None, "", [], {})
        # r10: harness-side notes travel in the JSON the model receives; the transcript used to drop
        # them whenever stdout was present (r9 analysis §5.1), which hid every stall note and the
        # timeout retention note from the logs.
        notes: list[str] = []
        for key in ("harness_note", "function_retention", "action_execution_errors"):
            value = parsed.get(key)
            if value not in (None, "", [], {}):
                rendered = value if isinstance(value, str) else _render_human_readable_value(value)
                notes.append(f"[{key}] {rendered}")
        if stdout and not error and not has_result and not notes:
            return stdout

        blocks: list[str] = []
        if stdout:
            blocks.append(stdout)
        if has_result:
            rendered_result = _render_human_readable_value(result)
            if stdout:
                blocks.append(f"result:\n{rendered_result}")
            else:
                blocks.append(rendered_result)
        if error:
            if stdout or has_result:
                blocks.append(f"error:\n{error}")
            else:
                blocks.append(error)
        blocks.extend(notes)
        if blocks:
            return "\n\n".join(block for block in blocks if block.strip())

    return _render_jsonish_text(content)


def _resolve_run_artifact_location(state_path: Path) -> tuple[Path, str | None]:
    parent = state_path.parent
    if parent.name == "artifacts" and parent.parent != parent:
        run_root = parent.parent
        runtime_state_files = list(parent.glob(f"*_{RUNTIME_STATE_FILENAME}"))
        if len(runtime_state_files) <= 1:
            return run_root, None
        runtime_state_stem = Path(RUNTIME_STATE_FILENAME).stem
        suffix = f"_{runtime_state_stem}"
        state_stem = state_path.stem
        game_stem = state_stem[:-len(suffix)] if state_stem.endswith(suffix) else state_stem
        return run_root, game_stem
    return parent, None


def _resolve_named_run_artifact(
    state_path: Path,
    *,
    default_name: str,
    per_game_suffix: str,
    directory_name: str | None = None,
) -> Path:
    run_root, game_stem = _resolve_run_artifact_location(state_path)
    output_root = run_root / directory_name if directory_name else run_root
    if game_stem:
        return output_root / f"{game_stem}{per_game_suffix}"
    return output_root / default_name


def _render_prompt_log_message(message: dict[str, Any]) -> str:
    role = str(message.get("role", "")).strip().upper() or "UNKNOWN"
    header = f"[{role}]"
    tool_call_id = str(message.get("tool_call_id", "")).strip()
    if role == "TOOL" and tool_call_id:
        header = f"[TOOL RESULT: {tool_call_id}]"
    blocks = [header]

    content = _normalize_message_content(message.get("content", ""))
    if content:
        blocks.append(_render_tool_result_display(content) if role == "TOOL" else content)

    reasoning = _extract_reasoning_text(message)
    if reasoning:
        blocks.append("[REASONING]")
        blocks.append(reasoning)

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        for tool_call in tool_calls:
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            name = str(function.get("name", "")).strip() or "unknown"
            blocks.append(f"[ASSISTANT TOOL CALL: {name}]")
            tool_call_id = str(tool_call.get("id", "")).strip()
            if tool_call_id:
                blocks.append(f"id: {tool_call_id}")
            rendered_tool_call = _render_tool_call_markup(name, function.get("arguments", "{}"))
            if rendered_tool_call:
                blocks.append(rendered_tool_call)
            else:
                raw_arguments = function.get("arguments", "{}")
                try:
                    parsed_arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                    rendered_arguments = json.dumps(parsed_arguments, indent=2, ensure_ascii=True)
                except (TypeError, ValueError, json.JSONDecodeError):
                    rendered_arguments = str(raw_arguments)
                blocks.append("arguments:")
                blocks.append(rendered_arguments if rendered_arguments.strip() else "{}")

    return "\n".join(blocks)


def _resolve_prompt_log_path(state_path: Path) -> Path:
    return _resolve_named_run_artifact(
        state_path,
        default_name="prompt.log",
        per_game_suffix=".log",
        directory_name="prompts",
    )


def _resolve_request_log_path(state_path: Path) -> Path:
    return _resolve_named_run_artifact(
        state_path,
        default_name="requests.jsonl",
        per_game_suffix="_requests.jsonl",
    )


def _append_request_snapshot(
    log_path: Path,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    event: str | None = None,
    tool_choice: str | None = None,
    finish_reason: str | None = None,
    served_by: str | None = None,
    analysis_step: int | None = None,
    action: int | None = None,
    request_index_within_turn: int | None = None,
    usage: dict[str, Any] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    response_message: dict[str, Any] | None = None,
    state_name: str | None = None,
) -> None:
    payload = {
        "messages": messages,
        "tools": tools or [],
    }
    if state_name:
        # Identifies the game: the log file name falls back to a shared default while only one game has state.
        payload["state_name"] = state_name
    if isinstance(response_message, dict):
        # The raw assistant output for this request, so each request/response pair is a complete record.
        payload["response_message"] = response_message
    if isinstance(chat_template_kwargs, dict) and chat_template_kwargs:
        payload["chat_template_kwargs"] = dict(chat_template_kwargs)
    if isinstance(usage, dict) and usage:
        payload["usage"] = usage
    if event:
        payload["event"] = event
    if tool_choice:
        payload["tool_choice"] = tool_choice
    if finish_reason is not None:
        payload["finish_reason"] = str(finish_reason)
    if served_by:
        payload["served_by"] = str(served_by)
    if analysis_step is not None:
        payload["analysis_step"] = analysis_step
    if action is not None:
        payload["action"] = action
    if request_index_within_turn is not None:
        payload["request_index_within_turn"] = request_index_within_turn
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                payload,
                ensure_ascii=True,
            )
        )
        f.write("\n")


def _write_prompt_log_snapshot(
    log_path: Path,
    *,
    model_id: str,
    base_url: str,
    display_action_num: int,
    analysis_step: int | None,
    request_index: int,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: str | None,
    transcript: str,
) -> None:
    rendered_messages = "\n\n".join(_render_prompt_log_message(message) for message in messages)
    rendered_tools: list[str] = []
    for tool in tools or []:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = str(function.get("name", "")).strip() or "unknown"
        description = str(function.get("description", "")).strip()
        if description:
            rendered_tools.append(f"- {name}: {description}")
        else:
            rendered_tools.append(f"- {name}")
    analysis_label = str(analysis_step) if analysis_step is not None else "n/a"
    transcript_text = transcript.strip()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("LATEST MODEL CALL SNAPSHOT\n")
        f.write(f"model: {model_id}\n")
        f.write(f"base_url: {base_url}\n")
        f.write(f"analysis_step: {analysis_label}\n")
        f.write(f"action: {display_action_num}\n")
        f.write(f"request_index_within_turn: {request_index}\n")
        f.write(f"message_count: {len(messages)}\n")
        f.write(f"tool_choice: {tool_choice or '(none)'}\n")
        f.write("\n[AVAILABLE TOOLS]\n")
        f.write("\n".join(rendered_tools) if rendered_tools else "(none)")
        f.write("\n\n[MODEL INPUT]\n")
        f.write(rendered_messages.strip())
        f.write("\n\n[TURN TRANSCRIPT SO FAR]\n")
        f.write(transcript_text)
        f.write("\n")


def _normalize_message_content(content: Any) -> str:
    def _strip_think_tags(text: str) -> str:
        cleaned = _THINK_TAG_RE.sub("", text)
        cleaned = "\n".join(line for line in cleaned.splitlines() if line.strip())
        return cleaned.strip()

    if isinstance(content, str):
        return _strip_think_tags(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return _strip_think_tags("\n".join(part for part in parts if part))
    return ""


def _extract_reasoning_text(message: dict[str, Any]) -> str:
    reasoning = message.get("reasoning")
    if reasoning in (None, ""):
        reasoning = message.get("reasoning_content", "")
    return _normalize_message_content(reasoning)


def _is_context_length_error(exc: BaseException) -> bool:
    message = str(exc).lower().replace("\u2019", "'")
    return (
        "maximum context length" in message
        # SGLang reports an oversized prompt with this wording instead of
        # "maximum context length". Route it through the same drain/retry path.
        or "is longer than the model's context length" in message
        or "context_length_exceeded" in message
        or "reduce the length of the input prompt" in message
        or "parameter=input_tokens" in message
        or '"param":"input_tokens"' in message
    )


@dataclass
class _ChatCompletionResult:
    message: dict[str, Any]
    finish_reason: str = ""
    usage: dict[str, Any] | None = None
    served_by: str = ""


class ToolAgent:
    """Direct tool-calling analyzer compatible with OpenAI-style endpoints."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_ANALYZER_MODEL,
        timeout: Optional[float] = None,
        save_request_logs: bool = False,
        dispatch_index: int = 0,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
    ) -> None:
        resolved_model = _resolve_analyzer_model(model)
        if base_url is not None or provider is not None:
            resolved_model = AnalyzerModelConfig(
                provider=str(provider or resolved_model.provider).strip() or resolved_model.provider,
                base_url=(
                    _host_accessible_base_url(str(base_url).strip())
                    if base_url is not None and str(base_url).strip()
                    else resolved_model.base_url
                ),
                model_id=resolved_model.model_id,
            )
        self._model = resolved_model
        configured_timeout = _LOCAL_ANALYZER_TIMEOUT if timeout is None else timeout
        self._timeout = None if configured_timeout is None or configured_timeout <= 0 else float(configured_timeout)
        self._api_key = str(api_key or "").strip()
        self._tool_steps = None if _LOCAL_ANALYZER_TOOL_STEPS <= 0 else max(1, _LOCAL_ANALYZER_TOOL_STEPS)
        self._python_timeout = min(30, max(1, _LOCAL_ANALYZER_TOOL_TIMEOUT))
        self._yield_seconds = None if _LOCAL_ANALYZER_YIELD_SECONDS <= 0 else float(_LOCAL_ANALYZER_YIELD_SECONDS)
        self._yield_tokens = (
            None if _LOCAL_ANALYZER_YIELD_TOKENS <= 0 else int(_LOCAL_ANALYZER_YIELD_TOKENS)
        )
        self._turn_generated_tokens = 0
        configured_max_output = _LOCAL_ANALYZER_MAX_OUTPUT
        self._max_output_tokens = None if configured_max_output <= 0 else max(1, configured_max_output)
        self._reply_reserve_tokens = self._max_output_tokens or 512
        self._tool_output_tokens = max(64, _LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS)
        self._tool_output_chars = _tool_output_chars()
        self._save_request_logs = bool(save_request_logs)
        self._system_prompt = effective_system_prompt()
        self._request_safety_margin_tokens = _REQUEST_SAFETY_MARGIN_TOKENS
        self._context_budget_tokens = max(
            1024,
            _LOCAL_ANALYZER_CONTEXT_WINDOW - self._reply_reserve_tokens - self._request_safety_margin_tokens,
        )
        self._auto_diff_prev_frame: Frame | None = None
        self._pending_diff_image: dict[str, Any] | None = None
        self._context_was_trimmed = False
        self._has_evicted = False
        self._actions_at_level_start = 0
        self._tokens_at_level_start = 0
        self._progress_pace = ProgressPace()
        self._pace_last_completed_level = 0
        self._goal_defined_level: int | None = None
        # queue position spreads the untrimmed band so games do not tie
        self._diag_name = f"g{int(dispatch_index):02d}"
        self._game_elapsed_seconds: float | None = None
        self._endgame_announced = False
        self._priority_current = _PRIORITY_UNTRIMMED_BASE - max(0, int(dispatch_index))
        self._priority_next = self._priority_current
        self._kept_functions: dict[str, str] = {}
        self._last_animation_chain: list[Any] = []
        self._last_action_gameplay_changed: bool | None = None
        self._pending_animation_image: dict[str, Any] | None = None
        self._pending_animation_image_label: str = ""
        self._pending_diff_image_label: str = ""
        self._turn_diff_image: dict[str, Any] | None = None
        self._turn_diff_image_label: str = ""
        self._pending_gameover_images: list[tuple[str, dict[str, Any]]] = []
        self._http_initial_grace_used = False
        # characters per token, measured from the last response; None until one
        # arrives, and reported in the transcript whenever it changes
        self._text_chars_per_token: float | None = None
        # index into the reasoning-effort ladder; -1 means "server default"
        self._resume_after_yield: bool = False
        self._resume_reason: str = ""
        self._prev_level_start_frame: Frame | None = None
        self._cur_level_start_frame: Frame | None = None
        self._history_messages: list[dict[str, Any]] = []
        self._session_runtime_dir: Path | None = None
        self._session_total_tokens = 0
        self._session_generated_tokens = 0
        self._step_env_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self._current_valid_actions: list[str] = []
        self._last_step_summary: dict[str, Any] | None = None
        self._last_action_call_result: dict[str, Any] | None = None

    def _headers(self) -> dict[str, str]:
        api_key = (
            self._api_key
            or os.environ.get("LOCAL_ANALYZER_API_KEY", "").strip()
            or os.environ.get("OPENROUTER_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
        )
        site_url = os.environ.get("LOCAL_ANALYZER_SITE_URL", "").strip()
        app_name = os.environ.get("LOCAL_ANALYZER_APP_NAME", "ARC3 Agent Harness").strip()
        return build_headers(
            provider=self._model.provider,
            api_key=api_key,
            referer=site_url,
            title=app_name,
        )

    def _ensure_session(self, state_path: Path) -> None:
        runtime_dir = state_path.parent
        if self._session_runtime_dir != runtime_dir:
            self._session_runtime_dir = runtime_dir
            self._priority_level_count_warned = False
            if _persistent_functions():
                self._kept_functions = {}
                self._retained_clear_notice = ""
            self._goal_defined_level = None
            self._goal_nudged = set()
            self._sim_nudged = set()
            self._sim_check_cache = None
            self._history_messages = []
            self._session_total_tokens = 0
            self._session_generated_tokens = 0
            self._last_step_summary = None
            self._last_action_call_result = None
    
    @property
    def total_tokens(self) -> int:
        return max(0, int(self._session_total_tokens))

    @property
    def generated_tokens(self) -> int:
        return max(0, int(self._session_generated_tokens))

    def _accumulate_usage_tokens(
        self, usage: dict[str, Any] | None, *, count_toward_turn: bool = True
    ) -> None:
        """Add a response's usage to the counters.

        count_toward_turn=False for generation that is not part of the turn's
        deliberation - the rolling summary. Its tokens are real GPU time and
        belong in the session and per-game totals, or a summary-heavy run looks
        cheaper than it is and gets more compute than
        max_generated_tokens_per_game intends. But the per-turn yield budget
        measures how long the model has been working on THIS turn, and charging
        it for a summary would shorten whichever turn happened to follow one.
        """
        if not isinstance(usage, dict):
            return
        generated_token_count = 0
        for key in ("completion_tokens", "output_tokens", "generated_tokens"):
            raw_value = usage.get(key)
            try:
                generated_token_count = max(0, int(raw_value))
                break
            except (TypeError, ValueError):
                continue
        self._session_generated_tokens += generated_token_count
        if count_toward_turn:
            self._turn_generated_tokens += generated_token_count

        total_tokens = usage.get("total_tokens")
        try:
            if total_tokens is not None:
                self._session_total_tokens += max(0, int(total_tokens))
                return
        except (TypeError, ValueError):
            pass

        token_count = 0
        for key in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens"):
            raw_value = usage.get(key)
            try:
                token_count += max(0, int(raw_value))
            except (TypeError, ValueError):
                continue
        self._session_total_tokens += token_count

    @staticmethod
    def _turn_action_trace(executed_results: list[dict[str, Any]]) -> list[str]:
        """Stitch per-call action traces into one sequentially numbered
        turn-level trace; synthesize entries for calls without a trace."""
        entries: list[str] = []
        for item in executed_results:
            item_trace = item.get("action_trace")
            if isinstance(item_trace, list) and item_trace:
                entries.extend(
                    re.sub(r"^\d+:\s*", "", str(line)) for line in item_trace
                )
                continue
            names = item.get("executed_actions")
            if not isinstance(names, list) or not names:
                display = str(item.get("action_display") or "").strip()
                names = [display] if display else []
            verdict = observation_verdict(item)
            entries.extend(f"{name} -> {verdict}" for name in names)
        return [f"{position}: {entry}" for position, entry in enumerate(entries, 1)]

    def _summarize_step_sequence(self, action_results: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not action_results:
            return None
        executed_results = [item for item in action_results if item.get("executed")]
        if not executed_results:
            return None

        total_executed = 0
        executed_actions: list[str] = []
        for item in executed_results:
            count = item.get("executed_count")
            try:
                parsed = int(count) if count is not None else 1
            except (TypeError, ValueError):
                parsed = 1
            total_executed += max(1, parsed)
            action_names = item.get("executed_actions")
            if isinstance(action_names, list):
                executed_actions.extend(str(name).strip() for name in action_names if str(name).strip())
            else:
                fallback_action = str(item.get("action_display") or "").strip()
                if fallback_action:
                    executed_actions.append(fallback_action)

        last = executed_results[-1]
        try:
            end_action_num = int(last.get("action_num"))
        except (TypeError, ValueError):
            end_action_num = None
        start_action_num = None
        if end_action_num is not None and total_executed > 0:
            start_action_num = max(1, end_action_num - total_executed + 1)

        return {
            "start_action_num": start_action_num,
            "end_action_num": end_action_num,
            "executed_count": total_executed,
            "executed_actions": executed_actions,
            "level": last.get("level"),
            "level_transition": any(bool(item.get("level_completed")) for item in executed_results),
            "run_complete": any(bool(item.get("run_complete")) for item in executed_results),
            "game_over": any(bool(item.get("game_over")) for item in executed_results),
            "game_over_trace": (
                ToolAgent._turn_action_trace(executed_results)
                if any(item.get("game_over") for item in executed_results)
                else []
            ),
            "fatal_action": next(
                (
                    str(item.get("fatal_action") or item.get("action_display") or "")
                    for item in executed_results
                    if item.get("game_over")
                ),
                "",
            ),
            "view_shifts": [
                shift
                for item in executed_results
                for shift in (item.get("view_shifts")
                              or ([{"action": item.get("action_display"), "shift": item["view_shift"],
                                    "other_cells": None}] if item.get("view_shift") else []))
            ],
            "view_offset": last.get("view_offset") or [0, 0],
            "view_offset_uncertain": last.get("view_offset_uncertain"),
            "board_changed": any(bool(item.get("board_changed")) for item in executed_results),
            # the interior-only flag, so the opener can say WHICH kind of change
            # happened instead of hedging over the broader one
            "gameplay_changed": any(
                bool(item.get("gameplay_changed")) for item in executed_results
            ),
            "stop_reason": last.get("stop_reason"),
            "observation_stops": [
                {
                    "action": (item.get("executed_actions") or [None])[-1],
                    "detail": item.get("stop_detail"),
                    "index": item.get("observation_action_index"),
                    "skipped": item.get("skipped_actions") or [],
                    "trace": item.get("action_trace") or [],
                }
                for item in executed_results
                if item.get("stop_reason") in {"no_visible_change", "border_only_change", "animation_only", "view_shift",
                                               "sim_mismatch", "sim_guard_error"}
            ],
            # the animation of the last executed action, which is what
            # last_animation_frames / last_animation_timeline will return when asked
            "animation": (executed_results[-1].get("animation") if executed_results else None),
            "animation_action": (
                executed_results[-1].get("action_display") if executed_results else None
            ),
            "guard_stops": [
                {
                    "kind": item.get("stop_reason"),
                    "action": item.get("blocked_action"),
                    "detail": item.get("stop_detail"),
                    "skipped": item.get("skipped_actions") or [],
                    "trace": item.get("action_trace") or [],
                }
                # scans ALL results, not just executed ones: a refusal has
                # executed=False by construction and is filtered out of
                # executed_results before this point
                for item in action_results
                if item.get("stop_reason") in ("known_noop", "known_death", "stale_state")
            ],
        }

    # _describe_last_outcome removed. It built an outcome sentence and was
    # called from nowhere - in this harness AND upstream - so neither its
    # wording nor the fallback it carried ever reached a transcript. Two
    # patches were spent revising text no model ever read. What it uniquely
    # provided is now rendered inline in _build_user_prompt.



    def _prepare_gameover_images(
        self,
        history_entries: list["HistoryEntry"],
        previous_step_summary: dict[str, Any] | None = None,
    ) -> None:
        """Queue the death-turn visuals (ARC3_GAMEOVER_DIFF_IMAGE).

        The auto-diff - and with it the diff image - is suppressed after a game
        over because the auto-RESET makes that span meaningless, so the visual
        slot is idle on exactly the turn where the death is invisible: the
        board the model sees has already been reset. Two parts fill it:

          1. the death frame itself (the budget bar in it is what the BAR RULE
             asks about, and it exists nowhere else in the message), and
          2. a change-highlight diff of the killing action (last alive frame ->
             death frame), the spatial form of the textual 'Fatal step diff'.

        Anchoring matches _game_over_diff_lines: history ends with the RESET
        marker, entries[-2] is the death frame, entries[-3] the last alive one.
        """
        self._pending_gameover_images = []
        if not _get_env_bool("ARC3_GAMEOVER_DIFF_IMAGE", False):
            return
        from inference.agent.vision_context import (
            current_grid_image_part,
            diff_image_part,
        )

        entries = list(history_entries or [])
        if len(entries) < 3 or str(entries[-1].action or "").upper() != "RESET":
            return
        death = entries[-2]
        last_alive = entries[-3]
        if death.frame is None or str(death.action or "").upper() == "RESET":
            return
        death_part = current_grid_image_part(death.frame)
        if death_part is not None:
            self._pending_gameover_images.append(
                (
                    "Death frame image (the board when the attempt ended, before the "
                    "automatic reset); check the budget bar here:",
                    death_part,
                )
            )
        self._append_fatal_animation_image(previous_step_summary)
        if last_alive.frame is None:
            return
        moment = getattr(self, "_death_moment", None)
        if moment is not None:
            # r14: under a scroll the last-alive -> death diff is the camera move; show the frame in which the
            # victim was last present instead, in the coordinates of the death-context lines
            label, grid = moment
            try:
                part = current_grid_image_part(Frame(grid=tuple(tuple(r) for r in grid), step=death.frame.step,
                                                     level=death.frame.level))
            except Exception as exc:  # diagnostics only
                log.warning("death moment image failed: %s", exc)
                part = None
            if part is not None:
                self._pending_gameover_images.append((label, part))
            return
        fatal_part = diff_image_part(last_alive.frame, death.frame)
        if fatal_part is not None and last_alive.frame.grid != death.frame.grid:
            self._pending_gameover_images.append(
                (
                    "Fatal step diff image (last alive frame -> death frame): cells "
                    "changed by the fatal action, in their new colour; unchanged cells dimmed:",
                    fatal_part,
                )
            )


    def _append_fatal_animation_image(self, summary: dict[str, Any] | None) -> None:
        """An image of what the fatal action's animation hid, on a game over.

        Gated on transient pixels alone, NOT on gameplay_changed: after a death
        the board has already reset, so gameplay_changed compares the
        pre-action board against the restarted level and is essentially always
        true. The justification differs from the ordinary case too - there the
        board shows nothing, here it shows a different board entirely, and
        either way the fatal moment is unreachable.
        """
        if not _animation_enabled():
            return
        mode = animation_death_image_mode()
        if mode == "off" or not summary:
            return
        animation = summary.get("animation") or {}
        if not int(animation.get("transient_pixels") or 0):
            return
        chain = self._last_animation_chain or []
        cells = _animation_transient_cells(chain)
        if not cells:
            return
        if mode == "composite":
            part = animation_composite_part(
                chain, cells, border=_get_env_int("ARC3_NOOP_GUARD_BORDER", 4)
            )
            label = (
                "Fatal action animation image: the cells that changed during the fatal "
                "action's animation, each in the last colour it held before reverting; "
                "everything else, including the border, is dimmed to dark navy (an "
                "off-palette colour). This is a composite of all cells that changed "
                "during the animation, not a single board state that the game "
                "displayed. A cell is drawn nearer its true colour the longer it held "
                "it, so a bright trail marks where something stayed and a uniformly "
                "dim one marks something that passed through. Neither the death frame "
                "nor the last alive frame contains these cells - they existed only "
                "between the two."
            )
        else:
            part, index = animation_peak_part(chain, cells)
            label = (
                f"Fatal action animation image (frame {index} of {len(chain) - 1}): the "
                "board partway through the fatal action's animation, at the point where "
                "the most cells were showing something they later reverted. This is a "
                "real state the game displayed, between the last alive frame and the "
                "death frame."
            )
        if part is not None:
            self._pending_gameover_images.append((label, part))

    def _prepare_animation_image(self, summary: dict[str, Any] | None) -> None:
        """Attach an image of what the animation hid, when the board did not move.

        Gated on gameplay_changed being false: that is exactly when the board
        image the model receives is uninformative, because the action did
        something and no frame it can reach shows what. On any other turn the
        current board already carries the outcome.
        """
        if not _animation_enabled():
            # ARC3_ANIMATION_IMAGE needs ARC3_ANIMATION: without it no chain is
            # captured, so this would silently do nothing. Stated rather than
            # left implicit - a knob that quietly has no effect is the failure
            # mode that costs the most time to diagnose.
            return
        mode = animation_image_mode()
        if mode == "off" or not summary:
            return
        if self._last_action_gameplay_changed is not False:
            return
        animation = summary.get("animation") or {}
        if not int(animation.get("transient_pixels") or 0):
            return
        chain = self._last_animation_chain or []
        cells = _animation_transient_cells(chain)
        if not cells:
            return
        if mode == "composite":
            part = animation_composite_part(
                chain, cells, border=_get_env_int("ARC3_NOOP_GUARD_BORDER", 4)
            )
            label = (
                "Animation image: the cells that changed during the animation, each in "
                "the last colour it held before reverting; everything else, including "
                "the border, is dimmed to dark navy (an off-palette colour). This is a "
                "composite of all cells that changed during the animation, not a single "
                "board state that the game displayed. A cell is drawn nearer its true "
                "colour the longer it held it, so a bright trail marks where something "
                "stayed and a uniformly dim one marks something that passed through. "
                "The action left the board area unchanged, so none of these cells "
                "appear in `current_frame`."
            )
        else:
            part, index = animation_peak_part(chain, cells)
            label = (
                f"Animation image (frame {index} of {len(chain) - 1}): the board partway "
                "through the animation, at the point where the most cells were showing "
                "something they later reverted. This is a real state the game displayed, "
                "but not one that survives into `current_frame`."
            )
        if part is not None:
            self._pending_animation_image = part
            self._pending_animation_image_label = label

    def _retained_function_context(self) -> str:
        """Optional prompt metadata must never interrupt game execution."""
        if not _persistent_functions():
            return ""
        try:
            notice = getattr(self, "_retained_clear_notice", "")
            signatures = [function_signature(src) for src in self._kept_functions.values()]
            parts = [notice] if notice else []
            if signatures:
                scope = {"levelup": "on this level", "turn": "in this turn", "game": "in this game"}[_persistent_functions_scope()]
                parts.append(
                    "Your retained functions " + scope + ": " + ", ".join(signatures)
                    + " (your earlier definitions; revise any that new evidence contradicts)."
                )
            self._retained_clear_notice = ""
            return "\n\n" + "\n".join(parts) if parts else ""
        except Exception:
            self._kept_functions = {}
            self._retained_clear_notice = ""
            return "\n\nYour retained functions were cleared after a retention error; redefine any you need."

    def _retained_sources(self):
        if not _persistent_functions():
            return None
        try:
            return list(self._kept_functions.values())
        except Exception:
            self._kept_functions = {}
            return []

    def _record_retained_functions(self, sandbox_result, payload) -> None:
        if not _persistent_functions():
            return
        if "keepable_functions" not in sandbox_result:
            # The snippet did not finish (timeout or sandbox failure), so nothing it defined can be read
            # back. Keep the functions retained before it: replacing them with an empty set wiped all of
            # them, goal_violations included, after a single timeout (r11l, run r7 B: 13 functions).
            if self._kept_functions:
                payload["function_retention"] = ("This snippet did not finish, so nothing it defined was retained; "
                                                 "your previously retained functions are unchanged.")
            return
        try:
            previous = self._kept_functions
            kept = {}
            signatures = {}
            for item in sandbox_result.get("keepable_functions") or []:
                name, source = item["name"], item["source"]
                signature = function_signature(source)
                if signature.split("(", 1)[0] != name:
                    raise ValueError("invalid retained function metadata")
                kept[name] = source
                signatures[name] = signature
            notes = []
            scope = {"levelup": "on this level", "turn": "in this turn", "game": "in this game"}[_persistent_functions_scope()]
            for name, source in kept.items():
                if previous.get(name) != source:
                    notes.append("Retained your function " + signatures[name] + " for later Python calls " + scope + ".")
            rejected = sandbox_result.get("retention_rejected") or []
            repair_hints = _get_env_bool("ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS", False)
            rejected_names = set()
            seen_rejections = set()
            for item in rejected:
                name, reason = str(item["name"]), str(item["reason"])
                hint = (str(item.get("hint") or "").strip() if repair_hints else
                        "Pass snippet-specific data as arguments or redefine it if needed.")
                rejection_key = (name, reason, hint)
                if rejection_key in seen_rejections:
                    continue
                seen_rejections.add(rejection_key)
                rejected_names.add(name)
                if name == "all":
                    notes.append("Your retained functions were cleared: " + reason + ".")
                    continue
                notes.append("Your function " + name + " was not retained: " + reason + "."
                             + (" " + hint if hint else ""))
            for name in previous.keys() - kept.keys() - rejected_names:
                notes.append("Your previous function " + name + " is no longer retained; redefine it if needed.")
            self._kept_functions = kept
            if notes:
                payload["function_retention"] = "\n".join(notes)
        except Exception:
            self._kept_functions = {}
            payload["function_retention"] = "Your retained functions were cleared after a retention error; redefine any you need."

    def _build_user_message(self, user_prompt: str, current_frame: Frame | None) -> dict[str, Any]:
        user_prompt += self._retained_function_context()
        image_part = current_grid_image_part(current_frame)
        diff_part = self._pending_diff_image
        diff_label = self._pending_diff_image_label
        animation_part = self._pending_animation_image
        animation_label = self._pending_animation_image_label
        gameover_images = list(self._pending_gameover_images)
        self._pending_diff_image = None
        self._pending_diff_image_label = ""
        self._pending_animation_image = None
        self._pending_animation_image_label = ""
        self._pending_gameover_images = []
        if animation_part is not None:
            # the animation image only ever fires when nothing inside the board
            # moved, which is precisely when the diff image shows the ticking
            # budget bar and nothing else - two images, one of them noise
            diff_part = None
        if image_part is None:
            return {"role": "user", "content": user_prompt}

        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"{user_prompt}\n\nCurrent grid image (the board AFTER the automatic reset - this is what you act on now):" if gameover_images else f"{user_prompt}\n\nCurrent grid image:"},
            image_part,
        ]
        if diff_part is not None:
            content.append({"type": "text", "text": diff_label})
            content.append(diff_part)
        if animation_part is not None:
            content.append({"type": "text", "text": animation_label})
            content.append(animation_part)
        for label, part in gameover_images:
            content.append({"type": "text", "text": label})
            content.append(part)
        return {"role": "user", "content": content}



    @staticmethod
    def _manual_reset_context_line(summary: dict[str, Any]) -> str | None:
        if not reset_exposed() or any(summary.get(flag) for flag in (
            "stale", "game_over", "run_complete", "done", "level_transition", "level_completed",
        )):
            return None
        actions = [str(name).strip().upper() for name in summary.get("executed_actions") or []]
        if "RESET" not in actions:
            return None
        if actions == ["RESET"]:
            return (
                "You deliberately reset the current level. The current frame shows "
                "the state returned by RESET."
            )
        return (
            "You deliberately reset the current level, then executed the subsequent "
            "actions listed above. The current frame shows their resulting state."
        )


    def _build_user_prompt(
        self,
        action_num: int,
        *,
        valid_actions: list[str] | None,
        current_frame: Frame | None = None,
        history_entries: list[HistoryEntry] | None = None,
        previous_step_summary: dict[str, Any] | None = None,
    ) -> str:
        """The turn opener: only what changed since the previous turn.

        Static rules live in the system prompt (inference.agent.prompts); nothing
        here restates them. Order: outcome of the last executed call(s), level or
        game-over context, current state, time, closing line.
        """
        history_entries = history_entries or []
        self._opener_history_entries = history_entries
        current_level = current_frame.level if current_frame is not None else 1
        current_step = current_frame.step if current_frame is not None else max(0, action_num)
        summary = previous_step_summary
        if summary is not None:
            try:
                current_level = max(current_level, int(summary.get("level")))
            except (TypeError, ValueError):
                pass
        self._note_level(current_level, current_step)
        lines: list[str] = []
        if summary and summary.get("stale"):
            lines.extend(_stale_summary_lines(summary))
        elif summary:
            lines.extend(self._executed_sequence_lines(summary))
            if summary.get("run_complete"):
                lines.append("You have completed the game.")
            elif summary.get("level_transition"):
                lines.append(LEVEL_START_USER_PROMPT)
            elif not summary.get("game_over"):
                lines.append(
                    self._manual_reset_context_line(summary) or "You are still on the same level."
                )
            if summary.get("game_over"):
                lines.extend(self._game_over_lines(history_entries, summary))
        elif current_step > 0 or action_num > 0:
            lines.append("No previous action sequence was captured.")
        else:
            lines.append("First turn of this game: no action has been executed yet.")
        # also prepares the diff image attached by _build_user_message
        self._prepare_auto_diff(current_frame, summary)
        exploration_lines = self._exploration_lines(history_entries, current_frame, current_level, valid_actions)
        lines.append(
            f"Current state: level {current_level}, {current_step} action(s) executed in this game. "
            f"Valid actions: {_format_valid_action_line(valid_actions)}."
        )
        time_line = self._time_line(current_step)
        if time_line:
            lines.append(time_line)
        lines.extend(self._goal_check_lines(history_entries, current_frame, current_level, summary))
        lines.extend(self._sim_check_lines(history_entries, current_frame))
        lines.extend(exploration_lines)
        lines.extend(self._ledger_lines(history_entries, current_level))
        lines.append("Inspect the evidence with `python`, then act.")
        return "\n".join(lines)

    def _prepare_auto_diff(
        self,
        current_frame: Frame | None,
        previous_step_summary: dict[str, Any] | None,
    ) -> None:
        """Prepare the diff image between the board shown at the previous turn and
        the current one. None on the first turn, after a level change (a
        cross-level diff is meaningless) and after a game over (the automatic
        reset makes the span meaningless)."""
        from inference.agent.vision_context import diff_image_part

        self._pending_diff_image = None
        self._pending_diff_image_label = ""
        self._turn_diff_image = None
        self._turn_diff_image_label = ""
        if current_frame is None:
            return
        previous_frame = self._auto_diff_prev_frame
        self._auto_diff_prev_frame = current_frame
        if previous_frame is None or not _DIFF_IMAGE:
            return
        if getattr(previous_frame, "level", None) != getattr(current_frame, "level", None):
            return
        if previous_step_summary and previous_step_summary.get("game_over"):
            return
        part = diff_image_part(previous_frame, current_frame)
        if part is None or previous_frame.grid == current_frame.grid:
            return
        label = (
            f"Diff image (actions {getattr(previous_frame, 'step', '?')} -> {getattr(current_frame, 'step', '?')}): "
            "cells changed since your previous turn, in their new colour; unchanged cells dimmed to dark navy."
        )
        self._pending_diff_image = part
        self._pending_diff_image_label = label
        self._turn_diff_image = part
        self._turn_diff_image_label = label

    def _stall_note(self, turn_started_at: float, step_executed: bool) -> str:
        """Anti-stall reminder once a turn has deliberated past the threshold
        without executing an action; repeated at most once per threshold."""
        threshold = _stall_seconds()
        if threshold <= 0 or step_executed:
            return ""
        now = time.monotonic()
        elapsed = now - turn_started_at
        last = getattr(self, "_stall_noted_at", None)
        if elapsed < threshold or (last is not None and last >= turn_started_at and now - last < threshold):
            return ""
        self._stall_noted_at = now
        remaining = ""
        total = getattr(self, "_time_remaining_seconds", None)
        reference = getattr(self, "_time_remaining_reference", None)
        if isinstance(total, (int, float)) and reference is not None:
            left = max(0.0, float(total) - (now - reference))
            remaining = f"; about {_format_duration(left)} left for this game"
        undo_hint = " (UNDO can revert most probes)" if "UNDO" in (self._current_valid_actions or []) else ""
        return STALL_NOTE.format(elapsed=int(elapsed), remaining=remaining, undo_hint=undo_hint)

    @staticmethod
    def _view_shift_lines(summary: dict[str, Any]) -> list[str]:
        """One line per action that scrolled the view (at most 3), with the level total."""
        from inference.utils.view_shift import describe

        shifts = [item for item in summary.get("view_shifts") or [] if item.get("shift")]
        lines = []
        for item in shifts[:3]:
            dy, dx = item["shift"]
            other = item.get("other_cells")
            lines.append(
                f"The view scrolled after `{item.get('action')}`: everything on screen moved {describe(dy, dx)} "
                "(camera movement)." + (f" Apart from that, {other} cells changed." if other is not None else "")
            )
        if len(shifts) > 3:
            lines.append(f"({len(shifts) - 3} further scroll(s) in that sequence.)")
        if shifts:
            total = summary.get("view_offset") or [0, 0]
            line = ("Total view shift on this level: "
                    + (describe(total[0], total[1]) if any(total) else "none (back to the start view)"))
            uncertain = summary.get("view_offset_uncertain")
            if uncertain:
                line += (f" (uncertain: action {uncertain} changed a quarter of the screen or more without a detected"
                         " scroll, so a scroll may have been missed)")
            lines.append(line + ".")
        return lines

    def _executed_sequence_lines(self, summary: dict[str, Any]) -> list[str]:
        """What the previous call(s) executed and what they changed."""
        lines: list[str] = []
        try:
            count = int(summary.get("executed_count") or 0)
        except (TypeError, ValueError):
            count = 0
        executed = [str(name).strip() for name in summary.get("executed_actions") or [] if str(name).strip()]
        shown = ", ".join(executed[:10]) + (f", ... ({len(executed) - 10} more)" if len(executed) > 10 else "")
        lines.append(
            f"Executed {count} action{'' if count == 1 else 's'} since your previous turn: {shown or 'none'}."
        )
        if count and not summary.get("game_over"):
            if summary.get("gameplay_changed"):
                lines.append("Result: cells inside the board interior changed.")
            elif summary.get("board_changed"):
                lines.append(
                    f"Result: only cells within {_get_env_int('ARC3_NOOP_GUARD_BORDER', 4)} of the edge changed; "
                    "the final interior is unchanged (meaning unknown)."
                )
            else:
                lines.append("Result: no cell changed in the final board.")
        if not summary.get("game_over") and not summary.get("level_transition"):
            lines.extend(self._view_shift_lines(summary))
        if count and not summary.get("game_over"):
            lines.extend(self._motion_summary_lines(summary))
        if _animation_enabled() and not summary.get("game_over"):
            animation_line = describe_animation(
                summary.get("animation"),
                action_display=summary.get("animation_action"),
                gameplay_changed=self._last_action_gameplay_changed,
                timeline=_animation_timeline_enabled(),
            )
            if animation_line:
                lines.append(animation_line)
                lines.extend(self._animation_track_lines())
            self._prepare_animation_image(summary)
        for stop in (summary.get("observation_stops") or [])[:3]:
            lines.append(str(stop.get("detail") or "Execution paused for inspection."))
        guard_stops = summary.get("guard_stops") or []
        for stop in guard_stops[:3]:
            detail = str(stop.get("detail") or "").strip()
            lines.append(
                f"Not executed: {stop.get('action')!r}" + (f" - {detail}" if detail else ".")
            )
        if len(guard_stops) > 3:
            lines.append(f"({len(guard_stops) - 3} further refusal(s) in that sequence.)")
        return lines

    def _motion_summary_lines(self, summary: dict[str, Any]) -> list[str]:
        """r10 lever 1: object-level motion for the actions since the previous turn.

        One action: what moved, appeared, changed colour (the ar25 "+4 read as +3" case). A batch
        of 2-12: one line per action. Boards come from the per-action history kept for the opener; a
        level transition drops the completing action (its after-board is the next level).
        """
        if not _motion_summary_enabled():
            return []
        try:
            from inference.utils import motion

            entries = list(getattr(self, "_opener_history_entries", None) or [])
            count = int(summary.get("executed_count") or 0)
            if count <= 0 or len(entries) < count + 1:
                return []
            tail = entries[-count:]
            if any((getattr(e, "result", None) or {}).get("automatic") for e in tail):
                return []
            if summary.get("level_transition"):
                tail = tail[:-1]
                if not tail:
                    return []
            start = len(entries) - count - 1
            grids = [entries[start].frame.grid] + [e.frame.grid for e in tail]
            actions = [str(getattr(e, "action", "") or "") for e in tail]
            border = _get_env_int("ARC3_NOOP_GUARD_BORDER", 4)
            lines: list[str] = []
            if len(actions) == 1:
                phrases = motion.motion_lines(grids[0], grids[1], border=border, max_items=4)
                if phrases:
                    lines.append("Motion (objects matched by colour and shape): " + "; ".join(phrases) + ".")
                return lines
            if len(actions) > 12:
                return []
            batch = motion.batch_motion(grids, actions, border=border, max_actions=12)
            per_action = batch["per_action"]
            if per_action and any(p[2] for p in per_action):
                parts = []
                for index, action, phrases in per_action:
                    text = "; ".join(phrases[:2]) if phrases else "no object changed"
                    parts.append(f"{index} {action}: {text}")
                lines.append("Per action (objects matched by colour and shape): " + " | ".join(parts) + ".")
            return lines[:3]
        except Exception as exc:  # diagnostics only; never block a turn
            log.warning("motion summary failed: %s", exc)
            return []

    def _animation_track_lines(self) -> list[str]:
        """r10 lever 1: where the moving objects of the last animation went, what flashed."""
        if not _motion_summary_enabled():
            return []
        try:
            from inference.utils import motion

            chain = self._last_animation_chain or []
            if len(chain) < 3:
                return []
            tracks = motion.animation_tracks(chain, border=_get_env_int("ARC3_NOOP_GUARD_BORDER", 4), max_tracks=4)
            if not tracks:
                return []
            return ["Animation tracks (objects matched by colour and shape, step = engine frame): "
                    + "; ".join(tracks) + "."]
        except Exception as exc:  # diagnostics only
            log.warning("animation tracks failed: %s", exc)
            return []

    def _death_context(self, history_entries: list[HistoryEntry]) -> dict[str, Any] | None:
        """r14: what the fatal action's animation shows (inference.agent.death_context); None when off or n/a."""
        from inference.agent import death_context

        if not death_context.enabled():
            return None
        try:
            entries = list(history_entries or [])
            if len(entries) < 3 or str(entries[-1].action or "").upper() != "RESET":
                return None
            death, alive = entries[-2], entries[-3]
            if death.frame is None or alive.frame is None or not (getattr(death, "result", None) or {}).get("game_over"):
                return None
            death_grid = [list(r) for r in death.frame.grid]
            chain = [[list(r) for r in g] for g in (self._last_animation_chain or [])]
            if len(chain) < 2 or chain[-1] != death_grid:
                chain = [[list(r) for r in alive.frame.grid], death_grid]
            level = int(getattr(death.frame, "level", 0) or 0)
            pairs = death_context.history_pairs(entries, level)
            prefer = death_context.recent_movers(pairs)
            alive_result = getattr(alive, "result", None) or {}
            offset = None
            if not alive_result.get("view_offset_uncertain"):
                raw = alive_result.get("view_offset") or [0, 0]
                offset = (int(raw[0]), int(raw[1]))
            return death_context.report(chain, border=_get_env_int("ARC3_NOOP_GUARD_BORDER", 4), prefer_colors=prefer,
                                        level_offset=offset)
        except Exception as exc:  # diagnostics only; never block a turn
            log.warning("death context failed: %s", exc)
            return None

    def _game_over_lines(
        self, history_entries: list[HistoryEntry], summary: dict[str, Any]
    ) -> list[str]:
        lines = [GAME_OVER_USER_PROMPT]
        context = self._death_context(history_entries)
        self._death_moment = None
        if context:
            lines.extend(context.get("lines") or [])
            if context.get("scrolled") and context.get("moment_frame") is not None:
                self._death_moment = (context.get("moment_label") or "", context["moment_frame"])
        if _animation_enabled() and not (context and context.get("scrolled")):
            # under a scroll the transient-cell count is the camera move; the scroll note replaces it
            death_animation_line = describe_animation(
                summary.get("animation"),
                action_display=summary.get("animation_action"),
                gameplay_changed=self._last_action_gameplay_changed,
                game_over=True,
                timeline=_animation_timeline_enabled(),
            )
            if death_animation_line:
                lines.append(death_animation_line)
        fatal_action = str(summary.get("fatal_action") or "")
        if fatal_action:
            lines.append(f"The game over happened immediately after {fatal_action!r}.")
        trace = summary.get("game_over_trace") or []
        if trace:
            lines.append("Per-action trace of that sequence: " + "; ".join(str(t) for t in trace) + ".")
        self._prepare_gameover_images(history_entries, summary)
        lines.append(GAME_OVER_REPLAY_WARNING)
        return lines

    def _note_level_transition(self, summary: dict[str, Any] | None) -> None:
        """Per-level bookkeeping at a level change, restored from the base harness.

        It lived in the world-model update method that the 2026-10-12 prune removed. Without
        it the priority snapshot counted actions and tokens since the start of the game rather
        than the level, and the game-scope notice below was no longer shown. The base pace
        bookkeeping is not restored (ARC3_PRIORITY_PACE is off and its helper was pruned).
        """
        if not summary or not summary.get("level_transition"):
            return
        self._actions_at_level_start = _priority_action_count(summary)
        self._tokens_at_level_start = self._session_generated_tokens
        if _persistent_functions_scope() == "levelup":
            # a helper can encode the old layout in its code even when it holds
            # no stale data, so the level boundary is where it stops being safe
            self._kept_functions = {}
            if _persistent_functions():
                self._retained_clear_notice = (
                    "Your retained functions were cleared at this level transition. "
                    "Redefine any helpers you still need after checking their assumptions "
                    "against the new board."
                )
        elif (
            _persistent_functions_scope() == "game"
            and _persistent_functions()
            and getattr(self, "_kept_functions", None)
        ):
            self._retained_clear_notice = (
                "Your retained functions remain available from previous levels. "
                "Check level-specific assumptions against the new board before reusing them."
            )

    def _goal_check_lines(self, history_entries, current_frame, level, summary) -> list[str]:
        """Evaluate the model's retained goal_violations(frame), or nudge it to write one."""
        from inference.agent import goal_check

        if not goal_check.enabled() or current_frame is None or not _persistent_functions():
            return []
        try:
            history_entries = list(history_entries or [])
            kept_pool = getattr(self, "_kept_functions", None) or {}
            defined = goal_check.NAME in kept_pool
            defined_level = getattr(self, "_goal_defined_level", None)
            # Checked only on the level where it was last defined; it stays retained.
            stale_from = defined_level if defined and defined_level is not None and defined_level != level else None
            current = None
            if defined and stale_from is None:
                state = {"current_frame": _ascii_frame_view_payload(current_frame),
                         "history": _ascii_history_view_payload(history_entries),
                         "valid_actions": [], "last_action_call_result": {}}
                current = goal_check.evaluate(
                    run_sandboxed_python, state, self._retained_sources() or [],
                    retain_imports=_get_env_bool("ARC3_PERSISTENT_FUNCTIONS_IMPORTS", False))
            if not hasattr(self, "_goal_nudged"):
                self._goal_nudged = set()
            return goal_check.lines(defined=defined, current=current, stale_from=stale_from,
                                    level_transition=bool(summary and summary.get("level_transition")),
                                    actions_on_level=goal_check.level_actions(history_entries, level),
                                    level=level, nudged=self._goal_nudged)
        except Exception as exc:  # diagnostics only; never block a turn
            log.warning("goal check failed: %s", exc)
            return []

    def _sim_check_lines(self, history_entries, current_frame) -> list[str]:
        """r15: replay the model's retained observe(frame) + step(state, action) over this level's recorded
        transitions (sim_check / sim_contract), or say once per level that the pair is missing."""
        from inference.agent import goal_check, sim_check

        if not sim_check.enabled() or current_frame is None or not _persistent_functions():
            return []
        try:
            kept_pool = getattr(self, "_kept_functions", None) or {}
            entries = list(history_entries or [])
            level = int(getattr(current_frame, "level", 0) or 0)
            missing = [name for name in sim_check.NAMES if name not in kept_pool]
            if missing:
                if not hasattr(self, "_sim_nudged"):
                    self._sim_nudged = set()
                return sim_check.missing_lines(missing, actions_on_level=goal_check.level_actions(entries, level),
                                               level=level, nudged=self._sim_nudged)
            # every retained source, not only observe/step: a changed helper they call changes the result
            key = (level, len(entries), tuple(sorted(kept_pool.items())))
            cached = getattr(self, "_sim_check_cache", None)
            if cached and cached[0] == key:
                return list(cached[1])
            state = {"current_frame": _ascii_frame_view_payload(current_frame),
                     "history": _ascii_history_view_payload(entries),
                     "valid_actions": [], "last_action_call_result": {}}
            result = sim_check.evaluate(
                run_sandboxed_python, state, self._retained_sources() or [],
                retain_imports=_get_env_bool("ARC3_PERSISTENT_FUNCTIONS_IMPORTS", False),
                border=_get_env_int("ARC3_NOOP_GUARD_BORDER", 4))
            out = sim_check.lines(result)
            self._sim_check_cache = (key, out)
            return out
        except Exception as exc:  # diagnostics only; never block a turn
            log.warning("sim check failed: %s", exc)
            return []

    def _ledger_lines(self, history_entries, level) -> list[str]:
        """r10 lever 5: the level's facts from history, on the first turn after a context trim (or always)."""
        from inference.agent import ledger

        try:
            how = ledger.mode()
            if how == "off":
                return []
            dropped = int(getattr(self, "_drain_pending", 0) or 0)
            if how == "drain" and dropped <= 0:
                return []
            self._drain_pending = 0
            lines = ledger.ledger_lines(list(history_entries or []), level,
                                        border=_get_env_int("ARC3_NOOP_GUARD_BORDER", 4))
            notes = bool(getattr(self, "_compaction_notes", False))
            self._compaction_notes = False
            notice = [ledger.drain_notice(dropped, notes=notes)] if dropped > 0 else []
            return notice + lines
        except Exception as exc:  # diagnostics only; never block a turn
            log.warning("level ledger failed: %s", exc)
            return []

    def _exploration_lines(self, history_entries, current_frame, level, valid_actions) -> list[str]:
        """Untested actions, state-specific no-ops and unclicked objects on this level."""
        from inference.agent import exploration

        if not exploration.enabled() or current_frame is None:
            return []
        try:
            return exploration.coverage_lines(
                history_entries, current_frame.grid, level, _normalize_valid_actions(valid_actions),
                border=_get_env_int("ARC3_NOOP_GUARD_BORDER", 4),
            )
        except Exception as exc:  # diagnostics only; never block a turn
            log.warning("exploration note failed: %s", exc)
            return []

    def _note_level(self, level: int, step: int) -> None:
        """Track when the current level started, for the time line."""
        if getattr(self, "_level_seen", None) != level:
            self._level_seen = level
            self._level_started_at = time.monotonic()
            self._level_start_step = step

    def _time_line(self, current_step: int) -> str:
        if not _time_cues_enabled():
            return ""
        parts: list[str] = []
        remaining = getattr(self, "_time_remaining_seconds", None)
        if isinstance(remaining, (int, float)) and remaining >= 0:
            parts.append(f"about {_format_duration(remaining)} left for this game")
        started = getattr(self, "_level_started_at", None)
        if started is not None:
            spent = max(0.0, time.monotonic() - started)
            actions = max(0, int(current_step) - int(getattr(self, "_level_start_step", 0) or 0))
            parts.append(f"{_format_duration(spent)} and {actions} action(s) spent on this level so far")
        line = ("Time: " + "; ".join(parts) + ".") if parts else ""
        if line and isinstance(remaining, (int, float)) and 0 <= remaining < _endgame_seconds():
            # r10: with 3 minutes left bp35 A wrote "Time is up" and spent 12 turns on summaries and no-op
            # calls; the cue is read as "stop". Say what the remaining time is for - at most every 120 s
            # (r10 showed it on every turn of the last five minutes, 80-90 times per run).
            now = time.monotonic()
            last = getattr(self, "_endgame_noted_at", None)
            if last is None or now - last >= 120:
                self._endgame_noted_at = now
                line += (" The harness stops the game by itself; do not write summaries or final answers. Spend the "
                         "remaining time on the cheapest action that could still complete the level.")
        return line

    def _tools(self, state_path: Path) -> list[dict[str, Any]]:
        self._ensure_session(state_path)
        return [
            {
                "type": "function",
                "function": {
                    "name": "python",
                    "description": PYTHON_TOOL_DESCRIPTION,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": PYTHON_CODE_PARAMETER_DESCRIPTION,
                            },
                        },
                        "required": ["code"],
                    },
                },
            }
        ]

    def _harness_template_kwargs(self) -> dict[str, Any]:
        """Chat-template kwargs the HARNESS sets per request.

        Shared by the request builder and the request log so the log records
        what was actually sent rather than a reconstruction - which matters
        for the effort ladder, where the value changes mid-run, and for
        preserve_thinking, where the question is whether it reaches the server
        at all."""
        kwargs: dict[str, Any] = {}
        preserve_thinking = _preserve_thinking_kwarg()
        if preserve_thinking is not None:
            kwargs["preserve_thinking"] = preserve_thinking
        effort = os.environ.get("ARC3_REASONING_EFFORT", "").strip()
        if effort:
            # chat-template reasoning_effort (Qwen3.8: low/medium/high/xhigh); unset = template default
            kwargs["reasoning_effort"] = effort
        return kwargs

    def _chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        request_timeout_seconds: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> _ChatCompletionResult:
        payload = build_chat_payload(
            provider=self._model.provider,
            model=self._model.model_id,
            messages=messages,
            max_tokens=self._max_output_tokens if max_tokens is None else max_tokens,
            temperature=_LOCAL_ANALYZER_TEMPERATURE,
            top_p=_LOCAL_ANALYZER_TOP_P,
            top_k=_LOCAL_ANALYZER_TOP_K,
            thinking=(
                bool(_LOCAL_ANALYZER_ENABLE_THINKING)
                if enable_thinking is None
                else enable_thinking
            ),
            tools=tools,
            tool_choice=_request_tool_choice(tools),
            seed=_LOCAL_ANALYZER_SEED,
        )
        provider_prefs = _openrouter_provider_prefs(self._model.provider)
        if provider_prefs:
            payload["provider"] = provider_prefs
        # merge rather than assign: the payload builder may already have set
        # chat_template_kwargs (enable_thinking), and replacing the dict would
        # drop it - as a request-level dict can also drop a server-side default
        template_kwargs = dict(payload.get("chat_template_kwargs") or {})
        template_kwargs.update(self._harness_template_kwargs())
        if enable_thinking is not None:
            # after the merge: _harness_template_kwargs does not set
            # enable_thinking today, but an override that a later edit could
            # silently undo is the bug that keeps recurring here
            template_kwargs["enable_thinking"] = bool(enable_thinking)
        if template_kwargs:
            payload["chat_template_kwargs"] = template_kwargs
        payload["messages"] = _strip_control_keys(payload.get("messages") or [])
        def post_chat(request_payload: dict[str, Any]) -> requests.Response:
            stopped, remaining = clock_status(self, request_timeout_seconds or self._timeout)
            if stopped or remaining <= 0:
                raise requests.exceptions.ReadTimeout("Game deadline/stop check")
            return requests.post(
                f"{self._model.base_url.rstrip('/')}/chat/completions",
                headers=self._headers(),
                json=request_payload,
                # (connect, read): one scalar gives the connect phase the whole
                # read budget, so an unreachable host stalls a single attempt
                # for the rest of the game
                timeout=(
                    min(_CONNECT_TIMEOUT_SECONDS, max(0.01, remaining)),
                    max(0.01, remaining),
                ),
                stream=bool(request_payload.get("stream")),
            )

        # Per agent, not per process: every game meets the cold server on its
        # own first request, so one shared flag would grant the grace to
        # whichever game happened to go first and let the rest fail fast.
        initial_grace = 0.0
        if not self._http_initial_grace_used:
            initial_grace = _env_float("ARC3_HTTP_RETRY_INITIAL_SECONDS", 0.0)
        _diag(self, "request")
        try:
            response = _post_with_retries(
                lambda: post_chat(payload),
                retries=_env_int("ARC3_HTTP_RETRIES", 3),
                base_seconds=_env_float("ARC3_HTTP_RETRY_BASE_SECONDS", 5.0),
                max_seconds=_env_float("ARC3_HTTP_RETRY_MAX_SECONDS", 5.0),
                initial_seconds=initial_grace,
            )
        finally:
            # consumed whether the call succeeded or failed: the grace covers
            # the server coming up, and by now it either did or will not
            self._http_initial_grace_used = True
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            message = f"{exc}"
            if detail:
                message += f" | response: {detail}"
            raise requests.RequestException(message) from exc
        if getattr(response, "status_code", 200) >= 400:
            detail = response.text.strip()
            message = f"{response.status_code} Error"
            if detail:
                message += f" | response: {detail}"
            raise requests.RequestException(message)
        if payload.get("stream"):
            # decode_unicode=True decodes each transport chunk INDEPENDENTLY, so a
            # multi-byte UTF-8 sequence straddling a chunk boundary is split into
            # two broken halves - an em-dash (E2 80 94) becomes "\u00e2" plus
            # fragments. Most characters survive, which is why short probes look
            # clean and only long streamed replies are corrupted. The assembler
            # already decodes bytes itself, and a complete SSE line is always a
            # whole UTF-8 sequence, so hand it raw bytes instead.
            payload = assemble_streamed_chat_response(
                response.iter_lines(decode_unicode=False)
            )
        else:
            payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            detail = payload.get("error") or {
                k: v for k, v in payload.items() if k not in ("choices",)
            }
            raise requests.RequestException(
                f"server returned no choices | body: {str(detail)[:600]}"
            )
        choice = choices[0]
        return _ChatCompletionResult(
            message=_extract_leaked_tool_calls(choice.get("message", {})),
            finish_reason=str(choice.get("finish_reason", "") or ""),
            usage=payload.get("usage"),
            served_by=str(payload.get("provider", "") or ""),
        )

    def _trim_tool_text(self, text: str) -> tuple[str, bool]:
        """Keep both ends and drop the middle.

        Truncating the tail loses the half that matters. An `error` field ends
        with `SomeError: <the explanation>` after the traceback, so a deep
        traceback pushed the explanation out entirely. And a snippet's `stdout`
        typically prints the state, acts, then prints the state again - so the
        tail holds the result and the head holds the setup.

        Weighted toward the HEAD: a snippet's output is mostly sequential
        reasoning that reads from the top, so two thirds there. The remaining
        third is ample for what has to survive at the end - the longest refusal
        message measures ~550 characters and a user-frame traceback ~250, well
        inside a 1365-character tail at the default budget.
        """
        limit = self._tool_output_chars
        if len(text) <= limit:
            return text, False
        omitted = len(text) - limit
        if not _get_env_bool("ARC3_MIDDLE_TRUNCATION", False):
            return f"{text[:limit]}\n... [truncated {omitted} chars]", True
        tail = limit // 3
        head = limit - tail
        return (
            f"{text[:head]}\n... [truncated {omitted} chars from the middle] ...\n"
            f"{text[-tail:]}",
            True,
        )

    def _summarize_planned_actions(self, value: Any) -> Any:
        if isinstance(value, dict):
            compacted = {
                key: self._summarize_planned_actions(item)
                for key, item in value.items()
            }
            planned_actions = compacted.pop("planned_actions", None)
            if isinstance(planned_actions, list):
                compacted["planned_action_count"] = len(planned_actions)
                action_result = compacted.get("action_result")
                if isinstance(action_result, dict):
                    executed_count = action_result.get("executed_count")
                    try:
                        compacted["executed_action_count"] = int(executed_count)
                    except (TypeError, ValueError):
                        compacted["executed_action_count"] = 1 if action_result.get("executed") else 0
            return compacted
        if isinstance(value, list):
            return [self._summarize_planned_actions(item) for item in value]
        return value

    def _render_tool_payload(self, payload: dict[str, Any], *, truncate_fields: tuple[str, ...] = ()) -> str:
        result = self._summarize_planned_actions(dict(payload))
        truncated = False
        for field in truncate_fields:
            value = result.get(field)
            if isinstance(value, str):
                result[field], field_truncated = self._trim_tool_text(value)
                truncated = truncated or field_truncated
        if truncated:
            result["truncated"] = True
            result["truncation_note"] = (
                f"Output longer than {self._tool_output_chars} characters per field; the middle was removed."
            )
        return json.dumps(result, indent=2)

    def _normalize_python_actions(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, dict):
            items = [value]
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                action_name = item.strip()
                if not action_name:
                    raise ValueError(f"Action {index} is empty.")
                normalized.append({"action": action_name})
                continue
            if isinstance(item, dict):
                action_name = str(item.get("action", "")).strip()
                if not action_name:
                    raise ValueError(f"Action {index} is missing an `action` field.")
                entry = {"action": action_name}
                if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                    raise ValueError(f"Action {index} uses legacy MOUSE x/y fields; use row and col.")
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized


    def _compact_action_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        compact = {
            "executed": bool(payload.get("executed")),
            "action_num": payload.get("action_num"),
            "level": payload.get("level"),
            "score": payload.get("score"),
            # `reward` is not carried: it is (levels_completed - previous) /
            # number_of_levels, so it reads 0.0 on every action that does not
            # finish a level. A model seeing 0.0 on almost every action can
            # reasonably take it for a verdict on the action, which it is not,
            # and `level_completed` in this same dict already says the thing it
            # was trying to say. The solver still computes it for the viewer
            # and the raw payload.
            "state": payload.get("state"),
            "valid_actions": payload.get("valid_actions", []),
            "board_changed": bool(payload.get("board_changed")),
            "done": bool(payload.get("done")),
            "level_completed": bool(payload.get("level_completed")),
            "game_over": bool(payload.get("game_over")),
            "run_complete": bool(payload.get("run_complete")),
            "action_display": payload.get("action_display") or payload.get("action_name"),
        }
        if "gameplay_changed" in payload:
            compact["gameplay_changed"] = bool(payload.get("gameplay_changed"))
        executed_actions = payload.get("executed_actions")
        if isinstance(executed_actions, list):
            compact["executed_actions"] = [str(action).strip() for action in executed_actions if str(action).strip()]
        elif compact.get("executed") and compact.get("action_display"):
            compact["executed_actions"] = [str(compact["action_display"]).strip()]
        batch_size = int(payload.get("requested_count") or payload.get("executed_count") or 1)
        if batch_size > 1 or bool(payload.get("stopped_early")):
            compact["requested_count"] = payload.get("requested_count", batch_size)
            compact["executed_count"] = payload.get("executed_count", batch_size)
            compact["stopped_early"] = bool(payload.get("stopped_early"))
        # blocked_action names the refused action for the next turn's opener;
        # without it the model was told "the sequence was stopped before None"
        for passthrough_key in ("observation_action_index", "change_scope", "interior_changed", "border_changed",
                                "changed_cells", "interior_changed_cells", "border_changed_cells", "transient_pixels", "no_op_per_action", "skipped_actions", "gameplay_changed_per_action", "action_trace", "no_op", "fatal_action", "blocked_action", "action_echo", "animation",
                                "view_shift", "view_shifts", "view_offset", "view_offset_uncertain",
                                "sim_mismatch", "sim_guard_error"):
            if passthrough_key in payload:
                compact[passthrough_key] = payload.get(passthrough_key)
        # always present (None when the batch ran to completion): r['stop_reason'] raised KeyError 8 times in r4-r7
        compact["stop_reason"] = payload.get("stop_reason") or None
        if payload.get("stop_detail"):
            compact["stop_detail"] = payload.get("stop_detail")
        for timing_key in ("run_elapsed_seconds", "time_remaining_seconds"):
            if timing_key in payload:
                compact[timing_key] = payload.get(timing_key)
        # the session's own clock, which is also what decides when the game
        # stops - so the endgame phase cannot drift away from the deadline it
        # is meant to anticipate. It starts at session construction, so a game
        # waiting on the gate is already ageing, which is the same convention
        # max_runtime_s_per_game uses.
        elapsed = payload.get("run_elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            self._game_elapsed_seconds = float(elapsed)
        if payload.get("error"):
            compact["error"] = payload.get("error")
        return compact

    def _run_python_tool(self, state_path: Path, arguments: dict[str, Any]) -> _ToolDispatchResult:
        self._ensure_session(state_path)
        code = str(arguments.get("code", "")).rstrip()
        if not code:
            return _ToolDispatchResult(json.dumps({"error": "python requires a non-empty `code` string."}, indent=2))
        try:
            compile(code, "<python_tool>", "exec")
        except SyntaxError as exc:
            return _ToolDispatchResult(json.dumps({"error": f"Python syntax error: {exc}"}, indent=2))

        current_frame, history_entries = load_runtime_state(state_path)
        valid_actions = list(_normalize_valid_actions(self._current_valid_actions))

        def _serialized_runtime_state(
            *,
            next_valid_actions: list[str] | None = None,
            last_action_call_result: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            refreshed_frame, refreshed_history = load_runtime_state(state_path)
            current_frame_payload = _ascii_frame_view_payload(refreshed_frame)
            if isinstance(next_valid_actions, list):
                sanitized_actions = [str(item).strip() for item in next_valid_actions if str(item).strip()]
            else:
                sanitized_actions = list(valid_actions)
            persisted_action_result = (
                last_action_call_result
                if isinstance(last_action_call_result, dict)
                else self._last_action_call_result
            )
            return {
                "current_frame": current_frame_payload,
                "history": _ascii_history_view_payload(refreshed_history),
                "valid_actions": sanitized_actions,
                "last_action_call_result": (
                    dict(persisted_action_result)
                    if isinstance(persisted_action_result, dict)
                    else {}
                ),
            }

        terminal_action_result: dict[str, Any] | None = None
        snippet_executed_actions = 0
        snippet_scroll = ""  # set once an action in this snippet scrolled the view

        def _handle_animation(request: dict[str, Any]) -> dict[str, Any]:
            if self._step_env_callback is None:
                return {"error": "Animation frames are not available in this session."}
            raw = self._step_env_callback({"query": "animation"})
            record = raw.get("record") if isinstance(raw, dict) else None
            record = record if isinstance(record, dict) else None
            if str(request.get("kind") or "") == "frames":
                return build_animation_frames(record)
            if not _animation_timeline_enabled():
                return {}
            return build_animation_view(record)

        def _handle_action(
            actions: list[dict[str, Any]], *, stale_after: str | None = None,
            sim_query: Callable[..., Any] | None = None,
        ) -> dict[str, Any]:
            nonlocal terminal_action_result, snippet_executed_actions, snippet_scroll
            if self._step_env_callback is None:
                raise RuntimeError("action(actions) is not available in this session.")
            normalized_actions = self._normalize_python_actions(actions)
            clicks = [item for item in normalized_actions if to_engine_action(item.get("action")) == "ACTION6"]
            if snippet_scroll and clicks:
                # same rule as the batch pause, across action() calls of one snippet
                def _shown(item: dict[str, Any]) -> str:
                    if to_engine_action(item.get("action")) == "ACTION6":
                        return f"MOUSE(row={item.get('row')}, col={item.get('col')})"
                    return to_model_action(to_engine_action(item.get("action")) or "") or str(item.get("action"))

                click = _shown(clicks[0])
                detail = (f"{click} was NOT EXECUTED: the view scrolled {snippet_scroll} earlier in this snippet, so "
                          "its position may be out of date. Nothing in this call ran. Look at the current board, then "
                          "send the next action in a new call.")
                refusal = {
                    "executed": False,
                    "stop_reason": "stale_state",
                    "stop_detail": detail,
                    "blocked_action": click,
                    "requested_count": len(normalized_actions),
                    "executed_count": 0,
                    "stopped_early": True,
                    "skipped_actions": [_shown(item) for item in normalized_actions],
                    "valid_actions": list(self._current_valid_actions),
                }
                return {
                    "action_result": refusal,
                    "state": _serialized_runtime_state(),
                }
            if terminal_action_result is not None:
                reason = _terminal_action_reason(terminal_action_result) or "terminal_state"
                compact_payload = {
                    "executed": False,
                    "action_num": terminal_action_result.get("action_num"),
                    "level": terminal_action_result.get("level"),
                    "score": terminal_action_result.get("score"),
                    "state": terminal_action_result.get("state"),
                    "valid_actions": [],
                    "board_changed": False,
                    "done": bool(terminal_action_result.get("done")),
                    "level_completed": bool(terminal_action_result.get("level_completed")),
                    "game_over": bool(terminal_action_result.get("game_over")),
                    "run_complete": bool(terminal_action_result.get("run_complete")),
                    "requested_count": len(normalized_actions),
                    "executed_count": 0,
                    "stopped_early": True,
                    "stop_reason": f"previous_{reason}",
                    "stop_detail": _terminal_action_stop_detail(reason),
                }
                self._last_action_call_result = dict(compact_payload)
                return {
                    "action_result": compact_payload,
                    "state": _serialized_runtime_state(
                        next_valid_actions=[],
                        last_action_call_result=compact_payload,
                    ),
                }
            if reset_exposed():
                reset_positions = [
                    index for index, item in enumerate(normalized_actions)
                    if to_engine_action(item.get("action")) == "RESET"
                ]
                if reset_positions and (snippet_executed_actions or reset_positions != [0]):
                    detail = (
                        "RESET must be the first executed game action in a Python snippet. "
                        "This call executed no actions. Start a new snippet with RESET "
                        "as its first action; inspection or computation may come before it."
                    )
                    refusal = {
                        "executed": False,
                        "stop_reason": "reset_not_first",
                        "stop_detail": detail,
                        "error": detail,
                        "valid_actions": list(self._current_valid_actions),
                    }
                    self._last_action_call_result = dict(refusal)
                    return {
                        "action_result": refusal,
                        "state": _serialized_runtime_state(last_action_call_result=refusal),
                    }
            # No pre-check here: the solver's per-position hook covers every
            # action of every call, including the first. Checking in both
            # places consumed the override slot twice - the pre-check armed it,
            # the retry cleared it, and the hook then blocked the very action
            # the model had been invited to repeat.
            step_arguments: dict[str, Any] = {"actions": normalized_actions}
            if stale_after:
                # forwarded as an argument so the solver can weigh it AFTER its
                # own per-position guards, which name a specific action and
                # carry an override
                step_arguments["stale_after"] = stale_after
            if sim_query is not None:
                # r15: the solver asks the sandbox after each executed action of the batch whether the
                # model's step(observe(before), action) predicted it, and stops at the first miss
                step_arguments["sim_query"] = sim_query
            raw_payload = self._step_env_callback(step_arguments)
            if not isinstance(raw_payload, dict):
                raise RuntimeError("action(actions) did not return a JSON-like payload.")
            if reset_exposed() and raw_payload.get("executed"):
                # Use actual execution, not submitted batch size: a refused
                # call does not consume the snippet's first-action allowance.
                snippet_executed_actions += max(1, int(raw_payload.get("executed_count") or 1))
            if _animation_enabled():
                # Held on the agent, not routed through the sandbox: the chain
                # is dropped by compaction so the raw grids never become text,
                # which also meant it never reached the summary. It is only
                # ever used to render an image here, so it has no business
                # crossing the pipe twice.
                self._last_animation_chain = raw_payload.get("animation_chain") or []
                # Whether the LAST executed action moved anything, which is not
                # the same as whether the sequence did: a batch of UP, DOWN,
                # LEFT, SPACE reports gameplay_changed for the batch, so an
                # inert SPACE at the end was hidden behind three moves that
                # were not. The image exists for the action that animated, so
                # it is that action's flag that decides.
                per_action = raw_payload.get("gameplay_changed_per_action")
                if isinstance(per_action, list) and per_action:
                    self._last_action_gameplay_changed = bool(per_action[-1])
                else:
                    self._last_action_gameplay_changed = raw_payload.get(
                        "gameplay_changed"
                    )
            compact_payload = self._compact_action_result(raw_payload)
            scrolls = compact_payload.get("view_shifts") or (
                [{"shift": compact_payload["view_shift"]}] if compact_payload.get("view_shift") else [])
            if scrolls and compact_payload.get("executed"):
                from inference.utils.view_shift import describe as _describe_shift
                snippet_scroll = _describe_shift(*scrolls[-1]["shift"])
            next_valid_actions = raw_payload.get("valid_actions")
            if isinstance(next_valid_actions, list):
                self._current_valid_actions = _normalize_valid_actions(next_valid_actions)
            if compact_payload.get("executed") and _terminal_action_reason(compact_payload):
                terminal_action_result = compact_payload
            if compact_payload.get("executed"):
                # A refusal executes nothing and must not replace the result of the
                # last call that did run; the refusal itself is still returned.
                self._last_action_call_result = dict(compact_payload)
            return {
                "action_result": compact_payload,
                "state": _serialized_runtime_state(
                    next_valid_actions=next_valid_actions if isinstance(next_valid_actions, list) else None,
                ),
            }

        sandbox_result = run_sandboxed_python(
            code=code,
            timeout_seconds=self._python_timeout,
            initial_state=_serialized_runtime_state(),
            action_handler=_handle_action,
            animation_handler=_handle_animation if _animation_enabled() else None,
            kept_functions=self._retained_sources(),
            retain_imports=_get_env_bool("ARC3_PERSISTENT_FUNCTIONS_IMPORTS", False),
            repair_hints=_get_env_bool("ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS", False),
            sim_guard=_sim_guard_enabled(),
            sim_border=_get_env_int("ARC3_NOOP_GUARD_BORDER", 4),
        )

        action_results = [
            item
            for item in sandbox_result.get("action_results") or []
            if isinstance(item, dict)
        ]
        payload: dict[str, Any] = {"tool": "python"}
        self._record_retained_functions(sandbox_result, payload)
        from inference.agent import goal_check
        if goal_check.defines(code) and goal_check.NAME in (getattr(self, "_kept_functions", None) or {}):
            # the goal check covers the level on which goal_violations was last defined
            self._goal_defined_level = int(getattr(current_frame, "level", 0) or 0) or None
        rendered_stdout = str(sandbox_result.get("stdout", "") or "")
        rendered_error = str(sandbox_result.get("error", "") or "")
        if rendered_error and _get_env_bool("ARC3_TOOL_OBSERVATION_ERROR_FIX", True):
            confirmed = [a for r in action_results for a in (r.get("executed_actions") or [])]
            skipped = [a for r in action_results for a in (r.get("skipped_actions") or [])]
            rendered_error = (
                "PYTHON INTERRUPTED; snippet incomplete. Confirmed executed actions: "
                + json.dumps(confirmed) + "; reported skipped actions: " + json.dumps(skipped)
                + ". Unreached code produced no observations; unknown dispatch outcomes remain unknown.\n"
                + rendered_error
            )
        if rendered_error:
            payload["error"] = rendered_error
            if rendered_stdout:
                payload["stdout"] = rendered_stdout
        else:
            payload["returncode"] = 0
            if rendered_stdout:
                payload["stdout"] = rendered_stdout
            elif sandbox_result.get("result") is not None:
                payload["result"] = sandbox_result.get("result")
            elif action_results:
                if len(action_results) == 1:
                    payload["result"] = action_results[-1]
                else:
                    payload["result"] = {
                        "action_calls": len(action_results),
                        "last_action_call_result": action_results[-1],
                    }

        # Host-owned diagnostics remain visible even if generated code prints "noop".
        execution_errors = sandbox_result.get("action_execution_errors") or []
        if execution_errors:
            payload["action_execution_errors"] = execution_errors

        step_executed = any(bool(item.get("executed")) for item in action_results)
        made_progress = any(
            bool(item.get("gameplay_changed"))
            or bool(item.get("level_completed"))
            or bool(item.get("game_over"))
            or bool(item.get("run_complete"))
            for item in action_results
        )
        if step_executed:
            self._last_step_summary = self._summarize_step_sequence(action_results)
            self._note_level_transition(self._last_step_summary)
        elif self._last_step_summary and not self._last_step_summary.get("stale"):
            # carried over across an exchange that executed nothing: mark it so
            # the next prompt reminds instead of re-narrating it as fresh
            self._last_step_summary = {**self._last_step_summary, "stale": True}
        return _ToolDispatchResult(
            self._render_tool_payload(payload, truncate_fields=("stdout", "error", "result")),
            step_executed=step_executed,
            made_progress=made_progress,
        )

    def _dispatch_tool(self, state_path: Path, name: str, arguments: dict[str, Any]) -> _ToolDispatchResult:
        self._ensure_session(state_path)
        if name == "python":
            return self._run_python_tool(state_path, arguments)
        return _ToolDispatchResult(json.dumps({"error": f"Unknown tool: {name}"}, indent=2))

    def _calibrate_from_usage(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        usage: dict[str, Any] | None,
    ) -> str:
        """Update the characters-per-token divisor from a served request.

        The images are subtracted using the SAME formula that will re-add them,
        so the two cancel and the divisor measures text alone. Clamped because
        an under-count risks context overflow, which is the dangerous
        direction; a bad measurement should cost accuracy, not a rejected
        request."""
        if not _calibrate_text_tokens() or not isinstance(usage, dict):
            return ""
        try:
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
        except (TypeError, ValueError):
            return ""
        if prompt_tokens <= 0:
            return ""
        scrubbed, image_tokens = _split_messages_for_estimate(list(messages))
        payload: dict[str, Any] = {"messages": scrubbed}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = _request_tool_choice(tools)
        try:
            rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            return ""
        text_tokens = prompt_tokens - image_tokens
        if text_tokens <= 0 or not rendered:
            return ""
        measured = len(rendered) / text_tokens
        clamped = min(_text_token_chars_max(), max(_text_token_chars_min(), measured))
        previous = self._text_chars_per_token
        self._text_chars_per_token = clamped
        if previous is None or abs(clamped - previous) >= 0.05:
            note = "" if clamped == measured else f" (clamped from {measured:.2f})"
            # returned rather than written here: append_transcript is a closure
            # inside analyze(), not reachable from a method
            return (
                f"text_token_calibration: {clamped:.2f} chars/token{note} "
                f"[prompt_tokens {prompt_tokens}, images {image_tokens}, "
                f"text chars {len(rendered)}]"
            )
        return ""

    def _append_context_message(
        self, messages: list[dict[str, Any]], message: dict[str, Any]
    ) -> None:
        """Append and count new context once, before any history can be evicted.

        Use the context estimator for all roles, including reasoning and image
        tokens. Subtract the empty request envelope; static system/tool schema
        tokens and re-sent history are not new context. The summary exchange
        itself bypasses this helper and starts a fresh interval.
        """
        messages.append(message)


    def _estimate_request_input_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        image_tokens = 0
        if _image_token_estimate_enabled():
            messages, image_tokens = _split_messages_for_estimate(messages)
        payload: dict[str, Any] = {"messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = _request_tool_choice(tools)
        return _estimate_tokens(payload, self._text_chars_per_token) + image_tokens

    def _drop_oldest_history_block(self, history: list[dict[str, Any]], *, preserve_recent: int) -> bool:
        removable = len(history) - preserve_recent
        if removable <= 0:
            return False
        # Never drop the last user message. Dropping one and draining to the
        # next is fine while another exists; when none does, the drain runs to
        # preserve_recent and _drop_until_first_user_message then empties the
        # history outright - the request becomes the system prompt alone and the
        # server answers "No user query found in messages".
        #
        # Reachable mid-turn, where the list ends with a tool result rather than
        # the opener, so preserve_recent protects the tool message and the
        # opener is the first thing popped. Seen in a live request log as a
        # 1-message request at request_index_within_turn=2. Pre-existing and
        # independent of the summary drain, which only makes it easier to reach
        # by evicting further in one step.
        if not any(str(m.get("role", "")).strip() == "user" for m in history[1:]):
            return False
        first = history.pop(0)
        first_role = str(first.get("role", "")).strip()
        if first_role in {"assistant", "tool"}:
            while history and history[0].get("role") == "tool" and len(history) > preserve_recent:
                history.pop(0)
            return True
        while history and history[0].get("role") == "tool" and len(history) > preserve_recent:
            history.pop(0)
        # Only navigate by summaries when there is one left to land on, and only
        # one that is a REQUEST (a user message) and sits within the removable
        # range. Checking merely that a summary exists somewhere let the loop
        # skip every remaining user message when the last one was already at the
        # head, stripping history down to preserve_recent with no user message
        # in it at all - which the chat template rejects outright:
        # "No user query found in messages", HTTP 400, turn lost.
        while history and history[0].get("role") != "user" and len(history) > preserve_recent:
            history.pop(0)
        return True

    def _keep_recent_history_turns(
        self,
        messages: list[dict[str, Any]],
        *,
        max_turns: int,
        drain_turns: int = 0,
        force_drain: bool = False,
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        if max_turns <= 0:
            # explicit opt-out: leave the token budget as the only limit
            return list(messages)

        drain_turns = min(max(0, drain_turns), max(0, max_turns - 1))
        if drain_turns:
            present = sum(
                1
                for message in messages
                if str(message.get("role", "")).strip() == "assistant"
            )
            # trigger at the cap unless this commit has already invalidated the
            # prefix for token reasons, in which case take the deep cut now
            if present > max_turns or force_drain:
                max_turns = max_turns - drain_turns
            elif not force_drain:
                return list(messages)

        kept_reversed: list[dict[str, Any]] = []
        assistant_turns = 0
        for message in reversed(messages):
            kept_reversed.append(message)
            if str(message.get("role", "")).strip() == "assistant":
                assistant_turns += 1
                if assistant_turns >= max_turns:
                    break

        kept = list(reversed(kept_reversed))
        while kept and str(kept[0].get("role", "")).strip() == "tool":
            kept.pop(0)
        return kept

    def _drop_until_first_user_message(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        trimmed = list(history)
        while trimmed and str(trimmed[0].get("role", "")).strip() != "user":
            trimmed.pop(0)
        return trimmed

    def _note_history_evicted(self) -> None:
        # The scheduler consumes the first flag at the next handover. The
        # second stays set so summaries remain visible after history is lost.
        self._context_was_trimmed = True
        self._has_evicted = True

    def _persistent_history_messages(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        original_history = messages[1:]
        keep_images = _history_image_keep()
        if keep_images is not None:
            # after pruning, so a dropped resumption opener cannot consume a
            # retention slot
            messages = _apply_history_image_window(messages, keep_images)
        before_trim = len(messages)
        trimmed = self._trim_messages_for_context(messages, tools=tools)
        if not trimmed:
            if original_history:
                self._note_history_evicted()
            return []
        token_trim_dropped = len(trimmed) < before_trim
        trimmed_history = trimmed[1:]
        history = self._keep_recent_history_turns(
            trimmed_history,
            max_turns=_persistent_history_assistant_turns(),
            drain_turns=_history_turn_drain(),
            force_drain=token_trim_dropped and _history_drain_coalesce(),
        )
        if (
            history
            and str(history[0].get("role", "")).strip() != "user"
            and len(trimmed_history) > len(history)
        ):
            previous_message = trimmed_history[len(trimmed_history) - len(history) - 1]
            if str(previous_message.get("role", "")).strip() == "user":
                history = [previous_message, *history]
        history = self._drop_until_first_user_message(history)
        # r10 lever 5: a real drain (messages dropped, not just images) is reported in the next
        # opener together with the level ledger; count the assistant replies that left the context
        def _replies(items):
            return sum(1 for m in items if str(m.get("role", "")).strip() == "assistant")
        dropped = _replies(original_history) - _replies(history)
        if dropped > 0:
            self._drain_pending = getattr(self, "_drain_pending", 0) + dropped
        # Include turn-cap eviction, control pruning and image removal, which
        # can change the cached prefix without the token trimmer dropping any
        # messages. Image removal can leave the message count unchanged.
        if history != original_history:
            self._note_history_evicted()
        return history

    def _priority_total_levels(self) -> int:
        """Read public game metadata through the existing session callback.

        No new solver state or prompt fields. A custom callback, old object,
        or unavailable metadata can fall back without interrupting a game.
        """
        try:
            callback = getattr(self, "_step_env_callback", None)
            session = getattr(callback, "__self__", None)
            game = getattr(session, "game", None)
            count = parse_total_levels(getattr(game, "number_of_levels", None))
        except Exception:
            count = None
        if count is not None:
            return count
        if not getattr(self, "_priority_level_count_warned", False):
            self._priority_level_count_warned = True
            log.warning("priority: total level count unavailable or invalid; using 10")
        return 10

    def _maybe_handover(self) -> None:
        """Re-price this game, and yield the slot if a gate is running.

        Called right after the trimmer and before the request goes out, which is
        where the prefix has just been invalidated. Consumes the trim flag: it
        describes the trim that produced THIS request, and leaving it set would
        keep the game demoted long after the prefix had been rebuilt.
        """
        if not self._context_was_trimmed:
            return
        self._context_was_trimmed = False
        summary = self._last_step_summary or {}
        level = int(summary.get("level") or 1)
        actions = max(
            0, _priority_action_count(summary) - self._actions_at_level_start
        )
        # getattr: an agent unpickled from before this field existed would
        # otherwise raise here on every turn
        tokens = max(
            0,
            self._session_generated_tokens
            - getattr(self, "_tokens_at_level_start", 0),
        )
        gate = _priority_gate()
        snapshot = None
        options = _priority_level_options()
        total_levels = (
            self._priority_total_levels()
            if options["normalize_score"] or options["tail_lookup"] else None
        )
        if gate is not None and _priority_refresh_enabled():
            pace = self._progress_pace.cost_multiplier() if _get_env_bool("ARC3_PRIORITY_PACE", False) else 1.0
            snapshot = PrioritySnapshot(level, actions, tokens, pace, total_levels)
            estimate = gate._snapshot_priority(snapshot, time.monotonic())
        else:
            estimate = _game_priority(
                level, actions, tokens, endgame=False,
                total_levels=total_levels,
            )
        # this request pays the prefill, so it drops to the bottom band; the one
        # after it, with the prefix rebuilt, returns to the warm band
        self._priority_current = estimate
        self._priority_next = estimate + _PRIORITY_BAND
        if gate is not None:
            _diag(self, "gate")
            gate.handover(estimate, snapshot)


    def _note_request_succeeded(self) -> None:
        """Promote to the warm band once a request has actually landed.

        Called on a completed response rather than on dispatch: a retry after a
        connection failure has rebuilt nothing, so the game stays in the bottom
        band until one lands.
        """
        nxt = getattr(self, "_priority_next", None)
        if nxt is not None:
            self._priority_current = nxt

    def _compact_context(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        append_transcript: Callable[[str, str], None],
    ) -> list[dict[str, Any]] | None:
        """r17 compaction: once the context nears the budget, ask the model for notes on the turns that are about to
        leave, drain as usual, and put the notes right after the system prompt. Returns the new message list, or
        None when nothing needs to leave yet. On any failure the drain still happens (the pre-r17 behaviour)."""
        if not _compaction_enabled() or len(messages) < 3:
            return None
        used = self._estimate_request_input_tokens(messages, tools=tools)
        if used <= self._context_budget_tokens - _compaction_margin_tokens():
            return None
        drained = self._trim_messages_for_context(messages, tools=tools, force_drain=True)
        if len(drained) >= len(messages):
            return None

        def replies(items: list[dict[str, Any]]) -> int:
            return sum(1 for m in items if str(m.get("role", "")).strip() == "assistant")

        leaving = replies(messages) - replies(drained)
        request = [*messages, {"role": "user", "content": COMPACTION_REQUEST_PROMPT}]
        notes = ""
        status = ""
        try:
            # the same request settings as a turn (tools, tool_choice, thinking), so the server reuses the cached
            # prefix of the conversation and only reads the request message; r17 runs with tool_choice none and
            # thinking off re-prefilled ~100k tokens per compaction
            result = self._chat_completion(request, tools=tools, max_tokens=_compaction_max_tokens())
            # real GPU time for the game totals, but not this turn's deliberation budget
            self._accumulate_usage_tokens(result.usage, count_toward_turn=False)
            notes = _normalize_message_content(result.message.get("content", "")).strip()
            generated = (result.usage or {}).get("completion_tokens")
            status = (f"finish_reason={result.finish_reason}, generated={generated}, "
                      f"reasoning {len(_extract_reasoning_text(result.message))} chars, "
                      f"tool calls {len(result.message.get('tool_calls') or [])}, notes {len(notes)} chars")
        except Exception as exc:  # the notes are optional; a failed request must not cost the turn
            status = f"request failed: {type(exc).__name__}: {str(exc)[:300]}"
        if len(notes) < 200:
            append_transcript("ANALYZER STATUS", f"context_compaction_failed ({status}); drained without notes.")
            return drained
        self._drain_pending = getattr(self, "_drain_pending", 0) + max(0, leaving)
        self._compaction_notes = True
        append_transcript(
            "CONTEXT COMPACTION",
            f"input ~{used} tokens; {leaving} older replies replaced by notes ({status}).\n{notes}",
        )
        note = _mark_control_message({"role": "user", "content": COMPACTION_NOTE_HEADER + notes}, "compaction")
        return [drained[0], note, *drained[1:]]

    def _trim_messages_for_context(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        preserve_recent: int = 1,
        extra_safety_tokens: int = 0,
        force_drain: bool = False,
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        system_message = messages[0]
        history = list(messages[1:])
        preserve_recent = max(0, preserve_recent)
        budget_tokens = max(1, self._context_budget_tokens - max(0, extra_safety_tokens))
        # capped at half the budget: a mis-set knob should cost history depth,
        # not erase history
        drain_tokens = min(_context_drain_tokens(), budget_tokens // 2)
        target_tokens = max(1, budget_tokens - drain_tokens) if drain_tokens else budget_tokens
        floor_mode = bool(drain_tokens) and _context_drain_floor_mode()
        # Normally eviction starts only once the estimate exceeds the budget.
        # After the server has rejected a request the estimate is known to be
        # wrong, so waiting for it to admit the request is too big would drop
        # nothing at all; force_drain enters the loop already draining and lets
        # the existing target - budget minus one drain, computed against the
        # real budget - decide where to stop.
        draining = bool(force_drain and drain_tokens)
        def _stop_early(used_tokens: int) -> bool:
            return False

        while history:
            used = self._estimate_request_input_tokens([system_message, *history], tools=tools)
            over_budget = used > budget_tokens
            if not over_budget and not draining:
                break
            if floor_mode and not over_budget:
                # the mandatory phase is done; keep draining only while the NEXT
                # drop would still leave us at or above the target. Looking
                # ahead rather than dropping and checking is what puts a real
                # floor under the retained history.
                if _stop_early(used):
                    break
                probe = list(history)
                if not self._drop_oldest_history_block(probe, preserve_recent=preserve_recent):
                    break
                after = self._estimate_request_input_tokens(
                    [system_message, *probe], tools=tools
                )
                if after < target_tokens:
                    break
                history[:] = probe
                continue
            if not floor_mode and used <= (target_tokens if draining else budget_tokens):
                break
            if _stop_early(used):
                break
            draining = True
            if not self._drop_oldest_history_block(history, preserve_recent=preserve_recent):
                break
        history = self._drop_until_first_user_message(history)
        # A dropped message invalidates the server's prefix for this
        # conversation, so the next request has to be prefilled from the
        # divergence point regardless. That makes it the one moment when
        # yielding the slot to another game costs nothing.
        if len(history) != len(messages) - 1:
            self._note_history_evicted()
        return [system_message, *history]

    def _force_reduce_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        preserve_recent: int = 1,
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        system_message = messages[0]
        history = list(messages[1:])
        if not self._drop_oldest_history_block(history, preserve_recent=max(0, preserve_recent)):
            return list(messages)
        self._note_history_evicted()
        return [system_message, *history]

    def analyze(
        self,
        state_path: Path,
        action_num: int,
        valid_actions: list[str] | None = None,
        step_env: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        transcript_path: Path | None = None,
        analysis_step: int | None = None,
        transcript_updated: Callable[[str], None] | None = None,
        request_timeout_seconds: float | None = None,
        should_stop: Callable[[], bool] | None = None,
        time_remaining_seconds: float | None = None,
    ) -> AnalyzerTurnResult | None:
        if not state_path.exists():
            return None
        self._time_remaining_seconds = time_remaining_seconds
        self._time_remaining_reference = time.monotonic()
        self._ensure_session(state_path)
        self._step_env_callback = step_env
        self._current_valid_actions = _normalize_valid_actions(valid_actions)

        analyzer_log = transcript_path or (state_path.parent / f"{state_path.stem}_analyzer.txt")
        prompt_log = _resolve_prompt_log_path(state_path)
        current_frame, history_entries = load_runtime_state(state_path)
        user_prompt = self._build_user_prompt(
            action_num,
            valid_actions=valid_actions,
            current_frame=current_frame,
            history_entries=history_entries,
            previous_step_summary=self._last_step_summary,
        )
        display_action_num = _display_action_number(action_num)

        with open(analyzer_log, "a", encoding="utf-8") as f:
            step_label = f"analysis_step={analysis_step} | " if analysis_step is not None else ""
            transcript_header = (
                f"\n--- {step_label}action={display_action_num} | "
                f"{time.strftime('%H:%M:%S')} | tool-agent ---\n"
            )
            f.write(transcript_header)
        transcript_parts = [transcript_header]

        def append_transcript(label: str, content: str) -> None:
            _append_transcript_section(analyzer_log, label, content)
            transcript_parts.append(_render_transcript_section(label, content))
            if transcript_updated is not None:
                transcript_updated("".join(transcript_parts))

        append_transcript("SYSTEM PROMPT", self._system_prompt)

        previous_history_messages = list(self._history_messages)
        preserve_history = True
        resuming_after_yield = bool(getattr(self, "_resume_after_yield", False))
        self._resume_after_yield = False
        self._resume_reason = ""
        if not resuming_after_yield and _persistent_functions_scope() == "turn":
            # A resumption is the same turn, so helpers survive it either way.
            if _persistent_functions() and getattr(self, "_kept_functions", None):
                self._retained_clear_notice = (
                    "Your retained functions were cleared at the start of this new turn; "
                    "redefine any you still need."
                )
            self._kept_functions = {}
        # A resumed invocation (after a request timeout or an exhausted tool-step
        # budget) re-opens with the same full opener describing the unchanged state.
        opening_message = self._build_user_message(user_prompt, current_frame)
        append_transcript("USER PROMPT", _message_display_text(opening_message))
        if resuming_after_yield:
            # Tag EVERY resumption message, not just the short form. A yield
            # leaves this turn's original opener in history, so the message
            # appended here is a duplicate restatement of unchanged state
            # (same step, level, board, diffs, ledger - a no-execution yield
            # changes nothing). It is needed for the request that resumes the
            # turn and is pure duplication afterwards, which is precisely what
            # ARC3_PRUNE_CONTROL_CONTEXT exists to drop.
            _mark_control_message(opening_message, "resume")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt}, *self._history_messages,
        ]
        self._append_context_message(messages, opening_message)
        compacted = self._compact_context(messages, tools=self._tools(state_path), append_transcript=append_transcript)
        if compacted is not None:
            messages = compacted
        messages = self._trim_messages_for_context(
            messages,
            tools=self._tools(state_path),
            preserve_recent=1,
        )
        step_executed = False
        turn_tool_call_count = 0
        captured_reasoning = ""
        latest_request_messages: list[dict[str, Any]] | None = None
        latest_request_tools: list[dict[str, Any]] | None = None
        latest_request_tool_choice: str | None = None
        latest_request_index = 0
        turn_started_at = time.monotonic()
        # per TURN, not per session: the budget is how much the model may
        # generate before the turn ends, and a resumption continues the same
        # turn, so this resets where the clock does
        self._turn_generated_tokens = 0
        yielded_control_reason: str | None = None

        def control_yield_reason() -> str | None:
            if should_stop is not None:
                try:
                    if should_stop():
                        return "stop_requested"
                except Exception as exc:
                    log.warning("analyzer stop check failed at action %d: %s", display_action_num, exc)
            if self._yield_seconds is not None and (time.monotonic() - turn_started_at) >= self._yield_seconds:
                return "turn_time_budget"
            if (
                self._yield_tokens is not None
                and self._turn_generated_tokens >= self._yield_tokens
            ):
                return "turn_token_budget"
            return None

        try:
            turn_count = 0
            while self._tool_steps is None or turn_count < self._tool_steps:
                yielded_control_reason = control_yield_reason()
                if yielded_control_reason is not None:
                    break
                turn_count += 1
                tools = self._tools(state_path)
                tool_choice = _request_tool_choice(tools)
                compacted = self._compact_context(messages, tools=tools, append_transcript=append_transcript)
                if compacted is not None:
                    messages = compacted
                messages = self._trim_messages_for_context(messages, tools=tools)
                self._maybe_handover()
                yielded_control_reason = control_yield_reason()
                if yielded_control_reason:
                    break
                latest_request_messages = json.loads(json.dumps(messages))
                latest_request_tools = json.loads(json.dumps(tools))
                latest_request_tool_choice = tool_choice
                latest_request_index = turn_count
                _write_prompt_log_snapshot(
                    prompt_log,
                    model_id=self._model.model_id,
                    base_url=self._model.base_url,
                    display_action_num=display_action_num,
                    analysis_step=analysis_step,
                    request_index=turn_count,
                    messages=latest_request_messages,
                    tools=latest_request_tools,
                    tool_choice=tool_choice,
                    transcript="".join(transcript_parts),
                )
                try:
                    request_kwargs: dict[str, Any] = {"tools": tools}
                    if request_timeout_seconds is not None:
                        request_kwargs["request_timeout_seconds"] = request_timeout_seconds
                    if self._save_request_logs:
                        _append_request_snapshot(
                            _resolve_request_log_path(state_path),
                            state_name=state_path.name,
                            messages=latest_request_messages,
                            tools=latest_request_tools,
                            event="request",
                            tool_choice=latest_request_tool_choice,
                            analysis_step=analysis_step,
                            action=display_action_num,
                            request_index_within_turn=latest_request_index,
                            chat_template_kwargs=self._harness_template_kwargs(),
                        )
                    result = self._chat_completion(messages, **request_kwargs)
                    _calibration_note = self._calibrate_from_usage(
                        messages, request_kwargs.get("tools"), result.usage
                    )
                    if _calibration_note:
                        append_transcript("ANALYZER STATUS", _calibration_note)
                    self._accumulate_usage_tokens(result.usage)
                    self._note_request_succeeded()
                    _diag(self, "local")
                    if self._save_request_logs:
                        _append_request_snapshot(
                            _resolve_request_log_path(state_path),
                            state_name=state_path.name,
                            messages=[],  # already in the matching "request" record; not repeated
                            tools=[],
                            event="response",
                            tool_choice=latest_request_tool_choice,
                            analysis_step=analysis_step,
                            action=display_action_num,
                            request_index_within_turn=latest_request_index,
                            finish_reason=result.finish_reason,
                            served_by=result.served_by,
                            usage=result.usage,
                            chat_template_kwargs=self._harness_template_kwargs(),
                            response_message={
                                "role": "assistant",
                                "content": _normalize_message_content(result.message.get("content", "")),
                                "reasoning": _extract_reasoning_text(result.message),
                                "tool_calls": json.loads(json.dumps(result.message.get("tool_calls") or [])),
                            },
                        )
                except requests.RequestException as exc:
                    if not _is_context_length_error(exc):
                        raise
                    # A rejected request means the estimate was wrong, and 512
                    # tokens only helps when it was wrong by less than that -
                    # otherwise nothing is dropped and the fallback removes one
                    # block per rejected round trip. force_drain evicts down to
                    # the drain target instead, in one step, which is what the
                    # drain is for: land clear of the ceiling rather than just
                    # under it.
                    trimmed_messages = self._trim_messages_for_context(
                        messages,
                        tools=tools,
                        extra_safety_tokens=_CONTEXT_OVERFLOW_RETRY_TRIM_TOKENS,
                        force_drain=True,
                    )
                    if trimmed_messages == messages:
                        trimmed_messages = self._force_reduce_messages(messages)
                    if trimmed_messages == messages:
                        raise
                    append_transcript(
                        "ANALYZER STATUS",
                        "context_overflow_recovered: dropped older history after server rejected the request as too long.",
                    )
                    messages = trimmed_messages
                    continue
                truncated_tool_call = False
                raw_reasoning = _extract_reasoning_text(result.message)
                raw_content = _normalize_message_content(result.message.get("content", ""))
                tool_calls = json.loads(json.dumps(result.message.get("tool_calls") or []))
                tool_call_markup_in_text = _contains_tool_call_markup(raw_reasoning, raw_content)
                recovered_tool_calls_from_markup = False
                if not tool_calls and tool_call_markup_in_text:
                    tool_calls = _recover_tool_calls_from_markup(raw_reasoning, raw_content)
                    recovered_tool_calls_from_markup = bool(tool_calls)
                reasoning = _strip_tool_call_markup(raw_reasoning) if tool_call_markup_in_text else raw_reasoning
                content = _strip_tool_call_markup(raw_content) if tool_call_markup_in_text else raw_content
                malformed_argument_errors: list[str] = []
                for tool_call in tool_calls:
                    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    tool_name = str(function.get("name", "")).strip() or "unknown"
                    raw_arguments = function.get("arguments", "{}")
                    if isinstance(raw_arguments, str):
                        try:
                            json.loads(raw_arguments)
                        except json.JSONDecodeError as exc:
                            malformed_argument_errors.append(f"{tool_name}: invalid JSON arguments ({exc})")
                if malformed_argument_errors:
                    # Drop ALL calls from this reply rather than the bad one.
                    #
                    # An unparseable arguments string means the reply was cut
                    # off mid-call (finish_reason=length is the usual cause),
                    # so anything after the truncation point is missing and an
                    # earlier "valid" call in the same message expresses only
                    # part of an intent - dispatching half a plan is worse than
                    # dispatching none.
                    #
                    # Critically, the message must also never reach history:
                    # servers re-parse tool calls in the incoming message list
                    # when rendering the chat template, so one malformed call
                    # committed to history makes EVERY later request fail at
                    # render time, before inference - an unbreakable loop that
                    # the HTTP retry cannot help with. Content and reasoning
                    # are kept; only the calls are dropped, which routes this
                    # into the existing no-tool-call path.
                    log.warning(
                        "dropping %d malformed tool call(s) (finish_reason=%s): %s",
                        len(tool_calls),
                        result.finish_reason or "?",
                        "; ".join(malformed_argument_errors),
                    )
                    tool_calls = []
                    truncated_tool_call = str(result.finish_reason or "").strip() == "length"
                response_meta = _format_model_response_meta(
                    finish_reason=result.finish_reason,
                    served_by=result.served_by,
                    reasoning=reasoning,
                    content=content,
                    tool_calls=tool_calls,
                    tool_call_markup_in_text=tool_call_markup_in_text,
                    recovered_tool_calls_from_markup=recovered_tool_calls_from_markup,
                    malformed_argument_errors=malformed_argument_errors,
                )
                append_transcript(
                    "MODEL RESPONSE META",
                    response_meta,
                )
                assistant_message: dict[str, Any] = {"role": "assistant"}

                if reasoning:
                    captured_reasoning = reasoning
                    append_transcript("THINKING", reasoning)
                    for _reasoning_key in _reasoning_history_keys():
                        assistant_message[_reasoning_key] = reasoning

                if not tool_calls:
                    if content:
                        append_transcript("ASSISTANT", content)
                        assistant_message["content"] = content
                    elif reasoning:
                        assistant_message["content"] = None

                    if content or reasoning:
                        if (
                            not content
                            and reasoning
                            and _is_degenerate_text(reasoning)
                        ):
                            # A reasoning-only reply that is one character
                            # repeated to the output cap is a serving fault, not
                            # thinking: observed as exactly 6144 "!" from the
                            # first token, with content_chars 0 and no tool
                            # call, under speculative decoding. Keeping it cost
                            # far more than the lost turn - 12k characters of
                            # noise pushed the trimmer from 31 history messages
                            # to 12, evicting real history, and the model then
                            # concluded "my replies keep failing to emit the
                            # tool call" and started writing shorter answers to
                            # fix a formatting error it had not made.
                            # transcript AND log: the transcript records it
                            # where it happened, but a serving fault should be
                            # visible while the run is going rather than only
                            # on a later grep
                            log.warning(
                                "degenerate reasoning dropped: %d chars, no content "
                                "and no tool call - the reply was a repetition loop. "
                                "This is a serving fault, not the model reasoning.",
                                len(reasoning),
                            )
                            append_transcript(
                                "ANALYZER STATUS",
                                "degenerate_reasoning_dropped: the reply was a "
                                f"repetition loop ({len(reasoning)} chars) with no "
                                "content or tool call, and was replaced in context.",
                            )
                            assistant_message = _mark_control_message(
                                {"role": "assistant", "content": _DEGENERATE_CONTENT_STUB},
                                "stub",
                            )
                        elif content and _is_degenerate_text(content):
                            # a repetition loop feeds on its own output, so the
                            # garbage must not reach the next request's context
                            append_transcript(
                                "ANALYZER STATUS",
                                "degenerate_content_stubbed: assistant text was highly "
                                "repetitive and was replaced in context.",
                            )
                            assistant_message = {
                                "role": "assistant",
                                "content": _DEGENERATE_CONTENT_STUB,
                            }
                        elif (
                            not content
                            and reasoning
                            and str(result.finish_reason or "").strip() == "length"
                            and _truncation_recovery_enabled()
                        ):
                            # A reply cut off at the output limit carries ~12k tokens
                            # of unfinished deliberation. Keeping all of it fills the
                            # context and invites the model to resume the same loop;
                            # keep only its tail, which holds the latest conclusions.
                            append_transcript(
                                "ANALYZER STATUS",
                                f"truncated_reply_stubbed: {len(reasoning)} reasoning chars "
                                "hit the output limit without a tool call; only the tail is kept in context.",
                            )
                            assistant_message = _mark_control_message(
                                {
                                    "role": "assistant",
                                    "content": (
                                        "[My previous reply was cut off at the output limit before the tool "
                                        "call. Its last part was:] ..." + reasoning[-_TRUNCATED_TAIL_CHARS:]
                                    ),
                                },
                                "stub",
                            )
                        self._append_context_message(messages, assistant_message)
                    yielded_control_reason = control_yield_reason()
                    if yielded_control_reason is not None:
                        break
                    if tool_call_markup_in_text:
                        followup_prompt = MARKUP_TOOL_CALL_PROMPT
                    elif truncated_tool_call or str(result.finish_reason or "").strip() == "length":
                        followup_prompt = TRUNCATED_REPLY_PROMPT
                    elif malformed_argument_errors:
                        followup_prompt = MALFORMED_TOOL_CALL_PROMPT
                    else:
                        followup_prompt = NO_TOOL_CALL_PROMPT
                    stall_note = self._stall_note(turn_started_at, step_executed)
                    if stall_note:
                        followup_prompt += " " + stall_note
                    append_transcript("USER PROMPT", followup_prompt)
                    self._append_context_message(
                        messages,
                        _mark_control_message(
                            {"role": "user", "content": followup_prompt}, "nudge"
                        )
                    )
                    continue

                if content:
                    append_transcript("ASSISTANT", content)
                    assistant_message["content"] = content
                assistant_message["tool_calls"] = tool_calls
                turn_tool_call_count += len(tool_calls)
                self._append_context_message(messages, assistant_message)

                for tool_index, tool_call in enumerate(tool_calls):
                    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    tool_name = str(function.get("name", "")).strip()
                    raw_args = function.get("arguments", "{}")
                    try:
                        if isinstance(raw_args, str):
                            arguments = json.loads(raw_args)
                        elif isinstance(raw_args, dict):
                            arguments = json.loads(json.dumps(raw_args))
                        else:
                            arguments = {}
                    except json.JSONDecodeError:
                        arguments = {}
                    rendered_tool_call = _render_tool_call_markup(tool_name, raw_args)
                    append_transcript(
                        f"TOOL CALL: {tool_name}",
                        rendered_tool_call or (json.dumps(arguments, indent=2) if arguments else "{}"),
                    )
                    dispatch = self._dispatch_tool(state_path, tool_name, arguments)
                    if dispatch.step_executed:
                        step_executed = True
                    tool_content = dispatch.content
                    stall_note = self._stall_note(turn_started_at, step_executed)
                    if stall_note:
                        tool_content = _with_harness_note(tool_content, stall_note)
                    append_transcript(f"TOOL RESULT: {tool_name}", _render_tool_result_display(tool_content))
                    self._append_context_message(
                        messages,
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", ""),
                            "content": tool_content,
                        }
                    )
                    if dispatch.step_executed:
                        if tool_index < len(tool_calls) - 1:
                            preserve_history = False
                        break
                    yielded_control_reason = control_yield_reason()
                    if yielded_control_reason is not None:
                        if tool_index < len(tool_calls) - 1:
                            preserve_history = False
                        break
                if yielded_control_reason is not None:
                    break
                if step_executed:
                    break

        except requests.RequestException as exc:
            if _is_read_timeout(exc) and _yield_on_timeout_enabled():
                # Keep this turn's completed exchanges and hand control back to
                # the solver, which re-invokes on the same game state.
                append_transcript(
                    "ANALYZER STATUS",
                    f"request_timeout (yielding, history preserved): {exc}",
                )
                log.warning(
                    "analyzer request timed out at action %d; yielding with %d tool call(s) preserved: %s",
                    display_action_num,
                    turn_tool_call_count,
                    exc,
                )
                self._resume_after_yield = True
                self._resume_reason = "timeout"
                if latest_request_messages is not None:
                    _write_prompt_log_snapshot(
                        prompt_log,
                        model_id=self._model.model_id,
                        base_url=self._model.base_url,
                        display_action_num=display_action_num,
                        analysis_step=analysis_step,
                        request_index=latest_request_index,
                        messages=latest_request_messages,
                        tools=latest_request_tools,
                        tool_choice=latest_request_tool_choice,
                        transcript="".join(transcript_parts),
                    )
                # preserve_history stays True, so the finally block commits
                # this turn's exchanges exactly as a normal yield would.
                return AnalyzerTurnResult(
                    step_executed=False,
                    reasoning=captured_reasoning,
                    yielded_control=True,
                )
            append_transcript("ANALYZER STATUS", f"request_error: {exc}")
            preserve_history = False
            # Generated tokens remain spent even if request history rolls back.
            if latest_request_messages is not None:
                _write_prompt_log_snapshot(
                    prompt_log,
                    model_id=self._model.model_id,
                    base_url=self._model.base_url,
                    display_action_num=display_action_num,
                    analysis_step=analysis_step,
                    request_index=latest_request_index,
                    messages=latest_request_messages,
                    tools=latest_request_tools,
                    tool_choice=latest_request_tool_choice,
                    transcript="".join(transcript_parts),
                )
            log.warning("analyzer request failed at action %d: %s", display_action_num, exc)
            return AnalyzerTurnResult(step_executed=False, retryable_failure=True, reasoning=captured_reasoning)
        except Exception as exc:
            append_transcript("ANALYZER STATUS", f"error: {exc}")
            preserve_history = False
            # Generated tokens remain spent even if request history rolls back.
            if latest_request_messages is not None:
                _write_prompt_log_snapshot(
                    prompt_log,
                    model_id=self._model.model_id,
                    base_url=self._model.base_url,
                    display_action_num=display_action_num,
                    analysis_step=analysis_step,
                    request_index=latest_request_index,
                    messages=latest_request_messages,
                    tools=latest_request_tools,
                    tool_choice=latest_request_tool_choice,
                    transcript="".join(transcript_parts),
                )
            log.warning("analyzer failed at action %d: %s", display_action_num, exc)
            return None
        finally:
            if preserve_history:
                self._history_messages = self._persistent_history_messages(messages, tools=self._tools(state_path))
            else:
                self._history_messages = previous_history_messages
            solver_obj = getattr(self._step_env_callback, "__self__", None)
            if solver_obj is not None and hasattr(solver_obj, "action_guard_hook"):
                solver_obj.action_guard_hook = None
            self._step_env_callback = None
            self._current_valid_actions = []

        if step_executed:
            status_message = "Step executed."
        elif yielded_control_reason is not None:
            status_message = f"Yielded control to solver: {yielded_control_reason}."
        else:
            status_message = "No action(...) call was captured."

        status = (
            f"model: {self._model.model_id}\n"
            f"base_url: {self._model.base_url}\n"
            f"max_output_tokens: {self._max_output_tokens if self._max_output_tokens is not None else 'server default'}\n"
            f"reply_reserve_tokens: {self._reply_reserve_tokens}\n"
            f"context_budget_tokens: {self._context_budget_tokens}\n"
            f"request_safety_margin_tokens: {self._request_safety_margin_tokens}\n"
            f"tool_output_tokens: {self._tool_output_tokens}\n"
            f"yield_seconds: {self._yield_seconds if self._yield_seconds is not None else 'disabled'}\n"
            f"yield_tokens: {self._yield_tokens if self._yield_tokens is not None else 'disabled'}\n"
            f"available_tools: python\n"
            f"python_timeout_seconds: {self._python_timeout}\n"
            f"history_messages: {len(self._history_messages)}\n"
            f"step_executed: {step_executed}\n"
            f"message: {status_message}"
        )
        append_transcript("ANALYZER STATUS", status)
        if latest_request_messages is not None:
            _write_prompt_log_snapshot(
                prompt_log,
                model_id=self._model.model_id,
                base_url=self._model.base_url,
                display_action_num=display_action_num,
                analysis_step=analysis_step,
                request_index=latest_request_index,
                messages=latest_request_messages,
                tools=latest_request_tools,
                tool_choice=latest_request_tool_choice,
                transcript="".join(transcript_parts),
            )
        # An invocation that executed nothing will be re-invoked by the solver
        # on the same game state, so whatever it appends next is a duplicate
        # restatement. This covers BOTH non-execution exits: the time budget
        # (yielded_control_reason set) and the tool-step budget (loop condition
        # went false, no yield reason). Execution and yielding are mutually
        # exclusive, so `not step_executed` is the whole condition.
        self._resume_after_yield = not step_executed
        if yielded_control_reason is not None:
            self._resume_reason = (
                "yield_tools" if turn_tool_call_count > 0 else "yield_bare"
            )
        else:
            self._resume_reason = (
                "tool_steps" if turn_tool_call_count > 0 else "yield_bare"
            )
        if not step_executed and turn_tool_call_count == 0 and self._last_step_summary:
            # no dispatch ran, so _dispatch_tool never marked the summary
            # stale; without this the next opener re-narrates an older
            # execution as if it had just happened (or claims nothing has run
            # yet) directly beneath this turn's own reasoning.
            self._last_step_summary = {
                **self._last_step_summary,
                "stale": True,
                "stale_reason": "no_tool_call",
            }
        return AnalyzerTurnResult(
            step_executed=step_executed,
            reasoning=captured_reasoning,
            yielded_control=yielded_control_reason is not None,
        )

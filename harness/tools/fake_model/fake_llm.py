"""Scripted OpenAI-compatible fake model for end-to-end CPU tests of the ARC3 harness.

Replays the real actions recorded in a previous run (so level completions, deaths and
wins happen for real in the offline engine) and periodically injects edge cases:
long reasoning (token-budget yields), output-limit truncation with no tool call,
inspection-only calls that exercise the documented Python API, and a deliberate
second action after a no-op (stale-state path).
"""
import json, os, re, sys, threading, time, random
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
FP = json.load(open(os.path.join(HERE, "fingerprints.json")))
REPLAY = json.load(open(os.path.join(HERE, "replay_actions.json")))
LOG = os.environ.get("FAKE_LOG", os.path.join(HERE, "requests.jsonl"))
MODE = os.environ.get("FAKE_MODE", "replay")  # replay | edge
STATE = {}
LOCK = threading.Lock()
COUNTER = {"n": 0}

FINGERPRINT_CODE = (
    "fp = sum(ord(c)*(i%997+1) for i,c in enumerate(current_frame.ascii))\n"
    "print('FP=' + str(fp))\n"
)

INSPECT_CODE = """print('GAME={game}')
seg = current_frame.segmentation
print('nodes', len(seg['nodes']), 'level', current_frame.level, 'step', current_frame.step)
d = frame_diff()
print('diff', d.get('changed_cell_count'))
tl = last_animation_timeline
print('timeline_keys', sorted(tl.keys()) if hasattr(tl, 'keys') else type(tl).__name__)
fr = last_animation_frames
print('anim_frames', len(fr) if fr is not None else None)
print('hist', len(history), 'trans', len(transitions), 'valid', valid_actions)
print('prev_is_none', previous_frame is None)
import json
print('tl_json', len(json.dumps(last_animation_timeline)))
class Probe:
    def __init__(self, v):
        self.v = v
print('cls', Probe(3).v, isinstance(last_animation_frames, list))
def helper_{game_tag}(n):
    return n + 1
def goal_violations(frame):
    return []
"""


def to_action(disp):
    m = re.match(r"MOUSE\(row=(\d+), col=(\d+)\)", disp)
    if m:
        return {"action": "MOUSE", "row": int(m.group(1)), "col": int(m.group(2))}
    return disp


def action_code(game, batch, extra_second=False):
    acts = json.dumps([to_action(a) for a in batch])
    code = (
        f"print('GAME={game}')\n"
        "try:\n"
        f"    r = action({acts})\n"
        "    print('EXEC', len(r.get('executed_actions') or []))\n"
    )
    if extra_second:
        code += (
            "    r2 = action(" + json.dumps([to_action(batch[0])]) + ")\n"
            "    print('EXEC2', len(r2.get('executed_actions') or []))\n"
        )
    code += (
        "except Exception as e:\n"
        "    print('EXC', type(e).__name__)\n"
        "    print('EXEC', 0)\n"
    )
    return code


def find_game(messages):
    for m in reversed(messages):
        c = m.get("content")
        text = c if isinstance(c, str) else json.dumps(c)
        g = re.findall(r"GAME=([a-z0-9]{4}-[0-9a-f]{8})", text)
        if g:
            return g[-1]
        f = re.findall(r"FP=(\d+)", text)
        if f and f[-1] in FP:
            return FP[f[-1]]
    return None


def last_tool_text(messages):
    for m in reversed(messages):
        if m.get("role") == "tool":
            c = m.get("content")
            return c if isinstance(c, str) else json.dumps(c)
        if m.get("role") == "user":
            return None
    return None


def respond(payload):
    messages = payload.get("messages", [])
    game = find_game(messages)
    last = messages[-1] if messages else {}
    with LOCK:
        COUNTER["n"] += 1
        n = COUNTER["n"]
        st = STATE.setdefault(game or "?", {"ptr": 0, "turn": 0, "phase": 0, "seen_tool": set()})
    reasoning = "Thinking about the board. " * 20
    finish = "tool_calls"
    code = None
    if game is None:
        code = FINGERPRINT_CODE
    else:
        pend = st.setdefault("pending", set())
        for m in messages:
            if m.get("role") == "tool" and m.get("tool_call_id") in pend:
                c = m.get("content"); text = c if isinstance(c, str) else json.dumps(c)
                st["ptr"] += sum(int(x) for x in re.findall(r"EXEC2? (\d+)", text))
                pend.discard(m.get("tool_call_id"))
        acts = REPLAY.get(game, {}).get("actions", [])
        opener = last.get("role") == "user"
        if opener:
            st["turn"] += 1
        t = st["turn"]
        if MODE == "edge" and opener and t % 7 == 3:
            # output-limit truncation with no tool call
            words = ["mirror", "axis", "block", "shift", "row", "col", "goal", "test", "probe", "level", "bar", "path"]
            rr = random.Random(f"{game}|{t}")
            text = " ".join(rr.choice(words) + str(rr.randint(0, 99)) for _ in range(6000))
            return {"message": {"role": "assistant", "content": "", "reasoning_content": text},
                    "finish_reason": "length", "completion_tokens": 12288}
        if MODE == "edge" and opener and t % 5 == 2:
            reasoning = "Deliberating without acting. " * 600  # ~3.5k tokens -> yield path
            time.sleep(float(os.environ.get("FAKE_SLOW", "0")))
        if st["ptr"] >= len(acts):
            code = f"print('GAME={game}')\nprint('done replay')\n"
        elif opener and t % 2 == 1:
            code = INSPECT_CODE.format(game=game, game_tag=game.split('-')[0])
        else:
            k = random.Random(f"{game}|{st['ptr']}|{t}").randint(1, 4)
            batch = acts[st["ptr"]: st["ptr"] + k]
            code = action_code(game, batch, extra_second=(MODE == "edge" and t % 4 == 0))
    if game is not None and code and "action(" in code:
        st.setdefault("pending", set()).add(f"call_{game}_{n}")
    call = {"id": f"call_{game}_{n}", "type": "function",
            "function": {"name": "python", "arguments": json.dumps({"code": code})}}
    msg = {"role": "assistant", "content": "", "reasoning_content": reasoning, "tool_calls": [call]}
    return {"message": msg, "finish_reason": finish, "completion_tokens": max(50, len(reasoning) // 4 + len(code) // 3)}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._send({"ok": True})
        if self.path.startswith("/v1/models"):
            return self._send({"data": [{"id": "flashnext", "max_model_len": 139264}]})
        self._send({}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n))
        out = respond(payload)
        prompt_chars = sum(len(json.dumps(m.get("content"))) for m in payload.get("messages", []))
        usage = {"prompt_tokens": prompt_chars // 3, "completion_tokens": out["completion_tokens"],
                 "total_tokens": prompt_chars // 3 + out["completion_tokens"],
                 "prompt_tokens_details": {"cached_tokens": 0}}
        with LOCK, open(LOG, "a") as f:
            f.write(json.dumps({"t": time.time(), "payload": payload, "response": out["message"],
                                "finish_reason": out["finish_reason"]}) + "\n")
        self._send({"id": "x", "object": "chat.completion", "model": "flashnext",
                    "choices": [{"index": 0, "message": out["message"], "finish_reason": out["finish_reason"]}],
                    "usage": usage})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()

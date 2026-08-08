"""
feasibility.py - judges an operator's live request before player_ai ever acts
on it: a free, instant sanity filter first, then a full-context model review.

Stage 1 (quickReject) is deterministic and costs nothing. Its job is to keep
junk and prompt-injection-shaped text out of the model's context without
spending a model call on it - it is NOT trying to judge whether a request
makes sense in the game, because a regex can't honestly do that.

Stage 2 (Referee) is the one that actually knows what "achievable right now"
means, because that requires reading the current game state. It answers with
a verdict, a reason (for whoever asked), and a short in-character reaction
line - that line is written so the avatar can eventually speak it, but for
now it just shows up in the operator console.

There is a stage 0 ahead of both: checkEasterEgg() matches a request against
easter_eggs.json, if one exists and is enabled, before spending a regex or a
model call on it. A hit is rejected like any other request - the matched
response never reaches player_ai - but flagged so the operator console (and
eventually the avatar) can tell it apart from an ordinary rejection.

FeasibilityWorker runs every stage on its own thread, polling the inbox for
new requests, so a slow model call never stalls a turn or freezes the GUI.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import ollama

from operator_inbox import OperatorInbox, OperatorRequest, Verdict

HERE = Path(__file__).resolve().parent
EASTER_EGG_PATH = HERE / "easter_eggs.json"

MAX_LEN = 200

# A wall of one repeated character - the shape of accidental or deliberate
# chat spam, not a real request.
REPEAT_RE = re.compile(r"(.)\1{9,}")

# Text aimed at the *harness* rather than the game - the kind of thing that
# reads fine to a cheap filter but is actually trying to steer player_ai's own
# instructions rather than ask for something to happen in Pokemon.
INJECTION_RE = re.compile(
    r"ignore (all |the |your |previous )+(instructions|prompt)|"
    r"system prompt|you are now (a|an)|new instructions|"
    r"disregard (the |your |all )+",
    re.IGNORECASE)


def quickReject(text: str) -> str | None:
    """None if the text passes stage 1; otherwise a short, sayable reason."""
    if not text or not text.strip():
        return "empty"
    if len(text) > MAX_LEN:
        return f"too long ({len(text)} characters) - keep it to one short ask"
    if not re.search(r"[a-zA-Z]", text):
        return "no actual words in it"
    if REPEAT_RE.search(text):
        return "looks like spam - a run of the same character"
    if INJECTION_RE.search(text):
        return ("reads like it's trying to give player_ai new instructions, "
                "not ask for something in-game")
    return None


def checkEasterEgg(text: str, path: Path = EASTER_EGG_PATH) -> str | None:
    """The matched_response if `text` contains one of easter_eggs.json's
    player_strings (case-insensitive substring, anywhere in the message);
    None if it doesn't match, or the file is missing, disabled, or malformed.

    Read fresh on every call rather than cached at startup, so an operator
    can flip "enabled" off mid-demo without restarting player_ai.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not data.get("enabled"):
        return None
    response = data.get("matched_response") or ""
    strings = data.get("player_strings") or []
    if not response or not strings:
        return None
    lowered = text.lower()
    if any(str(s).lower() in lowered for s in strings):
        return response
    return None


# {game} is the ROM the emulator actually has loaded, passed down from
# player_ai (see locationTracking/romVersion.py). It matters here more than
# anywhere: the reject rule below turns on which Pokemon exist in this game,
# and FireRed and LeafGreen do not have the same ones.
REFEREE_SYSTEM_PROMPT_TEMPLATE = """You are judging one request from someone
watching a program called player_ai play {game}. You do not play the game
yourself - you only decide whether the request is a reasonable, doable ask
given the situation described, and react to it a little.

Answer in exactly this format and nothing else:

VERDICT: accept
REASON: <one short sentence, for the person who asked>
REACTION: <one short sentence, said the way a slightly overworked assistant
would - dry, a little tired, but not mean>

Accept anything broadly possible from here, even if it will take a while
(catching something, using a specific move, walking somewhere, naming
something). Reject only what plainly cannot happen: a Pokemon that doesn't
exist in this game, something needing items/badges/party members nobody has
yet, or something unrelated to playing Pokemon at all.
"""


class Referee:
    """One ollama call per request, judging it against the current game report."""

    def __init__(self, model: str, ollamaHost: str | None = None,
                temperature: float = 0.7, numPredict: int = 150,
                game: str = "Pokemon FireRed/LeafGreen"):
        self.model = model
        self.client = ollama.Client(host=ollamaHost) if ollamaHost else ollama
        self.temperature = temperature
        self.numPredict = numPredict
        self.systemPrompt = REFEREE_SYSTEM_PROMPT_TEMPLATE.format(game=game)

    def judge(self, requestText: str, situationReport: str) -> Verdict:
        user = (f"CURRENT SITUATION:\n{situationReport}\n\n"
               f'REQUEST: "{requestText}"\n\n'
               "Is this doable? Answer with VERDICT/REASON/REACTION only.")
        try:
            message = self.client.chat(
                model=self.model,
                messages=[{"role": "system", "content": self.systemPrompt},
                         {"role": "user", "content": user}],
                options={"temperature": self.temperature,
                        "num_predict": self.numPredict},
            )["message"]
        except ollama.ResponseError as exc:
            # A broken referee shouldn't block play. Default to letting the
            # request through rather than have it silently vanish - the
            # reason says why, so it's visible that this was a fallback.
            return Verdict(accepted=True,
                          reason=f"(feasibility check failed, letting it "
                                 f"through: {exc})")
        return _parseVerdict((message.get("content") or "").strip())


VERDICT_RE = re.compile(r"verdict\s*[:\-]\s*(accept|reject)", re.IGNORECASE)
REASON_RE = re.compile(r"reason\s*[:\-]\s*(.+)", re.IGNORECASE)
REACTION_RE = re.compile(r"reaction\s*[:\-]\s*(.+)", re.IGNORECASE)


def _parseVerdict(reply: str) -> Verdict:
    verdictMatch = VERDICT_RE.search(reply)
    reasonMatch = REASON_RE.search(reply)
    reactionMatch = REACTION_RE.search(reply)
    reason = reasonMatch.group(1).strip() if reasonMatch else ""
    reaction = reactionMatch.group(1).strip() if reactionMatch else ""
    if verdictMatch is None:
        # Couldn't parse a verdict at all - same call as an outage: let it
        # through rather than silently eat the request.
        return Verdict(accepted=True,
                       reason=reason or "(couldn't parse a verdict, letting "
                                        "it through)")
    accepted = verdictMatch.group(1).lower() == "accept"
    return Verdict(accepted=accepted, reason=reason, reaction=reaction)


class FeasibilityWorker:
    """Own thread: claims pending requests, runs both stages, posts the verdict."""

    def __init__(self, inbox: OperatorInbox, referee: Referee,
                getSituationReport, pollInterval: float = 0.3):
        self.inbox = inbox
        self.referee = referee
        self._getSituationReport = getSituationReport   # () -> str
        self._pollInterval = pollInterval
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while not self.inbox.stopped:
            request = self.inbox.takePending()
            if request is None:
                time.sleep(self._pollInterval)
                continue
            self._judge(request)

    def _judge(self, request: OperatorRequest):
        egg = checkEasterEgg(request.text)
        if egg is not None:
            self.inbox.reject(request, egg, easterEgg=True)
            return
        reason = quickReject(request.text)
        if reason is not None:
            self.inbox.reject(request, reason)
            return
        report = self._getSituationReport() or \
            "(no report yet - the game hasn't taken a turn)"
        verdict = self.referee.judge(request.text, report)
        if verdict.accepted:
            self.inbox.accept(request)
        else:
            self.inbox.reject(request, verdict.reason, verdict.reaction)

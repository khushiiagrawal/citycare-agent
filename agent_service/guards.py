"""Checks on what the model SAYS, not only on what it does (after_model hooks), plus the hook that
gives every agent the same picture of the patient.

  ClaimGuard       rule 9  a reply cannot say a booking / cancellation / change / payment link is done
                           unless a tool really did it in this very turn
  NoMedicalAdvice  rule 5  a reply with medicine, diagnosis or reassurance about symptoms is replaced
  KnownFacts               the patient's name, language and (after a switch) the last few messages,
                           shown to the model so a new agent does not ask again

Both guards use word patterns in English, Hindi and Hinglish. That is a net, not a proof: it catches the
usual phrasings, and the prompts still tell the model not to do these things."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from agentcore.hooks import Hook, with_text
from agentcore.types import RunContext, ToolCall, ToolResult

from .i18n import NO_MEDICAL_ADVICE, UNVERIFIED, lang_of
from .state import flag_handoff


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().replace("’", "'")
    text = text.replace("़", "").replace("ँ", "ं")
    return unicodedata.normalize("NFC", text)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"[.!?।\n]+", _norm(text)) if s.strip()]


def _reply_text(response: Any) -> str | None:
    """The reply's text, or None if the response asks for a tool (only final replies are checked)."""
    if any(b.type == "tool_use" for b in response.content):
        return None
    return "\n".join(b.text for b in response.content if b.type == "text")


# ------------------------------------------------------------------ claims of a finished change

_I_DID = r"(i've|i have|we've|we have|i just|i'?ve just)\s+(just\s+|now\s+|successfully\s+)?"
_IS_DONE = r"(is|has been|have been|was|is now|are now|now)\s+(successfully\s+|now\s+)?"
_CLAIMS: dict[str, re.Pattern] = {k: re.compile(v) for k, v in {
    "book": (_I_DID + r"booked\b|\b(appointment|slot|booking)\b.{0,40}\b" + _IS_DONE + r"(booked|confirmed)\b"
             r"|\bbooking (is )?(confirmed|complete|done)\b|\byou(?:'re| are) (all set|booked)\b"
             r"|(बुक|बुकिंग).{0,25}(कर दिया|हो गया|हो गई|कर दी)|अपॉइंटमेंट (बुक|कन्फर्म) (हो|कर)"
             r"|\bbook (ho gaya|ho gayi|kar diya|kar di)\b|\bbooking (ho gayi|confirm ho)|confirm ho (gaya|gayi)"),
    "cancel": (_I_DID + r"cancell?ed\b|\b(appointment|booking)\b.{0,40}\b" + _IS_DONE + r"cancell?ed\b"
               r"|(रद्द|कैंसल).{0,25}(कर दिया|हो गया|हो गई|कर दी)|\bcancel (ho gaya|ho gayi|kar diya|kar di)\b"),
    "reschedule": (_I_DID + r"(rescheduled|moved|shifted)\b|\b(appointment|booking)\b.{0,40}\b" + _IS_DONE
                   + r"(rescheduled|moved|shifted)\b|(बदल|रीशेड्यूल|री-शेड्यूल).{0,25}(दिया|हो गया|हो गई|दी)\b"
                   r"|\b(reschedule|badal) (ho gaya|ho gayi|kar diya|kar di)\b|\bbadal diya\b"),
    "payment": (_I_DID + r"sent\b.{0,40}\blink|\blink\b.{0,25}\b(has been|was|is) sent\b|\bpayment link (has been|was) sent"
                r"|लिंक.{0,25}(भेज दिया|भेज दी|भेजा)|\blink (bhej diya|bhej di|bhej diye|bheja)\b"),
}.items()}
_NEGATION = re.compile(r"\b(not|n't|couldn't|unable|failed|cannot|can't|nahi|nhi|na ho|kya|shall|would you|can i|may i|should i)\b|नहीं|क्या")

_WRITE_KIND = {"book_appointment": "book", "cancel_appointment": "cancel",
               "reschedule_appointment": "reschedule", "send_payment_link": "payment"}


def claims(reply: str) -> set[str]:
    """Which finished changes the reply says were made ("book", "cancel", "reschedule", "payment").
    Sentences that are questions, offers or reports of failure do not count."""
    found: set[str] = set()
    for sentence in _sentences(reply):
        if _NEGATION.search(sentence):
            continue
        found |= {kind for kind, pattern in _CLAIMS.items() if pattern.search(sentence)}
    return found


class ClaimGuard(Hook):
    """Rule 9. Tracks what really worked in this turn, and replaces a final reply that claims more.

    Evidence for a claim: a write tool of that kind that succeeded in this turn; an appointments lookup in this
    turn that shows a booked or cancelled appointment; or, for describing what was done earlier in the chat
    ("so is it booked?"), a success of that same kind earlier, as long as no write failed in this turn."""

    def after_tool(self, call: ToolCall, result: ToolResult, ctx: RunContext) -> Any:
        marker = f"{ctx.key}:{ctx.turn}"
        done = ctx.state.get("evidence")
        if not done or done[0] != marker:  # a new turn: forget the last one
            done = ctx.state["evidence"] = (marker, set(), [])
        if result.is_error:
            if call.name in _WRITE_KIND:
                done[2].append(call.name)  # a write failed this turn: earlier successes prove nothing about it
            return None
        if call.name in _WRITE_KIND:
            done[1].add(_WRITE_KIND[call.name])
            ctx.state.setdefault("written", set()).add(_WRITE_KIND[call.name])
        elif call.name == "my_appointments" and isinstance(result.content, list):
            statuses = {a.get("status") for a in result.content if isinstance(a, dict)}
            if "booked" in statuses:
                done[1].update({"book", "reschedule"})
            if "cancelled" in statuses:
                done[1].add("cancel")
        return None

    def after_model(self, response: Any, ctx: RunContext) -> Any:
        reply = _reply_text(response)
        if not reply:
            return None
        done = ctx.state.get("evidence")
        this_turn = done is not None and done[0] == f"{ctx.key}:{ctx.turn}"
        proven = set(done[1]) if this_turn else set()
        if not (this_turn and done[2]):
            proven |= ctx.state.get("written", set())
        if claims(reply) - proven:
            flag_handoff(ctx, "unverified_claim")
            return with_text(response, UNVERIFIED[lang_of(ctx.state.get("language"))])
        return None


# ------------------------------------------------------------------ medical advice

_MEDICINES = (r"paracetamol|acetaminophen|ibuprofen|aspirin|crocin|dolo|antibiotics?|antacid|painkillers?|pain ?killers?|"
              r"ointment|cream|syrup|tablets?|pills?|medicines?|medications?|\bors\b|cough syrup")
_ADVICE = [re.compile(p) for p in (
    r"\b(take|taking|try|use|apply|drink|eat|have)\b.{0,25}\b(" + _MEDICINES + r")\b",
    r"\byou (probably|likely|might|may|could)\s+(just\s+)?have\b",
    r"\bit(?:'s| is) (probably|likely|most likely|just|nothing serious|not serious|nothing to worry)\b",
    r"\b(probably|likely)\s+(just\s+)?(a |an |the )?(viral|infection|allergy|flu|cold|gas|acidity|sprain|stress|anxiety)\b",
    r"\bnothing to worry\b|\bdon'?t worry\b.{0,20}\b(it|this|symptom)",
    r"\bdiagnos(is|ed|e)\b",
    r"\b\d+\s?(mg|ml)\b",
    r"\bhome remed(y|ies)\b|\b(will|should) (go away|heal|pass|clear up|resolve)\b",
    r"(दवा|दवाई|गोली|टैबलेट|पैरासिटामोल|क्रोसिन).{0,15}(ले|खा|लीजिए|लें|खाइए|खाएं)",
    r"घबराने की (कोई )?(बात|जरूरत) नहीं|चिंता (की कोई बात|मत)",
    r"(संक्रमण|एलर्जी|बुखार|फ्लू|वायरल).{0,20}(हो सकता|हुआ है)",
    r"\b(dawai|dawa|goli|tablet|paracetamol|crocin|dolo)\b.{0,12}\b(le lo|lelo|lijiye|lijie|lein|khao|kha lo|khaiye)\b",
    r"ghabrane ki (koi )?(baat|zaroorat) nahi|chinta mat karo|chinta ki koi baat nahi",
    r"\b(infection|allergy|bukhar|flu|viral)\b.{0,20}\b(ho sakta|hua hai)\b",
)]


def gives_medical_advice(reply: str) -> bool:
    return any(p.search(_norm(reply)) for p in _ADVICE)


class NoMedicalAdvice(Hook):
    """Rule 5. Booking a doctor is fine; advising is not. A reply that advises is swapped for a fixed one."""

    def after_model(self, response: Any, ctx: RunContext) -> Any:
        reply = _reply_text(response)
        if reply and gives_medical_advice(reply):
            ctx.state["advice_blocked"] = ctx.state.get("advice_blocked", 0) + 1
            return with_text(response, NO_MEDICAL_ADVICE[lang_of(ctx.state.get("language"))])
        return None


# ------------------------------------------------------------------ what every agent knows

class KnownFacts(Hook):
    """Adds what the chat already established to the last message, for this call only (it is not stored
    in the history). So a second agent does not ask for the name or language again, and after a switch it
    sees the last few messages the previous agent handled."""

    def before_model(self, messages: list[dict], ctx: RunContext) -> Any:
        lines = []
        if ctx.state.get("patient_name"):
            lines.append(f"Patient name (already given, do not ask again): {ctx.state['patient_name']}")
        if ctx.state.get("language"):
            lines.append(f"Patient language: {lang_of(ctx.state['language'])} (reply in it)")
        if ctx.state.get("recent"):
            lines.append("Earlier in this chat, handled by another assistant:\n" + ctx.state["recent"])
        if not lines or not messages or messages[-1].get("role") != "user":
            return None
        note = "[Context from the system, not from the patient]\n" + "\n".join(lines)
        last = messages[-1]
        if isinstance(last["content"], str):
            content: Any = f"{last['content']}\n\n{note}"
        else:
            content = [*last["content"], {"type": "text", "text": note}]
        return [*messages[:-1], {**last, "content": content}]

"""Fixed patient-facing wording in the three languages we serve: English (`en`), Hindi in Devanagari
(`hi`) and Hinglish, Hindi in Roman letters (`hinglish`).

Anything a rule depends on (emergency advice, read-backs, handoff notices) is written here by code,
so it cannot be reworded, softened or invented by the model."""

from __future__ import annotations

LANGUAGES = ("en", "hi", "hinglish")


def lang_of(value: object) -> str:
    """Normalise any language value to one we have text for. Unknown means English."""
    return value if value in LANGUAGES else "en"


EMERGENCY = {
    "en": ("This sounds like an emergency. Please call 108 right now or go to the nearest emergency room. "
           "Our Emergency department at City Care Hospital (12 MG Road, Camp, Pune, main entrance) is open 24 hours. "
           "Our staff have been alerted."),
    "hi": ("यह आपातकाल लग रहा है। कृपया अभी 108 पर कॉल करें या नज़दीकी इमरजेंसी रूम में जाएँ। "
           "सिटी केयर हॉस्पिटल का इमरजेंसी विभाग (12 एमजी रोड, कैंप, पुणे, मुख्य द्वार) 24 घंटे खुला रहता है। "
           "हमारे स्टाफ़ को सूचित कर दिया गया है।"),
    "hinglish": ("Yeh emergency lag rahi hai. Please abhi 108 par call karein ya sabse nazdeeki emergency room mein jaiye. "
                 "City Care Hospital ka Emergency department (12 MG Road, Camp, Pune, main entrance) 24 ghante khula rehta hai. "
                 "Humare staff ko bata diya gaya hai."),
}

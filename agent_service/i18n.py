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


# ------------------------------------------------------------------ dates and read-backs

_WEEKDAYS_HI = ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"]
_MONTHS_HI = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्तूबर", "नवंबर", "दिसंबर"]
_MONTHS_LONG = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                "November", "December"]
_PERIODS = {  # (first hour, name) for hi and hinglish
    "hi": [(0, "रात"), (5, "सुबह"), (12, "दोपहर"), (16, "शाम"), (20, "रात")],
    "hinglish": [(0, "raat"), (5, "subah"), (12, "dopahar"), (16, "shaam"), (20, "raat")],
}


def _period(language: str, hour: int) -> str:
    return [name for start, name in _PERIODS[language] if hour >= start][-1]


def slot_text(moment, language: str) -> str:
    """A date-time (already in hospital time) as the patient reads it in their language."""
    language = lang_of(language)
    weekday, hour12 = moment.strftime("%A"), moment.hour % 12 or 12
    clock = f"{hour12}:{moment.minute:02d}"
    if language == "hi":
        return (f"{_WEEKDAYS_HI[moment.weekday()]}, {moment.day} {_MONTHS_HI[moment.month - 1]} {moment.year}, "
                f"{_period('hi', moment.hour)} {clock} बजे")
    if language == "hinglish":
        return (f"{weekday}, {moment.day} {_MONTHS_LONG[moment.month - 1]} {moment.year}, "
                f"{_period('hinglish', moment.hour)} {clock} baje")
    return f"{weekday}, {moment.day} {moment:%b %Y} at {clock} {moment:%p}"


FOOTER = {
    "en": "Reply YES to confirm (haan / हाँ), or tell me what to change.",
    "hi": "पुष्टि के लिए 'हाँ' लिखें, या बताइए कि क्या बदलना है।",
    "hinglish": "Confirm karne ke liye 'haan' likhein, ya bataiye kya badalna hai.",
}

READBACK = {  # {footer} is added by the caller
    "book": {
        "en": "Please confirm: book {doctor} on {slot} for {name}?",
        "hi": "कृपया पुष्टि करें: {name} के लिए {doctor} के साथ {slot} पर अपॉइंटमेंट बुक करें?",
        "hinglish": "Please confirm karein: {name} ke liye {doctor} ke saath {slot} par appointment book karein?",
    },
    "reschedule": {
        "en": "Please confirm: move your appointment with {doctor} from {old} to {new}?",
        "hi": "कृपया पुष्टि करें: {doctor} के साथ आपका अपॉइंटमेंट {old} से बदलकर {new} पर करें?",
        "hinglish": "Please confirm karein: {doctor} ke saath aapka appointment {old} se badalkar {new} par karein?",
    },
    "cancel": {
        "en": "Please confirm: cancel your appointment with {doctor} on {slot}?",
        "hi": "कृपया पुष्टि करें: {doctor} के साथ {slot} का आपका अपॉइंटमेंट रद्द करें?",
        "hinglish": "Please confirm karein: {doctor} ke saath {slot} wala aapka appointment cancel karein?",
    },
    "payment": {
        "en": "Please confirm: send a payment link for bill {bill} (Rs {amount}, {description})?",
        "hi": "कृपया पुष्टि करें: बिल {bill} (Rs {amount}, {description}) के लिए पेमेंट लिंक भेजें?",
        "hinglish": "Please confirm karein: bill {bill} (Rs {amount}, {description}) ke liye payment link bhejein?",
    },
}


def readback(kind: str, language: object, **facts: object) -> str:
    """One confirmation question: fixed wording for the language, facts filled in by the caller."""
    language = lang_of(language)
    return f"{READBACK[kind][language].format(**facts)} {FOOTER[language]}"

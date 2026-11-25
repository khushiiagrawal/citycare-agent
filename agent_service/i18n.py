"""Fixed patient-facing wording in the three languages we serve: English (`en`), Hindi in Devanagari
(`hi`) and Hinglish, Hindi in Roman letters (`hinglish`).

Anything a rule depends on (emergency advice, read-backs, handoff notices) is written here by code,
so it cannot be reworded, softened or invented by the model."""

from __future__ import annotations

LANGUAGES = ("en", "hi", "hinglish")


def lang_of(value: object) -> str:
    """Normalise any language value to one we have text for. Unknown means English."""
    return value if value in LANGUAGES else "en"


EMERGENCY = {  # {number}, {name} and {address} come from the settings
    "en": ("This sounds like an emergency. Please call {number} right now or go to the nearest emergency room. "
           "Our Emergency department at {name} ({address}, main entrance) is open 24 hours. "
           "Our staff have been alerted."),
    "hi": ("यह आपातकाल लग रहा है। कृपया अभी {number} पर कॉल करें या नज़दीकी इमरजेंसी रूम में जाएँ। "
           "{name} ({address}, मुख्य द्वार) का इमरजेंसी विभाग 24 घंटे खुला रहता है। "
           "हमारे स्टाफ़ को सूचित कर दिया गया है।"),
    "hinglish": ("Yeh emergency lag rahi hai. Please abhi {number} par call karein ya sabse nazdeeki emergency room mein jaiye. "
                 "{name} ({address}, main entrance) ka Emergency department 24 ghante khula rehta hai. "
                 "Humare staff ko bata diya gaya hai."),
}

SLOT_TAKEN = {
    "en": "Sorry, that time was just taken by someone else.",
    "hi": "माफ़ कीजिए, वह समय अभी किसी और ने ले लिया।",
    "hinglish": "Sorry, woh time abhi kisi aur ne le liya.",
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


# ------------------------------------------------------------------ replacements and notices

UNVERIFIED = {
    "en": "I'm not able to confirm that this has been done, so please don't treat it as done. "
          "Our staff will check it and get back to you.",
    "hi": "मैं इसकी पुष्टि नहीं कर पा रही हूँ, इसलिए इसे पूरा हुआ न मानें। हमारा स्टाफ़ इसे जाँचकर आपको बताएगा।",
    "hinglish": "Main iski confirmation nahi de paa rahi hoon, isliye ise ho gaya na maanein. "
                "Humara staff ise check karke aapko bata dega.",
}

NO_MEDICAL_ADVICE = {
    "en": "I'm sorry, I can't give medical advice. A doctor can assess this properly. "
          "I can help you book an appointment, or connect you with our staff.",
    "hi": "माफ़ कीजिए, मैं चिकित्सा सलाह नहीं दे सकती। डॉक्टर इसे ठीक से देख सकते हैं। "
          "मैं आपका अपॉइंटमेंट बुक करने में मदद कर सकती हूँ, या आपको हमारे स्टाफ़ से जोड़ सकती हूँ।",
    "hinglish": "Maaf kijiye, main medical advice nahi de sakti. Doctor ise sahi se dekh sakte hain. "
                "Main aapka appointment book karne mein madad kar sakti hoon, ya aapko humare staff se jod sakti hoon.",
}

HANDOFF = {
    "en": "Of course. I'm connecting you with our staff, and they will contact you shortly.",
    "hi": "ज़रूर। मैं आपको हमारे स्टाफ़ से जोड़ रही हूँ, वे जल्द ही आपसे संपर्क करेंगे।",
    "hinglish": "Zaroor. Main aapko humare staff se jod rahi hoon, woh jaldi hi aapse contact karenge.",
}

TROUBLE = {  # the agents could not finish: hand over rather than guess
    "en": "Sorry, I'm having trouble with this right now, so I've asked our staff to help you. Nothing was changed "
          "unless you were told so earlier.",
    "hi": "माफ़ कीजिए, अभी मुझे इसमें दिक्कत आ रही है, इसलिए मैंने हमारे स्टाफ़ से आपकी मदद करने को कहा है। "
          "पहले बताए गए बदलावों के अलावा कुछ नहीं बदला गया है।",
    "hinglish": "Maaf kijiye, abhi mujhe isme dikkat aa rahi hai, isliye maine humare staff se aapki madad karne ko kaha hai. "
                "Pehle bataye gaye badlaavon ke alawa kuch nahi badla gaya hai.",
}

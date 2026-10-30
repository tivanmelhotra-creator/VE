"""
toonface.kit.g2p_fa — universal rule-based Persian grapheme → phoneme converter for lip-sync (no external deps).

Phoneme alphabet (lip-sync oriented):
    consonants  b p m f v t d n s z zh sh ch j k g q x h r l y '        (' = ع / ء glottal)
    vowels      a e o (short)   A i u (long)   ow ey (diphthongs)

Handles:
  * explicit diacritics  َ ِ ُ  (a e o),  ّ tashdid (gemination),  ً tanvin (an),  ٔ / ۀ (ezafe ye)
  * long vowels  ا آ و ی  with position rules (initial ا+ی → i, او → u, خوا → xA, و/ی as consonant or vowel)
  * unwritten short vowels: syllabification CV(C)(C) — every onset consonant without a following vowel gets a
    short vowel inserted (default a, e before final ه / in prefixes, o in a few patterns)
  * silent final ه (→ e), ه‌ی / ۀ / ی after a vowel-final word  → kasra-ezafe "ye"
  * kasra-ezafe flag  g2p(word, ezafe=True)  or a trailing kasra  ِ  on the word
  * ZWNJ-separated affixes: می / نمی / ها / های / ام / ات / اش / ای / اند / تر / ترین / ی …
  * colloquial forms in a lexicon (میخوام، دیگه، اصلاً، بهت …) — lexicon wins, rules generalise

    from .g2p_fa import g2p, sentence_g2p
    g2p("حوصله‌ی")            → ['h','o','w','s','e','l','e','y','e']
    sentence_g2p("ببین، من …") → [(word, [phonemes]), …]
"""
from __future__ import annotations

import re

VOWELS = {"a", "e", "o", "A", "i", "u", "ow", "ey"}
SHORT = {"a", "e", "o"}
ZWNJ = "\u200c"
FATHA, KASRA, DAMMA, SUKUN, TASHDID, TANVIN_AN = "\u064e", "\u0650", "\u064f", "\u0652", "\u0651", "\u064b"
HAMZA_ABOVE = "\u0654"

CONS = {"ب": "b", "پ": "p", "ت": "t", "ث": "s", "ج": "j", "چ": "ch", "ح": "h", "خ": "x", "د": "d", "ذ": "z",
        "ر": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "ع": "'",
        "غ": "q", "ف": "f", "ق": "q", "ک": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n", "ه": "h", "ء": "'",
        "ئ": "'", "أ": "'", "ؤ": "'"}

# frequent words — standard + colloquial (Tehrani).  Space-separated phonemes.
LEXICON = {
    # pronouns / function words
    "من": "m a n", "تو": "t o", "او": "u", "اون": "u n", "ما": "m A", "شما": "sh o m A", "اونا": "u n A",
    "ایشون": "i sh u n", "آنها": "A n h A", "این": "i n", "آن": "A n", "اینا": "i n A", "که": "k e", "چه": "ch e",
    "را": "r A", "رو": "r o", "به": "b e", "با": "b A", "از": "a z", "در": "d a r", "تا": "t A", "هم": "h a m",
    "و": "o", "یا": "y A", "اگر": "a g a r", "اگه": "a g e", "ولی": "v a l i", "اما": "a m m A", "چون": "ch o n",
    "پس": "p a s", "هر": "h a r", "همه": "h a m e", "هیچ": "h i ch", "یه": "y e", "یک": "y e k", "بر": "b a r",
    "برای": "b a r A y e", "واسه": "v A s e", "مثل": "m e s l e", "خیلی": "x e y l i", "کمی": "k a m i",
    "دیگه": "d i g e", "دیگر": "d i g a r", "حالا": "h A l A", "الان": "a l A n", "اینجا": "i n j A",
    "اونجا": "u n j A", "کجا": "k o j A", "چرا": "ch e r A", "چطور": "ch e t o w r", "چی": "ch i", "کی": "k i",
    "کدوم": "k o d u m", "چند": "ch a n d", "وقتی": "v a q t i", "باز": "b A z", "فقط": "f a q a t",
    "شاید": "sh A y a d", "حتما": "h a t m a n", "حتماً": "h a t m a n", "اصلا": "a s l a n", "اصلاً": "a s l a n",
    "واقعا": "v A q e a n", "واقعاً": "v A q e a n", "بله": "b a l e", "آره": "A r e", "نه": "n a", "نخیر": "n a x e y r",
    "باشه": "b A sh e", "خب": "x o b", "خوب": "x u b", "بد": "b a d", "سلام": "s a l A m", "مرسی": "m e r s i",
    "ممنون": "m a m n u n", "ممنونم": "m a m n u n a m", "خداحافظ": "x o d A h A f e z", "لطفا": "l o t f a n",
    "لطفاً": "l o t f a n",
    # verbs (frequent forms)
    "است": "a s t", "هست": "h a s t", "نیست": "n i s t", "بود": "b u d", "نبود": "n a b u d", "شد": "sh o d",
    "شده": "sh o d e", "میشه": "m i sh e", "نمیشه": "n a m i sh e", "کرد": "k a r d", "کن": "k o n", "بکن": "b o k o n",
    "نکن": "n a k o n", "برو": "b o r o", "بیا": "b i y A", "ببین": "b e b i n", "بگو": "b e g u", "بشین": "b e sh i n",
    "گفتم": "g o f t a m", "گفتی": "g o f t i", "گفت": "g o f t", "میگم": "m i g a m", "میگی": "m i g i",
    "میگه": "m i g e", "بهت": "b e h e t", "بهش": "b e h e sh", "بهم": "b e h e m", "میخوام": "m i x A m",
    "میخوای": "m i x A y", "نمیخوام": "n a m i x A m", "خواستم": "x A s t a m", "دارم": "d A r a m",
    "داری": "d A r i", "داره": "d A r e", "ندارم": "n a d A r a m", "نداره": "n a d A r e", "دارن": "d A r a n",
    "میدونم": "m i d u n a m", "نمیدونم": "n a m i d u n a m", "میدونی": "m i d u n i", "میرم": "m i r a m",
    "رفتم": "r a f t a m", "رفت": "r a f t", "اومدم": "u m a d a m", "اومد": "u m a d", "بخور": "b o x o r",
    "خوردم": "x o r d a m", "دیدم": "d i d a m", "دیدی": "d i d i", "بذار": "b e z A r", "شنیدی": "sh e n i d i",
    # nouns / adjectives (frequent)
    "حوصله": "h o w s e l e", "بازی": "b A z i", "بازیا": "b A z i y A", "خونه": "x u n e", "مدرسه": "m a d r e s e",
    "کتاب": "k e t A b", "آب": "A b", "غذا": "q a z A", "خسته": "x a s t e", "خواب": "x A b", "بیدار": "b i d A r",
    "دوست": "d u s t", "دلم": "d e l a m", "دل": "d e l", "مامان": "m A m A n", "بابا": "b A b A", "بچه": "b a c h e",
    "روز": "r u z", "شب": "sh a b", "امروز": "e m r u z", "فردا": "f a r d A", "دیروز": "d i r u z", "کار": "k A r",
    "چیز": "ch i z", "چیزی": "ch i z i", "آدم": "A d a m", "دنیا": "d o n y A", "خدا": "x o d A", "زندگی": "z e n d e g i",
    "مهم": "m o h e m", "عزیزم": "a z i z a m", "عزیز": "a z i z", "قشنگ": "q a sh a n g", "خوشگل": "x o sh g e l",
    "بزرگ": "b o z o r g", "کوچیک": "k u ch i k", "سخت": "s a x t", "راحت": "r A h a t", "زود": "z u d",
    "دیر": "d i r", "باید": "b A y a d", "نباید": "n a b A y a d", "همیشه": "h a m i sh e", "هنوز": "h a n u z",
    "خودم": "x o d a m", "خودت": "x o d e t", "خودش": "x o d e sh", "فکر": "f e k r", "حرف": "h a r f",
}

SUFFIXES = {"ها": "h A", "هایی": "h A y i", "های": "h A y e", "هایم": "h A y a m", "هایت": "h A y a t",
            "هایش": "h A y a sh", "ام": "a m", "ات": "a t", "اش": "a sh", "ای": "i", "ایم": "i m", "اید": "i d",
            "اند": "a n d", "تر": "t a r", "ترین": "t a r i n", "ی": "y e", "یی": "y i", "مان": "m A n",
            "تان": "t A n", "شان": "sh A n", "رو": "r o", "و": "o"}
PREFIXES = {"می": "m i", "نمی": "n a m i", "بی": "b i", "هم": "h a m"}


def normalize(s: str) -> str:
    s = s.replace("ي", "ی").replace("ك", "ک").replace("ى", "ی").replace("ۀ", "ه" + ZWNJ + "ی")
    s = s.replace("هٔ", "ه" + ZWNJ + "ی").replace("\u0640", "")             # ezafe hamza, tatweel
    s = re.sub(r"[\u200f\u200e]", "", s)
    s = re.sub(r"[،؛,.!؟?…\"«»:;()\[\]\-–—]", " ", s)
    return s


def _letters_to_tokens(w: str):
    """→ list of (kind, ph): kind 'C' consonant, 'V' explicit/long vowel; final silent ه → V 'e'."""
    toks = []
    n = len(w)
    i = 0
    while i < n:
        ch = w[i]
        nxt = w[i + 1] if i + 1 < n else ""
        prev_letter = toks[-1] if toks else None
        if ch == FATHA:
            toks.append(("V", "a"))
        elif ch == KASRA:
            toks.append(("V", "e"))
        elif ch == DAMMA:
            toks.append(("V", "o"))
        elif ch == TASHDID:
            if toks and toks[-1][0] == "C":
                toks.append(("G", toks[-1][1]))              # geminate: first copy = coda, second = onset
        elif ch == TANVIN_AN:
            if toks and toks[-1] == ("V", "A"):
                toks.append(("C", "n"))
            else:
                toks += [("V", "a"), ("C", "n")]
        elif ch in (SUKUN, HAMZA_ABOVE, "\u0670"):
            pass
        elif ch == "آ":
            toks.append(("V", "A"))
        elif ch in "اإ":
            if i == 0:
                if nxt in "یي":
                    toks.append(("V", "i"))
                    i += 1
                elif nxt == "و":
                    toks.append(("V", "u"))
                    i += 1
                elif nxt == KASRA:
                    toks.append(("V", "e"))
                    i += 1
                elif nxt == DAMMA:
                    toks.append(("V", "o"))
                    i += 1
                else:
                    toks.append(("V", "a"))                 # word-initial alef = glottal + short vowel
            else:
                toks.append(("V", "A"))
        elif ch == "و":
            prevV = prev_letter is not None and prev_letter[0] == "V"
            nextV = nxt in "اآیی" or nxt in (FATHA, KASRA, DAMMA)
            if i == 0:
                toks.append(("C", "v"))
            elif prev_letter == ("C", "x") and nxt == "ا":
                pass                                          # خوا → xA (silent و)
            elif prevV:
                toks.append(("C", "v"))
            elif nextV or nxt == TASHDID:
                toks.append(("C", "v"))
            elif i == n - 1 and n == 2 and w[0] in CONS:
                toks.append(("V", "o"))                       # تو/دو/رو → o
            else:
                toks.append(("V", "u"))
        elif ch in "یی":
            prevV = prev_letter is not None and prev_letter[0] == "V"
            nextV = nxt in "اآو" or nxt in (FATHA, KASRA, DAMMA)
            if i == 0 or prevV:
                toks.append(("C", "y"))
            elif nextV and i < n - 1:
                toks += [("V", "i"), ("C", "y")]              # بیا → b i y A
            else:
                toks.append(("V", "i"))
        elif ch == "ه":
            if i == n - 1 and i > 0 and (prev_letter is None or prev_letter[0] == "C" or prev_letter == ("V", "a")):
                toks.append(("V", "e"))                       # silent final ه
            else:
                toks.append(("C", "h"))
        elif ch in CONS:
            toks.append(("C", CONS[ch]))
        i += 1
    return toks


def _syllabify(toks, short="a"):
    """insert unwritten short vowels: every consonant that must be an onset and has no vowel gets one."""
    out = []
    state = "onset"                                           # onset | coda0 | coda1
    n = len(toks)
    for i, (k, p) in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < n else None
        if k == "V":
            if out and out[-1] in VOWELS and p in VOWELS and out[-1] != "'":
                out.append("y" if out[-1] in ("A", "u", "i") and p not in ("A",) else "'")   # hiatus glide
            out.append(p)
            state = "coda0"
            continue
        if k == "G":
            out.append(p)
            if nxt is None or nxt[0] != "V":
                out.append(_pick_short(toks, i, short))
            state = "coda0"
            continue
        if nxt is not None and nxt[0] == "G" and (not out or out[-1] not in VOWELS):
            out.append(_pick_short(toks, i, short))
            out.append(p)
            state = "coda0"
            continue
        if state == "onset":
            out.append(p)
            if nxt is None or nxt[0] != "V":
                out.append(_pick_short(toks, i, short))
                state = "coda0"
            continue
        if state == "coda0":
            out.append(p)
            state = "onset_ready" if (nxt is not None and nxt[0] == "V") else "coda1"
            continue
        if state == "coda1":
            if nxt is None and len(out) >= 2 and out[-2] in ("A", "i", "u") and (out[-1] + p) not in OK_CC_AFTER_LONG:
                out.append(short)                             # خواهر → x A h a r (no CVVCC unless a legal cluster)
                out.append(p)
                state = "coda0"
            elif nxt is None or (nxt is not None and nxt[0] == "V"):
                out.append(p)                                 # CC coda at the end, or onset of the next vowel
                state = "coda1" if nxt is None else "onset_ready"
            else:
                out.append(p)
                out.append(_pick_short(toks, i, short))
                state = "coda0"
            continue
        if state == "onset_ready":                            # previous C is onset and a vowel follows → never here
            out.append(p)
    return out


OK_CC_AFTER_LONG = {"st", "sht", "nd", "rd", "ft", "xt", "zd", "rf", "rz", "ns", "sh", "rs", "rt", "nt", "rm"}


def _pick_short(toks, i, default):
    nxt = toks[i + 1][1] if i + 1 < len(toks) else ""
    nn = toks[i + 2] if i + 2 < len(toks) else None
    if nxt in ("y",) or (nn == ("V", "e") and i + 2 == len(toks) - 1):
        return "e"
    return default


def _rules(w: str, short="a"):
    return _syllabify(_letters_to_tokens(w), short)


PERSON = (("یم", ["i", "m"]), ("ید", ["i", "d"]), ("ند", ["a", "n", "d"]), ("ین", ["i", "n"]), ("ن", ["a", "n"]),
          ("م", ["a", "m"]), ("ی", ["i"]), ("د", ["a", "d"]), ("ه", ["e"]))


def _verb_stem(rest: str):
    """present/past stem + person ending; و between consonants in a verb stem is usually o (خور، رو، شو)."""
    for suf, ph in PERSON:
        if rest.endswith(suf) and len(rest) - len(suf) >= 1:
            stem = rest[:-len(suf)]
            sp = _rules(stem)
            sp = ["o" if (x == "u" and 0 < j < len(sp) - 1 + (1 if stem.endswith("و") else 0)) else x
                  for j, x in enumerate(sp)]
            if sp and sp[-1] in SHORT and stem[-1] not in "ه":
                sp = sp[:-1]                                  # drop the inserted vowel before the ending
            if sp and sp[-1] in VOWELS and ph[0] in VOWELS:
                sp.append("y")
            return sp + ph
    return _rules(rest)


def _part(p: str, first: bool):
    if not p:
        return []
    if p in LEXICON:
        return LEXICON[p].split()
    if not first and p in SUFFIXES:
        return SUFFIXES[p].split()
    if first and p in PREFIXES:
        return PREFIXES[p].split()
    # colloquial verb prefix written joined: می/نمی + stem (≥ 2 letters)
    for pre in ("نمی", "می"):
        if first and p.startswith(pre) and len(p) >= len(pre) + 2:
            rest = p[len(pre):]
            if rest in LEXICON:
                return PREFIXES[pre].split() + LEXICON[rest].split()
            return PREFIXES[pre].split() + _verb_stem(rest)
    return _rules(p)


def g2p(word: str, ezafe: bool = False) -> list:
    """one written word → phoneme list.  ezafe=True appends the kasra-ezafe (e / ye after a vowel)."""
    w = normalize(word).strip()
    if not w:
        return []
    if w.endswith(KASRA):
        ezafe = True
        w = w[:-1]
    w0 = re.sub(r"[\u064b-\u0652]", "", w) if w.replace(ZWNJ, "") in LEXICON or re.sub(r"[\u064b-\u0652]", "", w) in LEXICON else w
    key = w0.replace(ZWNJ, "")
    if key in LEXICON and ZWNJ not in w0:
        ph = LEXICON[key].split()
    else:
        parts = w0.split(ZWNJ)
        ph = []
        for j, p in enumerate(parts):
            if j > 0 and p == "ی" and ph and ph[-1] in VOWELS:
                ph += ["y", "e"]                             # ه‌ی / ا‌ی → ezafe ye
                continue
            q = _part(p, j == 0)
            if ph and q and ph[-1] in VOWELS and q[0] in VOWELS:
                ph.append("y")                               # مدرسه‌ای → … e y i
            ph += q
    if ezafe:
        ph += ["y", "e"] if ph and ph[-1] in VOWELS else ["e"]
    return ph


def sentence_g2p(text: str, ezafe_words=()):
    """→ [(word, phonemes)].  ezafe_words: indices (or words) that carry an unwritten kasra-ezafe."""
    out = []
    for i, w in enumerate(normalize(text).split()):
        out.append((w, g2p(w, ezafe=(i in ezafe_words or w in ezafe_words))))
    return out

"""Tokenisation shared by indexing and querying."""

import re

# Question framing words carry no topical signal and, in a small corpus,
# otherwise match nearly every document ("did people find ... ?").
STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be because been
    before being below between both but by can could did do does doing don done
    down during each else ever every few for from further get got had has have
    having he her here hers him his how i if in into is it its itself just me
    more most much my no nor not now of off on once only or other our ours out
    over own really same she should so some such than that the their them then
    there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours
    s t ll ve re d m
    anyone anything everyone people person user users participant participants
    customer customers research study studies interview interviews think thought
    feel felt say said says tell told find found mention mentioned experience
    experienced many often number percent percentage
    """.split()
)

# (suffix, replacement); first match wins. "-ation" keeps its "at" so that
# "frustration" and "frustrating" both reduce to "frustrat".
_SUFFIXES = (
    ("ations", "at"), ("ation", "at"), ("ingly", ""), ("ings", ""), ("ing", ""),
    ("ments", ""), ("ment", ""), ("ances", ""), ("ance", ""), ("ness", ""),
    ("ions", ""), ("ion", ""), ("edly", ""), ("ed", ""), ("ly", ""), ("s", ""),
)


def stem(word: str) -> str:
    """Crude suffix stripping so "frustrating"/"frustration"/"frustrated" match."""
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    else:
        for suffix, replacement in _SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                if suffix == "s" and word.endswith("ss"):
                    break
                word = word[: -len(suffix)] + replacement
                break
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords, stem."""
    return [
        stem(token)
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS
    ]

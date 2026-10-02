"""Text normalisation for names and addresses (offline, rule-based; no external lookups)."""
import re
from unidecode import unidecode

# ---- abbreviations -----------------------------------------------------------------
NAME_ABBR = {
    "pvt": "private", "ltd": "limited", "corp": "corporation", "inc": "incorporated",
    "co": "company", "intl": "international", "mfg": "manufacturing", "assoc": "associates",
    "svcs": "services", "svc": "service", "bros": "brothers", "dept": "department",
}
# Tokens that carry no identity information in a business name.
LEGAL_DROP = {
    "private", "limited", "incorporated", "corporation", "company", "llc", "llp", "pc", "pllc",
    "lp", "plc", "and", "the", "com", "www", "net", "org", "in", "sarl", "sas", "sa", "eurl",
    "gmbh", "de", "du", "la", "le", "les", "des", "et", "l", "d",
    # French legal forms (generic domain knowledge; the test set adds France, the training data has none)
    "sasu", "sci", "snc", "scop", "sca", "scs", "selarl", "sarlu", "ei", "eirl", "sasp", "ste",
}
ADDR_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "dr": "drive",
    "ln": "lane", "ct": "court", "blvd": "boulevard", "hwy": "highway", "pkwy": "parkway",
    "pl": "place", "cir": "circle", "trl": "trail", "ter": "terrace", "apt": "apartment",
    "ste": "suite", "fl": "floor", "bldg": "building", "nr": "near", "opp": "opposite",
    "mt": "mount", "ft": "fort", "cres": "crescent", "sq": "square", "expy": "expressway",
    "aly": "alley", "ctr": "center", "cntr": "center",
    # French street-type abbreviations ('R. De Feltre' == 'Rue de Feltre', 'ALL' == 'Allee'); applied to every record
    "r": "rue", "bd": "boulevard", "bvd": "boulevard", "chem": "chemin", "imp": "impasse", "rte": "route",
    "all": "allee", "fg": "faubourg", "fbg": "faubourg", "res": "residence", "pass": "passage",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
}
# Address tokens that carry no location identity (routing / filler words), dropped entirely rather than mapped.
ADDR_DROP = {
    "cedex",                                          # French postal routing suffix (e.g. "75015 PARIS CEDEX 15")
    "po", "box",                                       # "PO Box" (US) - the box number alone is kept
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va",
    "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "telangana": "ts", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk",
    "ladakh": "la", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
}
STATE_MAP = {**US_STATES, **IN_STATES}
STATE_ALIAS = {"tg": "ts", "or": "or"}  # India alt codes -> canonical; harmless elsewhere
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, STATE_MAP), key=len, reverse=True)) + r")\b")

_DOTTED = re.compile(r"(?<=\b[a-z])\.(?=[a-z]\b)")   # l.l.p. -> llp, p.c. -> pc
_NONALNUM = re.compile(r"[^a-z0-9]+")
_NONLATIN = re.compile(r"[^\x00-˿Ḁ-ỿ -⁯]")  # any non-Latin-script char


def has_nonlatin(s: str) -> int:
    return int(bool(_NONLATIN.search(s)))


def _base(s: str) -> str:
    s = unidecode(s).lower().replace("&", " and ").replace("+", " and ")
    s = _DOTTED.sub("", s)
    return _NONALNUM.sub(" ", s).strip()


def norm_name(s: str):
    """Returns (full_name, core_name): abbreviations expanded; core also drops legal/stop tokens."""
    toks = [NAME_ABBR.get(t, t) for t in _base(s).split()]
    core = [t for t in toks if t not in LEGAL_DROP]
    return " ".join(toks), " ".join(core if core else toks)


def norm_addr(s: str) -> str:
    s = _base(s)
    s = _STATE_RE.sub(lambda m: STATE_MAP[m.group(0)], s)
    out = []
    for t in s.split():
        if t in ADDR_DROP:
            continue
        if t.isdigit():
            t = t.lstrip("0") or "0"
        else:
            t = ADDR_ABBR.get(t, STATE_ALIAS.get(t, t))
        out.append(t)
    return " ".join(out)


_SKEL_SUBS = (("ph", "f"), ("ck", "k"), ("x", "ks"), ("c", "k"), ("q", "k"), ("w", "v"), ("z", "s"))
_VOWELS_H = re.compile(r"[aeiouyh]+")
_REPEAT = re.compile(r"(.)\1+")


def skeleton(tok: str) -> str:
    """Rough phonetic consonant skeleton so transliterations / vowel typos collapse together
    (laxmi ~ lakssmi -> 'lksm', private ~ praivett -> 'prvt')."""
    if tok.isdigit():
        return tok
    for a, b in _SKEL_SUBS:
        tok = tok.replace(a, b)
    tok = _REPEAT.sub(r"\1", _VOWELS_H.sub("", tok))
    return tok


def skel_name(core: str) -> str:
    return " ".join(s for s in (skeleton(t) for t in core.split()) if s)


def normalise_frame(df):
    """Adds name_full, name_core, name_skel, addr, nonlatin columns to a records dataframe (in place, returns df)."""
    nm = [norm_name(x) for x in df["business_name"].tolist()]
    df["name_full"] = [a for a, _ in nm]
    df["name_core"] = [b for _, b in nm]
    df["name_skel"] = [skel_name(b) for _, b in nm]
    df["addr"] = [norm_addr(x) for x in df["business_address"].tolist()]
    # street + city key: the address with every token that contains a digit removed (house / unit / PIN numbers)
    df["addr_street"] = [" ".join(t for t in a.split() if not any(ch.isdigit() for ch in t)) for a in df["addr"].tolist()]
    df["nonlatin"] = [has_nonlatin(x) for x in df["business_name"].tolist()]
    return df

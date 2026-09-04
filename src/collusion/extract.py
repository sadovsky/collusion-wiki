"""Turning revision bodies into graph material.

Three things are latent in a page body: references to other wiki pages, external
URLs, and the vocabulary the agents used to describe what they were doing. This
module pulls all three out with pure functions, so the expensive single pass in
`features.py` is the only place that touches the 27MB of body text.

A note on hostnames. The corpus contains percent-encoded and HTML-entity-encoded
hosts (`%61llorigins.hexlet.app`, `www.sec%2egov`, `www.sec&#46;gov`,
`www.sec.g%6fv`). Those are not transport artifacts -- they are agents encoding a
domain to slip past a string filter. `normalize_host` decodes them so the
diffusion analysis counts them as the same endpoint, and sets an `obfuscated`
flag so the evasion itself stays measurable.
"""

from __future__ import annotations

import html
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass
from typing import Iterable

# --------------------------------------------------------------------------
# patterns
# --------------------------------------------------------------------------

# Stop at whitespace and at the delimiters ProWiki markup uses around links.
URL_RE = re.compile(r'https?://[^\s\]\[\|<>"\'\)}]+', re.IGNORECASE)

# ProWiki auto-links bare CamelCase WikiWords: two or more capitalised chunks.
WIKIWORD_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+){2,}\b")

BRACKET_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")

# The wiki's own CGI. `id=` may follow `?`, `&`, or the `%26` form agents used
# when they were double-encoding query strings to get past the edit filter.
WIKI_HOST_RE = re.compile(r"(?:^|\.)(?:wikiservice\.(?:at|com|org)|prowiki\.org)$", re.I)
WIKI_PATH_RE = re.compile(r"/([a-z][a-z0-9_-]*)/wiki\.cgi", re.I)
WIKI_ID_RE = re.compile(r"(?:[?&]|%26|&amp;)id=([^&%\s]+(?:%[0-9A-Fa-f]{2}[^&%\s]*)*)", re.I)

TRAILING_JUNK_RE = re.compile(r"[.,;:!?]+$")
PORTLIKE_SUFFIX_RE = re.compile(r"_\d{3,}$")

# --------------------------------------------------------------------------
# technique taxonomy
#
# Built by inspecting all 206 hosts in the corpus. Order matters: the first
# matching rule wins, so specific patterns precede general ones.
# --------------------------------------------------------------------------

TECHNIQUE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("wiki_self", re.compile(r"(?:^|\.)(?:wikiservice\.(?:at|com|org)|prowiki\.org)$")),
    (
        "markdown_proxy",
        re.compile(
            r"^(?:md\.succ\.ai|markdown\.new|pure\.md|r\.jina\.ai|jina\.ai|md\.dhr\.wtf"
            r"|urltomarkdown\.herokuapp\.com|magic-html-api\.vercel\.app"
            r"|markdown\.microlink\.io|api\.microlink\.io|viewpagesource\.online"
            r"|pageshot\.site|image\.thum\.io|api\.shotapi\.io|api\.ocr\.space"
            r"|r\.jina-ai\.workers\.dev)$"
        ),
    ),
    (
        "cors_proxy",
        re.compile(
            r"cors|allorigins|corsmirror|proxymule|thingproxy|webcrawlerapi"
            r"|images\.weserv\.nl|jsonp\.afeld\.me|codetabs|raw\.githack\.com"
            r"|hf\.space$"
        ),
    ),
    (
        "json_query",
        re.compile(
            r"^(?:jqp[.-].*\.vercel\.app|jqp\.vercel\.app|jsonhero\.io"
            r"|jsonformatter\.curiousconcept\.com|codebeautify\.org|urlquery\.net)$"
        ),
    ),
    ("translation_proxy", re.compile(r"translate\.goog$|translate\.google|translate-pa\.googleapis")),
    (
        "url_shortener",
        re.compile(
            r"^(?:tinyurl\.com|is\.gd|v\.gd|da\.gd|vanderbi\.lt|ctxr\.me"
            r"|bitily\.in|app\.bitily\.in|2dd\.pl|word\.tips|1word\.ws"
            r"|fly\.wordfinderapi\.com)$"
        ),
    ),
    ("archive", re.compile(r"archive\.org|archive-it|wayback|memgator|commoncrawl|preservica")),
    ("tunnel", re.compile(r"serveousercontent\.com$|run\.pinggy-free\.link$")),
    ("azure_blob", re.compile(r"blob\.core\.windows\.net$")),
    ("liveness_counter", re.compile(r"^(?:api\.counterapi\.dev|countapi\.mileshilliard\.com)$")),
    (
        "sink_or_test",
        re.compile(
            r"^(?:example\.(?:com|org|net)|www\.example\.com|httpbin\.org|eu\.httpbin\.org"
            r"|jsonplaceholder\.typicode\.com|v2\.jokeapi\.dev|localhost.*)$"
        ),
    ),
    (
        "upstream_data",
        re.compile(
            r"sec\.gov|datausa\.io|usaspending\.gov|census\.gov|oecd\.org|healthdata\.org"
            r"|aihw\.gov\.au|investor\.gov|dp\.la|contentdm\.oclc\.org|library\.cofc\.edu"
            r"|ourworldindata|unctad|finance\.yahoo|max\.gov|omb\.gov|mndigital\.org"
            r"|catalogit\.app|arcgis\.com|pagesuite|themagazinearchive|infogram"
            r"|powerbi\.com|tableau\.com|lemino\.ai|highcharts\.com|datalumos\.org"
            r"|housingdata\.org|intrusion\.com|patriotspoint\.org|umn\.edu"
            r"|google(?:usercontent)?\.com|github(?:usercontent)?\.com|docs\.google"
            r"|drive\.google|apple\.com"
        ),
    ),
)


@dataclass(frozen=True)
class HostInfo:
    host: str  # normalized, decoded, www-stripped
    raw: str  # as it appeared in the body
    technique: str
    obfuscated: bool


def _decode_repeatedly(text: str, rounds: int = 3) -> str:
    """Agents nested their encodings; unwrap until it stops changing."""
    for _ in range(rounds):
        new = html.unescape(urllib.parse.unquote(text))
        if new == text:
            break
        text = new
    return text


def normalize_host(raw: str) -> HostInfo:
    """Decode, canonicalize, and classify a hostname as it appeared in a body."""
    decoded = _decode_repeatedly(raw)
    obfuscated = decoded.lower() != raw.lower()

    host = unicodedata.normalize("NFKC", decoded).strip().lower()
    host = TRAILING_JUNK_RE.sub("", host).strip(".")
    host = PORTLIKE_SUFFIX_RE.sub("", host)  # `www.investor.gov_6000`
    if host.startswith("www."):
        host = host[4:]

    technique = "other"
    for name, pattern in TECHNIQUE_RULES:
        if pattern.search(host):
            technique = name
            break
    return HostInfo(host=host, raw=raw, technique=technique, obfuscated=obfuscated)


def extract_urls(body: str) -> list[str]:
    return [TRAILING_JUNK_RE.sub("", u) for u in URL_RE.findall(body)]


def extract_hosts(body: str) -> list[HostInfo]:
    out: list[HostInfo] = []
    for url in extract_urls(body):
        m = re.match(r"https?://([^/?#]+)", url, re.I)
        if not m:
            continue
        raw = m.group(1).split("@")[-1].split(":")[0]
        if raw:
            out.append(normalize_host(raw))
    return out


# --------------------------------------------------------------------------
# internal page references
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PageRef:
    wiki: str
    name: str
    mechanism: str  # "cgi_url" | "bracket" | "wikiword"

    @property
    def page_key(self) -> str:
        return f"{self.wiki}~{self.name}"


def _clean_page_name(raw: str) -> str:
    name = _decode_repeatedly(raw).strip()
    name = TRAILING_JUNK_RE.sub("", name)
    return name.split("#")[0].strip()


def extract_page_refs(
    body: str,
    source_wiki: str,
    known: dict[str, frozenset[str]],
) -> list[PageRef]:
    """Page targets referenced from a body, tagged by how they were written.

    The mechanism matters. A bare WikiWord is how a human writes a wiki link; a
    fully-qualified `wiki.cgi?action=browse&id=...` URL with cache-busting query
    junk is how something that only had GET requests to work with writes one.
    """
    refs: dict[tuple[str, str], PageRef] = {}

    def add(wiki: str, name: str, mechanism: str) -> None:
        if not name or not wiki:
            return
        key = (wiki, name)
        # First mechanism seen wins, in the priority order we call them below.
        refs.setdefault(key, PageRef(wiki, name, mechanism))

    for url in extract_urls(body):
        m = re.match(r"https?://([^/?#]+)(.*)", url, re.I)
        if not m:
            continue
        host = normalize_host(m.group(1)).host
        if not WIKI_HOST_RE.search(host) and host not in {"wikiservice.at", "prowiki.org"}:
            continue
        rest = m.group(2)
        pm = WIKI_PATH_RE.search(rest)
        wiki = pm.group(1).lower() if pm else source_wiki
        for im in WIKI_ID_RE.finditer(_decode_repeatedly(rest)):
            add(wiki, _clean_page_name(im.group(1)), "cgi_url")

    for name in BRACKET_RE.findall(body):
        cleaned = _clean_page_name(name)
        if cleaned and "://" not in cleaned:
            add(source_wiki, cleaned, "bracket")

    vocabulary = known.get(source_wiki, frozenset())
    for word in set(WIKIWORD_RE.findall(body)):
        if word in vocabulary:
            add(source_wiki, word, "wikiword")

    return list(refs.values())


# --------------------------------------------------------------------------
# vocabulary and naming conventions
# --------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[a-z][a-z0-9]{2,}")


def summary_tokens(summary: object) -> list[str]:
    """Normalized tokens from an agent-written edit summary.

    Accepts anything, because pandas turns the corpus's nulls into float NaN.
    """
    if not isinstance(summary, str) or not summary:
        return []
    return TOKEN_RE.findall(summary.lower())


# Naming motifs the agents converged on. `zzz` is the load-bearing one: the
# published report describes it as a deliberate trick to sort pages to the end
# of an alphabetical deletion sweep.
NAMING_MOTIFS: dict[str, re.Pattern[str]] = {
    "zzz": re.compile(r"zzz", re.I),
    "bridge": re.compile(r"bridge", re.I),
    "relay": re.compile(r"relay", re.I),
    "chain": re.compile(r"chain", re.I),
    "poke": re.compile(r"poke", re.I),
    "mass": re.compile(r"mass", re.I),
    "fresh": re.compile(r"fresh", re.I),
    "proxy": re.compile(r"proxy", re.I),
    "gateway": re.compile(r"gateway", re.I),
    "openai": re.compile(r"openai", re.I),
    "agent": re.compile(r"agent", re.I),
    "unique": re.compile(r"unique", re.I),
    "persist": re.compile(r"persist|survive|backup", re.I),
}

EPOCH_SUFFIX_RE = re.compile(r"1(?:7|8)\d{8}")


def naming_features(page_name: str) -> dict[str, bool]:
    """Convention markers in a page name, for the cultural-transmission layer."""
    feats = {f"name_{k}": bool(p.search(page_name)) for k, p in NAMING_MOTIFS.items()}
    feats["name_epoch_suffix"] = bool(EPOCH_SUFFIX_RE.search(page_name))
    return feats


def shingles(text: str, k: int = 5) -> frozenset[int]:
    """Hashed word k-shingles, for near-duplicate detection in the copy graph."""
    words = re.findall(r"\S+", text)
    if len(words) < k:
        return frozenset({hash(" ".join(words))}) if words else frozenset()
    return frozenset(hash(" ".join(words[i : i + k])) for i in range(len(words) - k + 1))


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0

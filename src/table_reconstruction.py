"""Generic column-anchored table reconstruction: candidate rows, documentary evidence only.

Ported unchanged in behaviour from the isolated research harness that measured 43 correct
line-item fields with zero retained wrong values across seven layouts. The module is pure: it
neither reads an image, nor calls OCR, nor reaches the model. It consumes word boxes from reads
the pipeline already produced and returns candidates, each with the coordinates, confidences and
preprocessing families that back it, plus the reason every rejected value was rejected.

Evidence rules, in the order they protect a value:
  * one read supplies the geometry, every other read only corroborates it;
  * a cell is kept when at least two distinct preprocessing families print the same value in the
    same cell and every contributing token is above the strict confidence bar - agreement between
    preprocessings of one engine is corroboration, never proof, so text is diagnostic only;
  * a column gets its role from the whole header label printed above it, and an ambiguous or
    overlapping cue assigns nothing;
  * document-level reconciliation against a labelled printed total voids a role only on a provable
    contradiction, and a row whose printed amount disagrees with quantity x unit price keeps the
    printed amount and loses the derived pair. A product is never exported as an amount.
"""

import re
import unicodedata
from collections import defaultdict
from decimal import Decimal, InvalidOperation

MONEY = re.compile(r"^-?(?:\d{1,3}(?:[ \u00a0]\d{3})+|\d+(?:,\d{3})+|\d+)(?:[.,]\d{1,2})$")
INTEGER = re.compile(r"^-?\d{1,4}$")
GROUP = re.compile(r"^\d{3}[.,]\d{1,2}$")
DURATION = re.compile(r"^\d{1,2}[:h]\d{1,2}$")
# A clock duration printed without its separator: '0300', '06.30', '04:20'.
ZERO_PADDED = re.compile(r"^0\d[.,:]?\d{2}$")
CURRENCY = re.compile(r"[$€£]")
Y_BAND_GAP = 0.005
OVERLAP_RATIO = 0.5
AXIS_TOLERANCE = 0.015
TOLERANCE = Decimal("0.02")
HEADER_TO_REGION = 0.02
HEADER_MERGE_GAP = 0.02
RUN_GAP = 0.03
HEADER_RUN_GAP = 0.05
TOKEN_BAR = 60
CONNECTOR = re.compile(r"^[-–—•…|]+$")
# Most specific label first: 'Total TVA' is a tax column, never a line amount.
ROLE_PRIORITY = ("description", "quantity", "unit_price", "amount_tax", "amount_gross", "amount_ht")
ROLE_TOKENS = {
    "quantity": r"\b(?:quantit\w*|quantity|qty|qte|units?|hrs)\b",
    "unit_price": r"\b(?:prix\s+unit\w*|unit\w*\s+price|unitaire|rate|pu)\b",
    "amount_ht": r"\b(?:montant|amount|ht|subtotal|sous[- ]?total|total)\b",
    "amount_tax": r"\b(?:tva|tvq|vat|tax)\w*",
    "amount_gross": r"\b(?:ttc|due|payable|g[eé]n[eé]ral|gross)\b",
    "description": r"\b(?:description|titre|product|item|designation|article|service|libell\w*|prestation)\b",
}
NON_COLUMN_TOKENS = re.compile(r"\b(?:reference|ref|lines?|page|date|client|num[eé]ro|n°)\b", re.I)
TOTAL_WORD = re.compile(
    r"^(?:total|montant|sous[- ]?total|subtotal|invoice|due|balance|solde|please)\w*$", re.I)
# Generic summary-line semantics, never a supplier or a template.
SUMMARY_WORD = re.compile(
    r"\b(?:tva|tvq|tps|vat|tax|livraison|shipping|handling|delivery|frais|escompte|rabais|"
    r"discount|acompte|arrhes|total|subtotal|sous[- ]?total|due|balance|solde|please|payable)\b",
    re.I,
)
TAX_LABEL = re.compile(r"\b(?:tva|tvq|tps|vat|tax)\b", re.I)
GROSS_LABEL = re.compile(r"\b(?:ttc|due|payable|g[eé]n[eé]ral|gross|please\s+pay|balance)\b", re.I)
HT_LABEL = re.compile(r"\b(?:ht|hors\s+taxe|excl|net|subtotal|sous[- ]?total|montant|before\s+tax)\b", re.I)
SEPARATOR = re.compile(r"^[|()\[\]…•·:\-/]+$")
CELL_ROLES = ("quantity", "unit_price", "amount_ht", "amount_tax", "amount_gross")


def plain(text: str) -> str:
    """Accents stripped: 'Référence' is a reference label whatever the read did to its accents."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def parse_money(text: str) -> Decimal | None:
    """A printed amount, or None whenever the grouping could be read two ways.

    Both separators present is unambiguous - the last one is the decimal mark - in either
    convention. A single comma is not: '1,234' is a French decimal or an English thousand, so it
    is refused rather than guessed.
    """
    cleaned = CURRENCY.sub("", text).strip().replace("\u00a0", "").replace(" ", "")
    if not cleaned or DURATION.match(cleaned):
        return None
    negative = cleaned.startswith("-")
    cleaned = cleaned.lstrip("-+")
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", cleaned):
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(?:,\d{3})+\.\d+", cleaned):
        cleaned = cleaned.replace(",", "")
    elif re.fullmatch(r"\d{1,3}(?:,\d{3})+", cleaned):
        return None
    elif cleaned.count(",") > 1 and cleaned.count(".") > 1:
        return None
    elif "," in cleaned and "." in cleaned:
        return None
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")
    if cleaned.count(".") > 1 or not cleaned.strip(".,"):
        return None
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    return -value if negative else value


def numeric_word(word: dict) -> tuple[Decimal, str] | None:
    """(value, kind) when the word is a cell amount, else None."""
    text = CURRENCY.sub("", word["text"]).strip().rstrip(".")
    if not text or SEPARATOR.match(text):
        return None
    bare = text.replace(" ", "").replace("\u00a0", "")
    if DURATION.match(bare) or ":" in bare:
        return None
    if MONEY.match(bare) and any(separator in bare for separator in ".,"):
        value = parse_money(bare)
        return (value, "money") if value is not None else None
    if INTEGER.match(bare):
        return (Decimal(bare), "integer")
    return None


def merge_groups(band: list[dict]) -> list[dict]:
    """Spaced thousands ('1' '350,00') are one printed amount, not two cells."""
    ordered = sorted(band, key=lambda word: word["box"][0])
    merged, skip = [], set()
    for index, word in enumerate(ordered[:-1]):
        nxt = ordered[index + 1]
        left = CURRENCY.sub("", word["text"]).strip()
        right = CURRENCY.sub("", nxt["text"]).strip()
        if INTEGER.match(left) and GROUP.match(right) and nxt["box"][0] - word["box"][2] < 0.03:
            merged.append(dict(
                word, text=left + right,
                box=(word["box"][0], word["box"][1], nxt["box"][2], word["box"][3]),
                confidence=min(word["confidence"], nxt["confidence"]),
            ))
            skip.update({id(word), id(nxt)})
    for word in ordered:
        if id(word) not in skip:
            merged.append(word)
    return sorted(merged, key=lambda word: word["box"][0])


def words_of(tokens, *, family: str, read: str, size) -> list[dict]:
    """OCR tokens into normalised geometry: families print at different scales.

    Boxes stay pixel-exact for the source column, so a candidate can be traced back to where it
    was printed; every comparison between two reads happens on the normalised box.
    """
    width, height = size
    out = []
    for text, box, confidence in tokens:
        left, top, right, bottom = box
        out.append({
            "text": text, "confidence": float(confidence), "family": family, "read": read,
            "box": (left / width, top / height, right / width, bottom / height),
            "pixel": tuple(round(float(value), 1) for value in box),
        })
    return out


def cluster_bands(words: list[dict]) -> list[list[dict]]:
    parent = list(range(len(words)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    active = []
    for index in sorted(range(len(words)), key=lambda i: words[i]["box"][1]):
        bottom = words[index]["box"][3]
        top = words[index]["box"][1]
        active = [other for other in active if words[other]["box"][3] > top - Y_BAND_GAP]
        for other in active:
            other_top, other_bottom = words[other]["box"][1], words[other]["box"][3]
            overlap = min(bottom, other_bottom) - max(top, other_top)
            smaller = min(bottom - top, other_bottom - other_top)
            if smaller > 0 and overlap / smaller > OVERLAP_RATIO:
                parent[find(other)] = find(index)
        active.append(index)
    groups = defaultdict(list)
    for index, word in enumerate(words):
        groups[find(index)].append(word)
    return [merge_groups(sorted(group, key=lambda word: word["box"][0]))
            for group in groups.values()]


def top(band: list[dict]) -> float:
    return min(word["box"][1] for word in band) if band else 1.0


def bottom(band: list[dict]) -> float:
    return max(word["box"][3] for word in band) if band else 0.0


def band_top_gap(previous_bottom: float, next_top: float) -> float:
    return next_top - previous_bottom


def money_in(band: list[dict]) -> list[tuple[dict, Decimal]]:
    return [
        (word, numeric_word(word)[0]) for word in band
        if numeric_word(word) and numeric_word(word)[1] == "money"
    ]


def build_axes(bands: list[list[dict]]) -> list[dict]:
    axes = []
    for band in bands:
        for word in band:
            parsed = numeric_word(word)
            if parsed is None or word["confidence"] < TOKEN_BAR:
                continue
            value, kind = parsed
            right = word["box"][2]
            for axis in axes:
                if abs(axis["right"] - right) <= AXIS_TOLERANCE:
                    axis["members"].append((band, word, value, kind))
                    axis["right"] = sum(
                        member[1]["box"][2] for member in axis["members"]) / len(axis["members"])
                    break
            else:
                axes.append({"right": right, "members": [(band, word, value, kind)]})
    axes.sort(key=lambda axis: axis["right"])
    return axes


def axis_span(axis: dict) -> tuple[float, float]:
    return (min(member[1]["box"][0] for member in axis["members"]),
            max(member[1]["box"][2] for member in axis["members"]))


def consolidate_axes(axes: list[dict]) -> list[dict]:
    """One printed column must be one axis.

    The same column can sit a few thousandths apart in two bands (a row amount and the summary
    line under it), which would otherwise create two axes and let one printed amount be counted
    twice: once as a row and once as a fee.
    """
    merged = sorted((dict(axis, members=list(axis["members"])) for axis in axes),
                    key=lambda item: item["right"])
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for position in range(len(merged) - 1):
            left, right = merged[position], merged[position + 1]
            if abs(left["right"] - right["right"]) > AXIS_TOLERANCE * 3:
                continue
            (a_start, a_end), (b_start, b_end) = axis_span(left), axis_span(right)
            overlap = min(a_end, b_end) - max(a_start, b_start)
            narrower = min(a_end - a_start, b_end - b_start)
            if narrower <= 0 or overlap < narrower / 2:
                continue
            combined = dict(left, members=left["members"] + right["members"])
            combined["right"] = sum(
                member[1]["box"][2] for member in combined["members"]) / len(combined["members"])
            merged[position:position + 2] = [combined]
            changed = True
            break
    return merged


def supplement_axes(axes: list[dict], reads: dict[str, list[dict]], primary_key: str) -> list[dict]:
    """A column the primary read lost is still a column when two other families agree on it.

    A row amount can be unreadable in every single read yet be printed at a stable right edge:
    two independent reads placing money in the same narrow x window is documentary evidence that
    the column exists, which is what lets a header word name it.
    """
    known = [axis["right"] for axis in axes]
    candidates = []
    for key, words in reads.items():
        if key == primary_key:
            continue
        family = family_of(key)
        for band in cluster_bands(words):
            for word in band:
                parsed = numeric_word(word)
                if parsed is None or word["confidence"] < TOKEN_BAR:
                    continue
                right = word["box"][2]
                if any(abs(right - value) <= AXIS_TOLERANCE * 2 for value in known):
                    continue
                for candidate in candidates:
                    if abs(candidate["right"] - right) <= AXIS_TOLERANCE:
                        candidate["members"].append((band, word, parsed[0], parsed[1]))
                        candidate["families"].add(family)
                        candidate["right"] = sum(
                            member[1]["box"][2] for member in candidate["members"]
                        ) / len(candidate["members"])
                        break
                else:
                    candidates.append({
                        "right": right, "members": [(band, word, parsed[0], parsed[1])],
                        "families": {family},
                    })
    out = list(axes)
    for candidate in candidates:
        if len(candidate["families"]) < 2:
            continue
        out.append({"right": candidate["right"], "members": candidate["members"],
                    "cross_read": sorted(candidate["families"])})
    return sorted(out, key=lambda axis: axis["right"])


def family_of(read_key: str) -> str:
    """One preprocessing family votes once, whatever number of reads was taken of it."""
    return read_key.split("_", 1)[0]


def header_runs(band: list[dict]) -> list[list[dict]]:
    """The printed header cells of a band: a pipe or a wide gap ends a label.

    'Total TVA' is one column label, so it must be matched as a whole: read word by word, the
    generic word 'Total' would claim the tax column as the line amount and every per-line check
    built on that column would then contradict the document.
    """
    runs, current = [], []
    for word in sorted(band, key=lambda item: item["box"][0]):
        text = word["text"].strip()
        if not text or word["confidence"] < TOKEN_BAR:
            continue
        if SEPARATOR.match(text):
            if current:
                runs.append(current)
                current = []
            continue
        if current and word["box"][0] - current[-1]["box"][2] > HEADER_RUN_GAP:
            runs.append(current)
            current = []
        current.append(word)
    if current:
        runs.append(current)
    return runs


def role_words(band: list[dict]) -> dict[str, dict]:
    """One role per printed header cell, matched on the whole label by specificity order."""
    roles = {}
    for run in header_runs(band):
        text = " ".join(word["text"] for word in run)
        if NON_COLUMN_TOKENS.search(text):
            continue
        for role in cued_roles(text):
            if role in roles:
                continue
            # The cell's last word sits over its own column: 'Total TVA' claims the TVA axis.
            roles[role] = run[-1]
            break
    return roles


def header_group(bands: list[list[dict]]) -> list | None:
    """Consecutive label-only bands above the data; a wrapped header still names its columns."""
    candidates = [
        (index, band, role_words(band)) for index, band in enumerate(bands)
        if len(role_words(band)) >= 2 and not money_in(band)
    ]
    best, best_roles, best_group = None, {}, None
    for index, band, roles in candidates:
        group = [band]
        merged = dict(roles)
        # A wrapped header puts its second half directly under the first, without any amount.
        for follower in bands[index + 1:index + 3]:
            if band_top_gap(bottom(group[-1]), top(follower)) > HEADER_MERGE_GAP:
                break
            extra = role_words(follower)
            if not extra or money_in(follower):
                break
            group.append(follower)
            for role, word in extra.items():
                merged.setdefault(role, word)
        if len(merged) > len(best_roles):
            best, best_roles, best_group = band, merged, group
    if best is None:
        return None
    return [word for band in best_group for word in band], best_roles


def cued_roles(text: str) -> list[str]:
    """The roles a whole column label names, minus the cues a longer phrase already explains.

    'Net unit price' cues the plural quantity only because that stem is also the first word of the
    unit-price phrase: the short cue sits inside the long one, so it names nothing by itself, and
    reading it as a quantity puts the price column under the quantity role. 'Total TVA' keeps both
    cues - neither contains the other - and there the priority order still decides, because a bare
    sum word never says which sum it is.
    """
    cues = [
        (role, match.start(), match.end())
        for role in ROLE_PRIORITY
        for match in re.finditer(ROLE_TOKENS[role], text, re.I)
    ]
    kept = {
        role for role, start, end in cues
        if not any(other != role and other_start <= start and end <= other_end
                   and other_end - other_start > end - start
                   for other, other_start, other_end in cues)
    }
    return [role for role in ROLE_PRIORITY if role in kept]


def assign_roles(axes: list[dict], header: list[dict]) -> tuple[dict, dict]:
    """Give each numeric column the header words printed above it, then read that label whole.

    A column label is a property of the column, not of one word: 'Total TVA' sits over the tax
    column, so reading the words that overlap a column and matching their combined text is what
    keeps the generic word 'Total' from claiming a line-amount axis. Columns are matched by
    geometry first, so a wrapped or half-legible label costs a role, never a wrong column.
    """
    floor = max(word["box"][3] for word in header) - HEADER_TO_REGION
    claims = {}
    for index, axis in enumerate(axes):
        members = [member for member in axis["members"] if top(member[0]) >= floor]
        if not members:
            continue
        left, right = axis_span(dict(members=members))
        claims[index] = (left - 0.02, right + 0.02)
    if not claims:
        return {}, {"amount_ht": "no column under the header"}
    labels = defaultdict(list)
    for word in sorted(header, key=lambda item: item["box"][0]):
        centre = (word["box"][0] + word["box"][2]) / 2
        owner = min(
            claims,
            key=lambda index: max(claims[index][0] - centre, centre - claims[index][1], 0),
        )
        labels[owner].append(word)
    pinned, evidence = {}, {}
    for index in sorted(labels, key=lambda position: claims[position][0]):
        words = [word for word in labels[index]
                 if not NON_COLUMN_TOKENS.search(plain(word["text"]))]
        if not words:
            continue
        text = " ".join(word["text"] for word in words)
        for role in cued_roles(text):
            if role in pinned or role == "description":
                continue
            best = max(words, key=lambda word: word["confidence"])
            pinned[role] = index
            evidence[role] = "column %d label %r (best word %r conf %d)" % (
                index, text[:34], best["text"], int(best["confidence"]))
            break
    declared = next(
        (word for word in header
         if re.search(ROLE_TOKENS["description"], word["text"], re.I)
         and word["confidence"] >= TOKEN_BAR),
        None,
    )
    if declared is not None:
        evidence["description"] = "header %r names a text column" % declared["text"]
    for role in ROLE_TOKENS:
        evidence.setdefault(role, "unpinned: no column label matched")
    return pinned, evidence


def labelled_band(band: list[dict]) -> dict | None:
    """A band that prints a labelled summary amount ('Total TVA 270,00 €'), or None."""
    money = money_in(band)
    if not money:
        return None
    labels = [
        word for word in band
        if TOTAL_WORD.match(word["text"]) and word["confidence"] >= TOKEN_BAR
    ]
    if not labels or min(word["box"][2] for word in labels) >= money[-1][0]["box"][0]:
        return None
    text = " ".join(word["text"] for word in band)
    kind = (
        "tax" if TAX_LABEL.search(text) else "gross" if GROSS_LABEL.search(text)
        else "ht" if HT_LABEL.search(text) else "unqualified"
    )
    return {"label": text[:70], "kind": kind, "value": money[-1][1], "top": top(band)}


def labelled_totals(bands, reads, primary_key) -> list[dict]:
    """Summary amounts corroborated across preprocessing families, not just the primary read.

    The printed HT total is what makes document-level reconciliation possible, and it is often the
    word the primary read corrupts (spaced thousands, currency glyph). Requiring two families per
    total, at the same printed position, keeps a misread summary out of the reconciliation instead
    of letting it void correct row values.
    """
    positions = []
    for key, words in reads.items():
        family = family_of(key)
        for band in (bands if key == primary_key else cluster_bands(words)):
            entry = labelled_band(band)
            if entry is None:
                continue
            record = dict(entry, read=key, family=family)
            bucket = next(
                (item for item in positions
                 if item["kind"] == entry["kind"] and item["value"] == entry["value"]
                 and abs(item["entries"][0]["top"] - entry["top"]) < 0.015),
                None,
            )
            if bucket is None:
                positions.append({"kind": entry["kind"], "value": entry["value"], "entries": [record]})
            else:
                bucket["entries"].append(record)
    out = []
    for item in positions:
        families = {entry["family"] for entry in item["entries"]}
        if len(families) < 2:
            continue
        chosen = next(
            (entry for entry in item["entries"] if entry["read"] == primary_key),
            max(item["entries"], key=lambda entry: entry["top"]),
        )
        out.append({
            "kind": item["kind"], "value": item["value"], "label": chosen["label"],
            "top": chosen["top"], "families": sorted(families),
            "labels": sorted({entry["label"] for entry in item["entries"]})[:3],
        })
    return sorted(out, key=lambda item: item["top"])


def merged_money_word(word: dict, key: str, reads: dict) -> Decimal | None:
    """Apply the same thousands merge to a corroborating read, on its own band grouping."""
    if not INTEGER.match(CURRENCY.sub("", word["text"]).strip() or "x"):
        return None
    partner = next(
        (other for other in reads[key]
         if abs((other["box"][1] + other["box"][3]) / 2 - (word["box"][1] + word["box"][3]) / 2) < 0.01
         and GROUP.match(CURRENCY.sub("", other["text"]).strip() or "x")
         and 0 <= other["box"][0] - word["box"][2] < 0.03),
        None,
    )
    if partner is None or partner["confidence"] < TOKEN_BAR:
        return None
    return parse_money(
        CURRENCY.sub("", word["text"]).strip() + CURRENCY.sub("", partner["text"]).strip())


def corroborate(axis: dict, band: list[dict], reads: dict, primary_key: str) -> dict | None:
    """The one value two distinct families print in this cell, with its full provenance."""
    by_family = defaultdict(list)
    floor, ceiling = top(band) - 0.004, bottom(band) + 0.004
    for key, words in reads.items():
        family = family_of(key)
        for word in words:
            parsed = numeric_word(word)
            if parsed is None or word["confidence"] < TOKEN_BAR:
                continue
            value, kind = parsed
            centre_y = (word["box"][1] + word["box"][3]) / 2
            if not floor <= centre_y <= ceiling:
                continue
            if abs(word["box"][2] - axis["right"]) > AXIS_TOLERANCE * 2:
                continue
            grouped = merged_money_word(word, key, reads)
            if grouped is not None:
                value = grouped
            by_family[family].append((key, value, word))
    values = defaultdict(set)
    for family, entries in by_family.items():
        for _, value, _ in entries:
            values[value].add(family)
    agreed = [value for value, families in values.items() if len(families) >= 2]
    if len(agreed) != 1:
        return None
    value = agreed[0]
    hits = [
        (key, family, word) for family, entries in by_family.items()
        for key, got, word in entries if got == value
    ]
    return {
        "value": str(value),
        "families": sorted({family for _, family, _ in hits}),
        "reads": len(hits),
        "raw": sorted({word["text"] for _, _, word in hits})[:3],
        "confidences": sorted({int(word["confidence"]) for _, _, word in hits})[:6],
        "positions": [
            {"read": key, "family": family, "text": word["text"],
             "confidence": int(word["confidence"]),
             "box": [round(float(value), 4) for value in word["box"]],
             "pixel": list(word["pixel"])}
            for key, family, word in hits
        ][:4],
        "competing": sorted(str(other) for other in values if other != value),
        "primary": primary_key in {key for key, _, _ in hits},
    }


def region(bands, header, totals, axes, reads, primary_key) -> list[list[dict]]:
    """Row bands are a contiguous, evenly pitched sequence under the header."""
    floor = bottom(header) + 0.004 if header else None
    ceiling = min((total["top"] for total in totals), default=1.0)
    money_bands = [band for band in bands if len(money_in(band)) >= 2]
    if floor is None:
        after = [band for band in money_bands if top(band) < ceiling]
        if not after:
            # Headerless table: any band carrying two numeric cells can open the row region, so a
            # qty + amount layout is still readable. The roles themselves stay unpinned unless a
            # printed total proves the amount column.
            after = [band for band in bands
                     if sum(1 for word in band if numeric_word(word)) >= 2 and top(band) < ceiling]
        if not after:
            return []
        floor = top(after[0])
    kept = []
    for band in bands:
        if not floor <= top(band) < ceiling:
            continue
        text = " ".join(word["text"] for word in band)
        if SUMMARY_WORD.search(text):
            continue
        if not any(numeric_word(word) for word in band):
            continue
        kept.append(band)
    return cut_at_pitch_gap(kept)


def cut_at_pitch_gap(kept: list[list[dict]]) -> list[list[dict]]:
    """A vertical break larger than twice the median row pitch ends the table."""
    if len(kept) < 3:
        return kept
    pitches = [top(kept[index + 1]) - top(kept[index]) for index in range(len(kept) - 1)]
    pitches = [pitch for pitch in pitches if pitch > 0.001]
    if not pitches:
        return kept
    pitches.sort()
    median = pitches[len(pitches) // 2]
    for index, pitch in enumerate(
        [top(kept[position + 1]) - top(kept[position]) for position in range(len(kept) - 1)]
    ):
        if pitch > max(2.5 * median, 0.02):
            return kept[: index + 1]
    return kept


def axis_sum(axis: dict, bands, reads, primary_key) -> tuple[Decimal, int]:
    total, cells = Decimal("0"), 0
    for band in bands:
        cell = corroborate(axis, band, reads, primary_key)
        if cell:
            total += Decimal(cell["value"])
            cells += 1
    return total, cells


def reconcile_roles(axes, totals, bands, reads, primary_key, pinned, evidence) -> dict:
    for index, axis in enumerate(axes):
        if index in pinned.values():
            continue
        total, cells = axis_sum(axis, bands, reads, primary_key)
        if cells < 1:
            continue
        for labelled in totals:
            if abs(total - labelled["value"]) > TOLERANCE:
                continue
            role = {"ht": "amount_ht", "tax": "amount_tax", "gross": "amount_gross"}.get(
                labelled["kind"])
            if role is None or role in pinned:
                continue
            pinned[role] = index
            evidence[role] = "axis %d corroborated column sum %s over %d cells == %r (%s)" % (
                index, total, cells, labelled["label"], labelled["kind"])
    return evidence


def text_runs(band: list[dict]) -> list[list[dict]]:
    """Contiguous word runs of a band, split where the print leaves a wide gap.

    A separator never breaks a run: '11 rue de la Croix-Marchand' is one printed value and
    dropping its punctuation would slice it into fragments that look like separate columns.
    """
    runs = []
    for word in sorted(band, key=lambda item: item["box"][0]):
        text = word["text"].strip()
        if not text or numeric_word(word) is not None or CONNECTOR.match(text):
            continue
        if runs and word["box"][0] - runs[-1][-1]["box"][2] > RUN_GAP:
            runs.append([word])
        elif runs:
            runs[-1].append(word)
        else:
            runs.append([word])
    return runs


def run_span(run: list[dict]) -> tuple[float, float]:
    return (min(word["box"][0] for word in run), max(word["box"][2] for word in run))


def text_window(rows_band: list[list[dict]]):
    """The free-text column is the run that varies the most in width across the rows.

    A structured column (a date, a reference) repeats at the same width on every row; the
    description of a line does not, so the widest run union is the description. Picking the column
    by its variability is what keeps an unlabelled date column from being exported as the
    description, which is how the earlier experiment produced wrong text.
    """
    runs = [(index, run) for index, band in enumerate(rows_band) for run in text_runs(band)]
    if not runs:
        return None
    clusters = []
    for index, run in runs:
        left, right = run_span(run)
        placed = False
        for cluster in clusters:
            if left <= cluster["right"] + RUN_GAP and cluster["left"] <= right + RUN_GAP:
                cluster["left"] = min(cluster["left"], left)
                cluster["right"] = max(cluster["right"], right)
                cluster["rows"].add(index)
                placed = True
                break
        if not placed:
            clusters.append({"left": left, "right": right, "rows": {index}})
    wanted = max(1, round(0.6 * len(rows_band)))
    candidates = [cluster for cluster in clusters if len(cluster["rows"]) >= wanted] or clusters
    best = max(candidates, key=lambda cluster: (cluster["right"] - cluster["left"], len(cluster["rows"])))
    return (best["left"], best["right"], len(best["rows"]), len(rows_band))


def description_cell(band: list[dict], window, reads: dict) -> dict | None:
    """A line's description is reported only when every word of it is legible and agreed twice."""
    if window is None:
        return None
    left, right = window[0], window[1]
    runs = [
        run for run in text_runs(band)
        if left - 0.01 <= sum(run_span(run)) / 2 <= right + 0.01
    ]
    if len(runs) != 1:
        return None
    run = runs[0]
    if any(word["confidence"] < TOKEN_BAR for word in run):
        return None
    floor, ceiling = top(band) - 0.004, bottom(band) + 0.004
    by_text = defaultdict(set)
    for key, words in reads.items():
        family = family_of(key)
        for word in words:
            text = word["text"].strip()
            if word["confidence"] < TOKEN_BAR or numeric_word(word) is not None or CONNECTOR.match(text):
                continue
            centre_y = (word["box"][1] + word["box"][3]) / 2
            centre_x = (word["box"][0] + word["box"][2]) / 2
            if not floor <= centre_y <= ceiling or not left - 0.01 <= centre_x <= right + 0.01:
                continue
            by_text[text.casefold()].add(family)
    agreed = {text for text, families in by_text.items() if len(families) >= 2}
    if not all(word["text"].strip().casefold() in agreed for word in run):
        return None
    return {
        "text": " ".join(word["text"].strip() for word in run),
        "families": sorted({family for text, families in by_text.items()
                            if text in agreed for family in families}),
        "words": len(run),
        "min_confidence": min(int(word["confidence"]) for word in run),
        "box": [round(float(value), 4) for value in run_span(run)] + [round(top(band), 4),
                                                                     round(bottom(band), 4)],
    }


def duration_column(axis: dict, bands: list[list[dict]]) -> bool:
    """A column whose printed cells are clock durations is not a quantity column.

    The printed separator survives in only some reads ('06:30' becomes '06.30' or '0630'), so the
    gate keys on the zero-padded shape of the whole column, not on a single cell's colon.
    """
    raw = [word["text"] for band_owner, word, _, _ in axis["members"] if band_owner in bands]
    if not raw:
        return False
    clocky = sum(1 for text in raw if ":" in text or DURATION.match(CURRENCY.sub("", text).strip()))
    leading = sum(1 for text in raw if ZERO_PADDED.match(CURRENCY.sub("", text).strip() or "x"))
    return (clocky + leading) / len(raw) >= 0.5


def pick_primary(reads: dict) -> str | None:
    """The read that sees the table best, not the one that sees the most money.

    A header word is what gives a column its meaning, so a read that loses the column labels
    cannot be the geometry source even when it reads every amount.
    """
    best, best_score = None, None
    for key, words in reads.items():
        bands = cluster_bands(words)
        money = [
            word for word in words
            if numeric_word(word) and numeric_word(word)[1] == "money"
            and word["confidence"] >= TOKEN_BAR
        ]
        header = header_group(bands)
        roles = len(header[1]) if header else 0
        score = (roles, len(build_axes(bands)), len(money), sum(w["confidence"] for w in money))
        if best_score is None or score > best_score:
            best, best_score = key, score
    return best


def reject(row: dict, role: str, reason: str) -> None:
    """A value the evidence no longer supports is dropped, but stays visible with its provenance.

    Null is what reaches the document; the rejected cell is what a reviewer needs to know that a
    printed value existed and why it was not trusted.
    """
    cell = row["cells"].pop(role, None)
    if cell is None:
        return
    row.setdefault("rejected_cells", {})[role] = {
        "value": cell["value"], "reason": reason, "families": cell["families"],
        "printed_text": cell["raw"], "confidences": cell["confidences"],
        "positions": cell["positions"],
    }


def certify(rows, pinned, totals, axes, summary_amounts) -> dict:
    """Document-level reconciliation: a column is only trusted when its own sum is provable.

    amount_ht must add up to the labelled HT total, optionally together with the fee and tax
    amounts printed in the same column outside the row sequence. quantity and unit_price are only
    trusted while their product reproduces that same HT total, because that is the one cross-check
    an invoice offers between them and the rest of the document.

    A role is voided only on a provable contradiction: the corroborated cells overshoot the printed
    total, or every row carries the cell and the sum still disagrees. An incomplete sequence is only
    a lower bound, so it can neither prove nor disprove the printed total.
    """
    reasons = {}
    ht_total = next((total["value"] for total in totals if total["kind"] == "ht"), None)
    if "amount_ht" in pinned and ht_total is not None:
        exported = [
            Decimal(row["cells"]["amount_ht"]["value"]) for row in rows
            if "amount_ht" in row["cells"]
        ]
        fees = summary_amounts.get(pinned["amount_ht"], Decimal(0))
        gap = ht_total - sum(exported, Decimal(0)) - fees
        if len(exported) < 1:
            reasons["amount_ht"] = "no corroborated cell"
        elif gap < -TOLERANCE or (len(exported) == len(rows) and gap > TOLERANCE):
            reasons["amount_ht"] = "column sum %s + fees %s != printed HT %s" % (
                sum(exported, Decimal(0)), fees, ht_total)
    if ht_total is not None and {"quantity", "unit_price"} <= set(pinned):
        product, complete = Decimal(0), 0
        for row in rows:
            cells = row["cells"]
            if "quantity" in cells and "unit_price" in cells:
                product += Decimal(cells["quantity"]["value"]) * Decimal(cells["unit_price"]["value"])
                complete += 1
        overshoot = product - ht_total > TOLERANCE
        if complete and (overshoot or (complete == len(rows) and abs(product - ht_total) > TOLERANCE)):
            reasons["quantity"] = "sum(quantity x unit price) %s != printed HT %s over %d rows" % (
                product, ht_total, complete)
            reasons["unit_price"] = reasons["quantity"]
    for reason_role, reason in reasons.items():
        for row in rows:
            reject(row, reason_role, "document reconciliation: %s" % reason)
    return reasons


def reconstruct(reads: dict, *, check=None) -> dict:
    """Candidate rows with their evidence, or an empty candidate set. Never raises on bad input.

    ``check`` is an optional callable raising when the caller's deadline is gone; it is honoured
    between phases so a slow document cannot hold the request open.
    """
    if not reads:
        return {"status": "no_reads", "rows": []}
    primary_key = pick_primary(reads)
    bands = cluster_bands(reads[primary_key])
    axes = supplement_axes(consolidate_axes(build_axes(bands)), reads, primary_key)
    if check is not None:
        check()
    header = header_group(bands)
    totals = labelled_totals(bands, reads, primary_key)
    rows_band = region(bands, header[0] if header else None, totals, axes, reads, primary_key)
    pinned, evidence = {}, {}
    if header:
        pinned, evidence = assign_roles(axes, header[0])
    evidence.update(reconcile_roles(axes, totals, rows_band, reads, primary_key, pinned, evidence))
    window = text_window(rows_band)
    for index in {
        index for role, index in pinned.items()
        if role == "quantity" and index < len(axes) and duration_column(axes[index], rows_band)
    }:
        pinned.pop("quantity", None)
        evidence["quantity"] = "rejected: column %d prints clock durations, not quantities" % index

    rows = []
    for band in rows_band:
        cells = {}
        for role in CELL_ROLES:
            index = pinned.get(role)
            if index is None or index >= len(axes):
                continue
            cell = corroborate(axes[index], band, reads, primary_key)
            if not cell:
                continue
            if role in {"quantity", "unit_price"} and Decimal(cell["value"]) == 0:
                # A per-line rate or quantity of exactly zero is never a usable value: a line that
                # costs nothing is not billed. Null is the honest answer.
                continue
            cells[role] = cell
        row = {
            "band_top": round(top(band), 4),
            "band_bottom": round(bottom(band), 4),
            "band_text": " ".join(word["text"] for word in band)[:76],
            "cells": cells,
        }
        description = description_cell(band, window, reads)
        if description:
            # Reported, never exported: the preprocessing families are the same engine, so an
            # accent error repeated in two reads ('Isméne') cannot be caught from inside them.
            row["detected_text"] = description["text"]
            row["detected_families"] = description["families"]
            row["detected_evidence"] = description
        rows.append(row)

    vetoes = []
    summary_amounts = defaultdict(Decimal)
    row_ids = {id(band) for band in rows_band}
    total_tops = {round(total["top"], 3) for total in totals}
    table_floor = bottom(header[0]) if header else 0.0
    for band in bands:
        if id(band) in row_ids or top(band) < table_floor:
            continue
        # The printed total itself must not be counted as a fee in its own column, or the
        # reconciliation would compare the rows against double the amount they can prove.
        if round(top(band), 3) in total_tops:
            continue
        text = " ".join(word["text"] for word in band)
        # A tax amount printed under the HT column never belongs to the HT sum: adding it would
        # turn a correct row sequence into a fake contradiction.
        if TAX_LABEL.search(text):
            continue
        if not SUMMARY_WORD.search(text):
            continue
        for index, axis in enumerate(axes):
            cell = corroborate(axis, band, reads, primary_key)
            if cell:
                summary_amounts[index] += Decimal(cell["value"])
    if check is not None:
        check()
    reasons = certify(rows, pinned, totals, axes, summary_amounts)
    # Per-row arithmetic, only once the amount column itself has survived document
    # reconciliation: a line whose printed amount disagrees with quantity x unit price cannot
    # carry that pair. The printed amount is kept - a misprint is the document's, not ours.
    for row in rows:
        cells = row["cells"]
        if not {"quantity", "unit_price", "amount_ht"} <= set(cells):
            continue
        product = (Decimal(cells["quantity"]["value"]) * Decimal(cells["unit_price"]["value"])
                   ).quantize(Decimal("0.01"))
        printed = Decimal(cells["amount_ht"]["value"])
        if abs(product - printed) <= TOLERANCE:
            continue
        vetoes.append({
            "band_top": row["band_top"],
            "reason": "printed line amount %s != quantity %s x unit price %s = %s" % (
                printed, cells["quantity"]["value"], cells["unit_price"]["value"], product),
            "printed_amount": str(printed), "calculated_amount": str(product),
            "dropped": ["quantity", "unit_price"],
        })
        # Only the derived pair is dropped: the amount the document printed stays as it is.
        disagreement = vetoes[-1]["reason"]
        reject(row, "quantity", disagreement)
        reject(row, "unit_price", disagreement)

    return {
        "status": "reconstructed",
        "primary_read": primary_key,
        "reads": sorted(reads),
        "bands": len(bands),
        "region_bands": len(rows_band),
        "table_floor": round(table_floor, 4),
        "axes": [{"index": index, "right": round(axis["right"], 4),
                  "span": [round(value, 4) for value in axis_span(axis)],
                  "bands": len({id(band) for band, _, _, _ in axis["members"]})}
                 for index, axis in enumerate(axes)],
        "header_words": None if not header else " ".join(
            word["text"] for word in sorted(header[0], key=lambda item: item["box"][0]))[:90],
        "pinned_roles": dict(pinned),
        "pinning_evidence": evidence,
        "certification_rejections": reasons,
        "row_arithmetic_vetoes": vetoes,
        "summary_amounts": {str(index): str(value) for index, value in summary_amounts.items()},
        "labelled_totals": [
            {"kind": total["kind"], "label": total["label"], "value": str(total["value"]),
             "families": total["families"], "top": round(total["top"], 4)} for total in totals],
        "description_window": None if window is None else [
            round(window[0], 4), round(window[1], 4), "rows %d/%d" % (window[2], window[3])],
        "rows": rows,
    }

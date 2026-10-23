"""Bounded query grammar with registered entity references and legacy fixture aliases."""

from __future__ import annotations

import hashlib
import math
import re

from video_intelligence.detection.prompts import PromptRegistry
from video_intelligence.retrieval.schemas import ParseStatus, QuerySpec

_ALIASES = {
    "red_square": (
        "red square",
        "red object",
        "red",
        "красный квадрат",
        "красный объект",
        "красным объектом",
    ),
    "blue_circle": (
        "blue circle",
        "blue object",
        "blue",
        "синий круг",
        "синий объект",
        "синим объектом",
    ),
    "green_object": (
        "green object",
        "green",
        "зелёный объект",
        "зеленый объект",
    ),
}
_RELATIONS = {
    "near": "near",
    "рядом с": "near",
    "близко к": "near",
    "left of": "left",
    "слева от": "left",
    "right of": "right",
    "справа от": "right",
    "overlapping": "overlap",
    "overlap": "overlap",
    "пересекается с": "overlap",
}
_ENTITY_BY_ALIAS = {
    alias.replace("ё", "е"): key for key, aliases in _ALIASES.items() for alias in aliases
}
_ENTITY = (
    "(?:"
    + "|".join(re.escape(alias) for alias in sorted(_ENTITY_BY_ALIAS, key=len, reverse=True))
    + ")"
)
_NUMBER = r"\d+(?:\.\d+)?"
_UNIT = r"seconds?|minutes?|секунд(?:а|ы)?|минут(?:а|ы)?"


def _normalise(raw_text: str) -> str:
    # Do not erase minus signs, operators or unknown punctuation from user intent.
    value = re.sub(r"\s+", " ", raw_text.lower().replace("ё", "е")).strip()
    return re.sub(r"[?!.]+$", "", value).strip()


def _seconds(number: str, unit: str | None) -> float:
    value = float(number) * (60 if unit and unit.startswith(("min", "мин")) else 1)
    if not math.isfinite(value):
        raise ValueError("time must be finite")
    return value


def _time_constraints(text: str) -> tuple[str, float | None, float | None]:
    marker = re.search(r"\s(?:after|before|between|после|до|между)\s", text)
    if marker is None:
        return text, None, None
    base, suffix = text[: marker.start()], text[marker.start() :].strip()
    between = re.fullmatch(
        rf"(?:between|между) ({_NUMBER}) (?:and|и) ({_NUMBER})"
        rf"(?: ({_UNIT}))?",
        suffix,
    )
    if between:
        start = _seconds(between[1], between[3])
        end = _seconds(between[2], between[3])
    else:
        limits = {}
        while suffix:
            bound = re.match(rf"(after|before|после|до) ({_NUMBER})(?: ({_UNIT}))?(?= |$)", suffix)
            if bound is None:
                raise ValueError("unsupported time constraint")
            name = "start" if bound[1] in {"after", "после"} else "end"
            if name in limits:
                raise ValueError("duplicate time constraint")
            limits[name] = _seconds(bound[2], bound[3])
            suffix = suffix[bound.end() :].strip()
        start, end = limits.get("start"), limits.get("end")
    if start is not None and end is not None and start > end:
        raise ValueError("reversed time bounds")
    return base, start, end


def parse_query(
    raw_text: str, *, query_namespace: str | None = None, registry: PromptRegistry | None = None,
) -> QuerySpec:
    """Parse only explicit grammar; unsupported input is never silently truncated."""

    if not isinstance(raw_text, str) or not raw_text.strip() or len(raw_text) > 16_384:
        raise ValueError("query must contain 1..16384 characters")
    entity_by_alias = dict(_ENTITY_BY_ALIAS)
    if registry is not None:
        if not isinstance(registry, PromptRegistry):
            raise TypeError("registry must be a PromptRegistry")
        ids = tuple(registry.to_dict()["specs"])
        if not 1 <= len(ids) <= 64 or any(
            re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", entity) is None for entity in ids
        ):
            raise ValueError("query registry requires 1..64 lowercase portable entity ids")
        entity_by_alias = {
            alias: entity for alias, entity in entity_by_alias.items() if entity in ids
        }
        entity_by_alias.update({"@" + entity: entity for entity in ids})
    entity_pattern = "(?:" + "|".join(
        re.escape(alias) for alias in sorted(entity_by_alias, key=len, reverse=True)
    ) + ")"
    identity = f"{query_namespace}|{raw_text}" if query_namespace is not None else raw_text
    query_id = f"q_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"
    text = _normalise(raw_text.strip())
    language = "ru" if re.search(r"[а-я]", text) else "en"
    try:
        text, start, end = _time_constraints(text)
    except ValueError:
        return QuerySpec(
            query_id,
            raw_text.strip(),
            language,
            ParseStatus.UNSUPPORTED_QUERY,
            reason="unsupported or conflicting time constraints",
        )
    relation_pattern = "|".join(re.escape(value) for value in _RELATIONS)
    relation = re.fullmatch(
        rf"(?:show|покажи) ({entity_pattern}) ({relation_pattern}) ({entity_pattern})", text,
    )
    if relation:
        if entity_by_alias[relation[1]] == entity_by_alias[relation[3]]:
            return QuerySpec(
                query_id,
                raw_text.strip(),
                language,
                ParseStatus.UNSUPPORTED_QUERY,
                reason="same-entity relations require instance disambiguation",
            )
        return QuerySpec(
            query_id,
            raw_text.strip(),
            language,
            ParseStatus.SUPPORTED,
            entity=entity_by_alias[relation[1]],
            secondary_entity=entity_by_alias[relation[3]],
            relation=_RELATIONS[relation[2]],
            answer_type="evidence",
            time_start_seconds=start,
            time_end_seconds=end,
        )
    entity = temporal = None
    show = re.fullmatch(rf"(?:show|покажи) ({entity_pattern})", text)
    when = re.fullmatch(rf"when does (?:the )?({entity_pattern})(?: (first|last))? appear", text)
    temporal_show = re.fullmatch(rf"show (first|last) ({entity_pattern})", text)
    russian_when = re.fullmatch(rf"когда (?:впервые )?появляется ({entity_pattern})", text)
    if show:
        entity = show[1]
    elif when:
        entity, temporal = when[1], when[2] or "first"
    elif temporal_show:
        entity, temporal = temporal_show[2], temporal_show[1]
    elif russian_when:
        entity, temporal = russian_when[1], "first"
    if entity is not None:
        return QuerySpec(
            query_id,
            raw_text.strip(),
            language,
            ParseStatus.SUPPORTED,
            entity=entity_by_alias[entity],
            temporal=temporal,
            answer_type="time" if temporal is not None else "evidence",
            time_start_seconds=start,
            time_end_seconds=end,
        )
    return QuerySpec(
        query_id,
        raw_text.strip(),
        language,
        ParseStatus.UNSUPPORTED_QUERY,
        reason="query grammar supports one entity or an explicit spatial relation",
    )

"""Immutable, explicit phrase expansion for open-vocabulary detector prompts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} must be a non-empty identifier")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty trimmed string")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{name} must not contain line breaks")
    return value


def _raw_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _phrases(value: object, name: str, *, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of phrases")
    result = tuple(_text(item, name) for item in value)
    if not allow_empty and not result:
        raise ValueError(f"{name} must contain at least one phrase")
    folded = tuple(item.casefold() for item in result)
    if len(set(folded)) != len(folded):
        raise ValueError(f"{name} must not contain duplicate phrases")
    return result


@dataclass(frozen=True, slots=True)
class PhraseSpec:
    """One canonical entity and its explicitly approved positive/negative phrases."""

    entity_id: str
    positive_phrases: tuple[str, ...]
    negative_phrases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _identifier(self.entity_id, "entity_id"))
        positive = _phrases(
            self.positive_phrases,
            "positive_phrases",
            allow_empty=False,
        )
        negative = _phrases(
            self.negative_phrases,
            "negative_phrases",
            allow_empty=True,
        )
        if set(item.casefold() for item in positive) & {item.casefold() for item in negative}:
            raise ValueError("positive and negative phrases must be disjoint")
        object.__setattr__(self, "positive_phrases", positive)
        object.__setattr__(self, "negative_phrases", negative)


@dataclass(frozen=True, slots=True)
class PromptPlan:
    """A deterministic prompt expansion with raw query preserved verbatim."""

    raw_query: str
    entity_ids: tuple[str, ...]
    positive_phrases: tuple[str, ...]
    phrase_entity_ids: tuple[str, ...]
    negative_phrases: tuple[str, ...]
    prompt_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_query", _raw_text(self.raw_query, "raw_query"))
        if isinstance(self.entity_ids, (str, bytes)) or not isinstance(self.entity_ids, Sequence):
            raise ValueError("entity_ids must be a sequence")
        entity_ids = tuple(_identifier(item, "entity_id") for item in self.entity_ids)
        if not entity_ids:
            raise ValueError("entity_ids must contain at least one entity")
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError("entity_ids must be unique")
        object.__setattr__(self, "entity_ids", entity_ids)
        positive = _phrases(
            self.positive_phrases,
            "positive_phrases",
            allow_empty=False,
        )
        negative = _phrases(
            self.negative_phrases,
            "negative_phrases",
            allow_empty=True,
        )
        if set(item.casefold() for item in positive) & {item.casefold() for item in negative}:
            raise ValueError("positive and negative phrases must be disjoint")
        object.__setattr__(self, "positive_phrases", positive)
        if isinstance(self.phrase_entity_ids, (str, bytes)) or not isinstance(
            self.phrase_entity_ids, Sequence
        ):
            raise ValueError("phrase_entity_ids must be a sequence")
        phrase_entity_ids = tuple(
            _identifier(item, "phrase_entity_id") for item in self.phrase_entity_ids
        )
        if len(phrase_entity_ids) != len(positive):
            raise ValueError("phrase_entity_ids must align with positive_phrases")
        if any(entity_id not in entity_ids for entity_id in phrase_entity_ids):
            raise ValueError("phrase_entity_ids must refer to entity_ids")
        object.__setattr__(self, "phrase_entity_ids", phrase_entity_ids)
        object.__setattr__(self, "negative_phrases", negative)
        object.__setattr__(
            self,
            "prompt_version",
            _identifier(self.prompt_version, "prompt_version"),
        )

    @property
    def rendered_prompt(self) -> str:
        return " . ".join(self.positive_phrases) + " ."

    @property
    def rendered_negative_prompt(self) -> str | None:
        if not self.negative_phrases:
            return None
        return " . ".join(self.negative_phrases) + " ."

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_query": self.raw_query,
            "entity_ids": list(self.entity_ids),
            "positive_phrases": list(self.positive_phrases),
            "phrase_entity_ids": list(self.phrase_entity_ids),
            "negative_phrases": list(self.negative_phrases),
            "prompt_version": self.prompt_version,
            "rendered_prompt": self.rendered_prompt,
            "rendered_negative_prompt": self.rendered_negative_prompt,
        }


class PromptRegistry:
    """Read-only phrase registry; prompt text is generated only from registered data."""

    def __init__(
        self,
        specs: Sequence[PhraseSpec],
        *,
        prompt_version: str = "prompt_v1",
    ) -> None:
        if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
            raise ValueError("specs must be a sequence of PhraseSpec values")
        values = tuple(specs)
        if any(not isinstance(spec, PhraseSpec) for spec in values):
            raise ValueError("specs must contain only PhraseSpec values")
        if len({spec.entity_id for spec in values}) != len(values):
            raise ValueError("entity_id values must be unique")
        self._specs = {spec.entity_id: spec for spec in values}
        self._prompt_version = _identifier(prompt_version, "prompt_version")

    def get(self, entity_id: str) -> PhraseSpec:
        entity_id = _identifier(entity_id, "entity_id")
        try:
            return self._specs[entity_id]
        except KeyError as error:
            raise KeyError(f"entity is not registered: {entity_id}") from error

    def plan(self, raw_query: str, entity_ids: Sequence[str]) -> PromptPlan:
        if isinstance(entity_ids, (str, bytes)) or not isinstance(entity_ids, Sequence):
            raise ValueError("entity_ids must be a sequence")
        ids = tuple(_identifier(item, "entity_id") for item in entity_ids)
        specs = tuple(self.get(entity_id) for entity_id in ids)
        positive_pairs = tuple(
            (spec.entity_id, phrase) for spec in specs for phrase in spec.positive_phrases
        )
        positive_entity_ids = tuple(entity_id for entity_id, _phrase in positive_pairs)
        positive = tuple(phrase for _entity_id, phrase in positive_pairs)
        negative = tuple(phrase for spec in specs for phrase in spec.negative_phrases)
        if len({item.casefold() for item in positive}) != len(positive):
            raise ValueError("positive phrases must be unique across entities")
        if len({item.casefold() for item in negative}) != len(negative):
            raise ValueError("negative phrases must be unique across entities")
        return PromptPlan(
            raw_query=raw_query,
            entity_ids=ids,
            positive_phrases=positive,
            phrase_entity_ids=positive_entity_ids,
            negative_phrases=negative,
            prompt_version=self._prompt_version,
        )

    def to_dict(self) -> dict[str, object]:
        payload = {
            "prompt_version": self._prompt_version,
            "specs": {
                entity_id: {
                    "positive_phrases": list(spec.positive_phrases),
                    "negative_phrases": list(spec.negative_phrases),
                }
                for entity_id, spec in sorted(self._specs.items())
            },
        }
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
        return payload

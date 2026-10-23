"""Bounded preflight for the first two trusted, flat local HF bundles; no ML imports.

Directory hash and upstream provenance must already be independently reviewed.
This rejects dispatch overrides; it does not certify arbitrary model files as safe.
"""

from __future__ import annotations

import json
import math
import re
import stat
from pathlib import Path

_FILES = frozenset({
    "config.json", "preprocessor_config.json", "processor_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.txt",
    "vocab.json", "merges.txt", "tokenizer.model", "model.safetensors",
    "model.safetensors.index.json", "README.md", "LICENSE", "LICENSE.txt", ".gitattributes",
})
_BLOCKED = frozenset({
    "auto_map", "custom_pipelines", "custom_generate", "trust_remote_code", "quantization_config",
    "adapter_config", "peft_type", "base_model_name_or_path", "attn_implementation",
    "experts_implementation", "use_kernels", "allow_all_kernels", "kernel_config",
    "backbone", "backbone_kwargs",
})
_TYPES = {
    "siglip": {"siglip", "siglip_text_model", "siglip_vision_model"},
    "grounding-dino": {"grounding-dino", "swin", "bert"},
}
_CLASSES = {
    "siglip": {"SiglipProcessor", "SiglipImageProcessor", "SiglipImageProcessorPil",
               "GemmaTokenizer", "GemmaTokenizerFast"},
    "grounding-dino": {"GroundingDinoProcessor", "GroundingDinoImageProcessor",
                       "GroundingDinoImageProcessorPil", "BertTokenizer", "BertTokenizerFast"},
}
# The reviewed SigLIP2 Gemma tokenizer is 34,363,039 bytes. Other config files
# retain their small cap; this is not permission to load arbitrary bundles.
_JSON_LIMITS = {
    "siglip": (40 * 1024**2, 48 * 1024**2),
    "grounding-dino": (16 * 1024**2, 20 * 1024**2),
}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate bundle JSON key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("nonfinite bundle JSON number")


def _inspect(value, model_type, depth=0, *, dispatch=True, tokenizer=False, dino_config=False,
             path=()):
    if depth > 24:
        raise ValueError("bundle JSON nesting exceeds policy")
    if isinstance(value, dict):
        for key, child in value.items():
            # Official DINO emits these inactive optional selectors as root nulls.
            inert_dino_default = (dino_config and model_type == "grounding-dino" and not path
                                 and key in {"backbone", "backbone_kwargs"} and child is None)
            if dispatch and (key in _BLOCKED or key.startswith(("_attn", "_experts", "adapter_"))
                    or key in {"use_pretrained_backbone", "use_timm_backbone"}
                    and child is not False) and not inert_dino_default:
                raise ValueError("bundle dispatch override is forbidden")
            if dispatch and key == "model_type" and (
                    not isinstance(child, str) or child not in _TYPES[model_type]):
                raise ValueError("unexpected nested model_type")
            if dispatch and key in {"processor_class", "image_processor_type", "tokenizer_class"}:
                if not isinstance(child, str) or child not in _CLASSES[model_type]:
                    raise ValueError("unexpected processor/tokenizer class")
            _inspect(child, model_type, depth + 1, dispatch=dispatch, tokenizer=tokenizer,
                     dino_config=dino_config, path=(*path, key))
    elif isinstance(value, list):
        # The reviewed SigLIP2 BPE tokenizer has 580,604 merge pairs. Relax only
        # this data field; every other list and all DINO files keep the old cap.
        cap = (600_000 if tokenizer and model_type == "siglip"
               and path == ("model", "merges") else 300_000)
        if len(value) > cap:
            raise ValueError("bundle JSON list exceeds policy")
        for child in value:
            _inspect(child, model_type, depth + 1, dispatch=dispatch, tokenizer=tokenizer,
                     dino_config=dino_config, path=(*path, None))
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite bundle JSON number")


def validate_local_bundle(location, *, model_type):
    """Fail before any processor/model loader. Initial policy intentionally rejects subfolders."""
    if model_type not in _TYPES:
        raise ValueError("unsupported bundle policy")
    root = Path(location).expanduser().absolute()
    # Do not resolve first: Windows junctions/symlinks must remain observable.
    for path in (root, *root.parents):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("bundle path contains a link/reparse point")
    if not root.is_dir():
        raise ValueError("bundle must be a directory")
    members = []
    for member in root.iterdir():
        members.append(member)
        if len(members) > 64:
            raise ValueError("bundle file count exceeds policy")
    if not 4 <= len(members) <= 64:
        raise ValueError("bundle file count exceeds policy")
    documents = {}
    json_bytes = 0
    tokenizer_cap, total_cap = _JSON_LIMITS[model_type]
    for member in members:
        info = member.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or getattr(info, "st_file_attributes", 0) & 0x400):
            raise ValueError("bundle member is not an ordinary single-link file")
        if (member.name not in _FILES
                and re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", member.name) is None):
            raise ValueError("unexpected bundle member")
        if member.suffix == ".json":
            cap = tokenizer_cap if member.name == "tokenizer.json" else 256 * 1024
            if info.st_size > cap:
                raise ValueError("bundle JSON byte cap exceeded")
            with member.open("rb") as stream:
                raw = stream.read(cap + 1)
            json_bytes += len(raw)
            if len(raw) > cap or json_bytes > total_cap:
                raise ValueError("bundle JSON byte cap exceeded")
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                               parse_constant=_nonfinite)
            if not isinstance(value, dict):
                raise ValueError("bundle JSON must be an object")
            # Vocabulary keys are token data, not HF configuration attributes.
            data_files = {"tokenizer.json", "vocab.json", "added_tokens.json",
                          "special_tokens_map.json", "model.safetensors.index.json"}
            _inspect(value, model_type, dispatch=member.name not in data_files,
                     tokenizer=member.name == "tokenizer.json",
                     dino_config=member.name == "config.json")
            documents[member.name] = value
    if not {"config.json", "preprocessor_config.json", "tokenizer_config.json"} <= documents.keys():
        raise ValueError("bundle lacks required processor/model config")
    names = {p.name for p in members}
    if "model.safetensors.index.json" in documents or "model.safetensors" not in names:
        index = documents.get("model.safetensors.index.json", {})
        weights = index.get("weight_map")
        if (not isinstance(weights, dict) or not weights
                or any(not isinstance(name, str) or name not in names
                       or re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name) is None
                       for name in weights.values())):
            raise ValueError("bundle has no local safetensors inventory")
    config = documents["config.json"]
    architecture = "SiglipModel" if model_type == "siglip" else "GroundingDinoForObjectDetection"
    # Explicit loader classes do not use architecture dispatch. The reviewed
    # upstream SigLIP2 config omits architectures and default geometry fields.
    architectures = config.get("architectures", [architecture] if model_type == "siglip" else None)
    if config.get("model_type") != model_type or architectures != [architecture]:
        raise ValueError("bundle architecture differs from policy")
    if model_type == "siglip":
        vision, text = config.get("vision_config", {}), config.get("text_config", {})
        if (not isinstance(vision, dict) or not isinstance(text, dict)
                or vision.get("model_type") != "siglip_vision_model"
                or text.get("model_type") != "siglip_text_model"
                or vision.get("image_size", 224) != 224 or vision.get("hidden_size", 768) != 768
                or text.get("max_position_embeddings", 64) != 64):
            raise ValueError("bundle SigLIP geometry differs from policy")
    elif (config.get("backbone_config", {}).get("model_type") != "swin"
          or config.get("text_config", {}).get("model_type") != "bert"):
        raise ValueError("bundle DINO backbone/text differs from policy")
    return None

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import httpx
import numpy as np

from regiomanip.env_utils import load_workspace_env
from regiomanip.region_grounding.utils import (
    build_mask_geometry_stats,
    canonical_part_name,
    clean_mask_ids,
    colorize_depth,
    compose_tool_whole,
    crop_from_masks,
    encode_image_to_data_url,
    extract_json_block,
    _keep_significant_components,
    keep_largest_cc,
    load_json,
    mask_area,
    mask_bbox_xyxy,
    mask_iou,
    save_json,
    union_masks,
)

load_workspace_env()

TOOL_INTENT_HINTS = {
    "grip",
    "grasp",
    "grab",
    "hold",
    "pick up",
    "pickup",
    "lift",
    "move",
    "place",
    "put",
    "pull",
    "push",
    "use",
    "scrape",
    "roll",
    "hammer",
    "tighten",
    "loosen",
    "screw",
    "unscrew",
    "drive",
    "paint",
    "brush",
    "cut",
    "slice",
    "pry",
    "turn",
    "twist",
    "insert",
}
MODEL_CFG = {
    "vlm_model": os.environ.get("VLM_MODEL", "gpt-4o"),
    "openai_base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "openai_api_key": os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE"),
    "openai_ignore_env_proxy": os.environ.get("OPENAI_IGNORE_ENV_PROXY", "1").strip().lower() not in {"0", "false", "no"},
    "openai_timeout_s": float(os.environ.get("OPENAI_TIMEOUT_S", "120")),
}

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore[misc,assignment]


def classify_openai_failure(exc: Exception) -> Dict[str, Any]:
    msg = str(exc) if exc is not None else ""
    msg_l = msg.lower()

    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        resp = getattr(exc, "response", None)
        status_code = getattr(resp, "status_code", None) if resp is not None else None
    try:
        status_code = int(status_code) if status_code is not None else None
    except Exception:
        status_code = None

    category = "unknown"
    hint = "Unexpected API error. Check provider logs and request payload."

    network_keys = [
        "connection error",
        "couldn't connect",
        "could not resolve host",
        "failed to establish a new connection",
        "maxretryerror",
        "proxyerror",
        "timed out",
        "timeout",
        "temporary failure in name resolution",
        "network is unreachable",
    ]
    if any(k in msg_l for k in network_keys):
        category = "network"
        hint = "Network/proxy/DNS issue. Verify internet, proxy settings, and OPENAI_BASE_URL reachability."
    elif status_code == 401 or "invalid api key" in msg_l or "incorrect api key" in msg_l or "unauthorized" in msg_l:
        category = "invalid_api_key"
        hint = "API key is invalid or missing permission. Check OPENAI_API_KEY."
    elif status_code == 429 and ("quota" in msg_l or "insufficient_quota" in msg_l or "billing" in msg_l or "credit" in msg_l):
        category = "quota_or_billing"
        hint = "Quota/billing exhausted. Add credits or upgrade billing plan."
    elif status_code == 429:
        category = "rate_limit"
        hint = "Rate-limited. Reduce request frequency or increase provider limits."
    elif status_code == 403 or "forbidden" in msg_l or "permission" in msg_l:
        category = "permission_denied"
        hint = "API key lacks required permissions for this endpoint/model."
    elif (status_code == 404 and "model" in msg_l) or "model_not_found" in msg_l:
        category = "model_not_found"
        hint = "Model name is unavailable for this provider. Check VLM_MODEL."
    elif status_code == 400 or "invalid_request_error" in msg_l:
        category = "bad_request"
        hint = "Request payload/model parameters are invalid for the provider."
    elif status_code is not None and status_code >= 500:
        category = "provider_server_error"
        hint = "Provider backend error. Retry later or switch provider endpoint."

    return {
        "category": category,
        "status_code": status_code,
        "message": msg,
        "hint": hint,
    }


def _safe_http_proxy_mounts() -> Optional[Dict[str, httpx.HTTPTransport]]:
    """Return httpx mounts built from HTTP(S)_PROXY only, skipping SOCKS entries.

    ALL_PROXY with a socks:// scheme causes httpx to fail unless socksio is
    installed.  We read only HTTP_PROXY / HTTPS_PROXY (both are plain-HTTP
    transports) so the API call can still go through the local proxy without
    requiring the SOCKS dependency.
    """
    https_proxy = (
        os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy", "")
    ).strip()
    http_proxy = (
        os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy", "")
    ).strip()

    mounts: Dict[str, httpx.HTTPTransport] = {}
    if https_proxy and https_proxy.startswith("http"):
        mounts["https://"] = httpx.HTTPTransport(proxy=https_proxy)
    if http_proxy and http_proxy.startswith("http"):
        mounts["http://"] = httpx.HTTPTransport(proxy=http_proxy)
    return mounts if mounts else None


def _build_openai_client_kwargs(trust_env_override: Optional[bool] = None) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "api_key": MODEL_CFG["openai_api_key"],
        "timeout": MODEL_CFG["openai_timeout_s"],
    }
    if MODEL_CFG["openai_base_url"]:
        kwargs["base_url"] = MODEL_CFG["openai_base_url"]

    if trust_env_override is None:
        if MODEL_CFG["openai_ignore_env_proxy"]:
            kwargs["http_client"] = httpx.Client(trust_env=False, timeout=MODEL_CFG["openai_timeout_s"])
        else:
            # trust_env=True would pick up ALL_PROXY (socks://) which httpx
            # cannot handle without socksio.  Build explicit mounts from the
            # HTTP(S) proxy vars only so SOCKS entries are silently skipped.
            mounts = _safe_http_proxy_mounts()
            # trust_env=False suppresses ALL_PROXY (socks://); explicit mounts
            # re-add the HTTP proxy so the API is still reachable.
            kwargs["http_client"] = httpx.Client(
                trust_env=False,
                mounts=mounts or {},
                timeout=MODEL_CFG["openai_timeout_s"],
            )
        return kwargs

    kwargs["http_client"] = httpx.Client(trust_env=bool(trust_env_override), timeout=MODEL_CFG["openai_timeout_s"])
    return kwargs


def build_openai_client_with_status(trust_env_override: Optional[bool] = None) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    if not OPENAI_AVAILABLE:
        return None, {
            "category": "openai_library_missing",
            "message": "openai python package is not available in this environment.",
            "hint": "Install/enable the openai client library.",
        }
    if not MODEL_CFG["openai_api_key"] or MODEL_CFG["openai_api_key"] == "YOUR_API_KEY_HERE":
        return None, {
            "category": "missing_api_key",
            "message": "OPENAI_API_KEY is empty or placeholder.",
            "hint": "Set a valid OPENAI_API_KEY.",
        }

    kwargs = _build_openai_client_kwargs(trust_env_override=trust_env_override)
    try:
        return OpenAI(**kwargs), None
    except Exception as exc:
        return None, classify_openai_failure(exc)


def _normalize_selection_dict(parsed: Any, n_masks: int, tool_mode: bool) -> Optional[Dict[str, List[int]]]:
    if not isinstance(parsed, dict):
        return None

    out: Dict[str, List[int]] = {
        "whole_ids": [],
        "handle_ids": [],
        "work_end_ids": [],
        "contact_region_ids": [],
    }

    key_map = {
        "whole_ids": ["whole_ids", "whole", "whole_mask_ids", "object_ids"],
        "handle_ids": ["handle_ids", "handle", "grip_ids", "grip_part_ids"],
        "work_end_ids": ["work_end_ids", "work_end", "tip_ids", "end_ids", "blade_ids"],
        "contact_region_ids": ["contact_region_ids", "contact_ids", "contact_region", "contact_area_ids"],
    }
    for dst, keys in key_map.items():
        for k in keys:
            if k in parsed and not out[dst]:
                out[dst] = clean_mask_ids(parsed.get(k), n_masks)

    parts = parsed.get("parts")
    if isinstance(parts, list):
        for p in parts:
            if not isinstance(p, dict):
                continue
            pname = canonical_part_name(str(p.get("name", p.get("part", ""))))
            ids = clean_mask_ids(p.get("mask_ids", p.get("ids", p.get("id"))), n_masks)
            if not ids:
                continue
            if pname == "whole":
                out["whole_ids"] = sorted(set(out["whole_ids"] + ids))
            elif pname == "handle":
                out["handle_ids"] = sorted(set(out["handle_ids"] + ids))
            elif pname == "work_end":
                out["work_end_ids"] = sorted(set(out["work_end_ids"] + ids))
            elif pname == "contact_region":
                out["contact_region_ids"] = sorted(set(out["contact_region_ids"] + ids))

    out = _repair_simple_tool_selection(out, n_masks=n_masks, tool_mode=tool_mode)

    if tool_mode:
        if not out["handle_ids"] or not out["work_end_ids"]:
            return None
    else:
        if not out["whole_ids"] and not out["contact_region_ids"]:
            return None
    return out


def _repair_simple_tool_selection(
    parsed_ids: Dict[str, List[int]],
    n_masks: int,
    tool_mode: bool,
) -> Dict[str, List[int]]:
    if not tool_mode or n_masks != 2:
        return parsed_ids

    out: Dict[str, List[int]] = {
        "whole_ids": sorted(set(int(x) for x in parsed_ids.get("whole_ids", []))),
        "handle_ids": sorted(set(int(x) for x in parsed_ids.get("handle_ids", []))),
        "work_end_ids": sorted(set(int(x) for x in parsed_ids.get("work_end_ids", []))),
        "contact_region_ids": sorted(set(int(x) for x in parsed_ids.get("contact_region_ids", []))),
    }

    valid_ids = [0, 1]
    if out["handle_ids"] and not out["work_end_ids"]:
        remaining = [i for i in valid_ids if i not in set(out["handle_ids"])]
        if len(remaining) == 1:
            out["work_end_ids"] = remaining
    elif out["work_end_ids"] and not out["handle_ids"]:
        remaining = [i for i in valid_ids if i not in set(out["work_end_ids"])]
        if len(remaining) == 1:
            out["handle_ids"] = remaining

    return out


def _parse_vlm_mask_selection(raw_text: str, n_masks: int, tool_mode: bool) -> Optional[Dict[str, List[int]]]:
    parsed = extract_json_block(raw_text)
    if parsed is None:
        return None

    if isinstance(parsed, dict):
        return _normalize_selection_dict(parsed, n_masks=n_masks, tool_mode=tool_mode)
    if isinstance(parsed, list):
        return _normalize_selection_dict({"parts": parsed}, n_masks=n_masks, tool_mode=tool_mode)
    return None


def _part_candidate_stats(mask_stats: List[Dict[str, Any]], whole_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    whole_set = set(int(x) for x in (whole_ids or []))
    out: List[Dict[str, Any]] = []
    for stat in mask_stats:
        try:
            mid = int(stat.get("id"))
        except Exception:
            continue
        area_ratio = float(stat.get("area_ratio_vs_support", 0.0) or 0.0)
        if mid in whole_set or area_ratio <= 0.0:
            continue
        if area_ratio > 0.92:
            continue
        out.append(stat)
    if not out and mask_stats:
        whole_id = max(mask_stats, key=lambda s: float(s.get("area_ratio_vs_support", 0.0) or 0.0)).get("id")
        try:
            whole_id = int(whole_id)
        except Exception:
            whole_id = -1
        out = [s for s in mask_stats if int(s.get("id", -1)) != whole_id]
    return out


def _infer_tool_role_family(
    mask_stats: List[Dict[str, Any]],
    *,
    object_name: str = "",
    instruction: str = "",
) -> Tuple[str, Dict[str, Any]]:
    family = "generic"
    source = "geometry_default"
    part_stats = _part_candidate_stats(mask_stats)
    if not part_stats:
        return family, {"family": family, "source": source}

    proj_vals = [float(s.get("axis_projection", 0.0) or 0.0) for s in part_stats]
    proj_span = max(max(proj_vals) - min(proj_vals), 1.0) if proj_vals else 1.0

    def _terminal_score(stat: Dict[str, Any]) -> float:
        return abs(float(stat.get("axis_projection", 0.0) or 0.0)) / proj_span

    max_area = max(float(s.get("area_ratio_vs_support", 0.0) or 0.0) for s in part_stats)
    compact_terminal_count = sum(
        1
        for s in part_stats
        if float(s.get("elongation", 1.0) or 1.0) < 2.6
        and float(s.get("area_ratio_vs_support", 0.0) or 0.0) > 0.12
        and _terminal_score(s) > 0.42
    )
    elongated_terminal_count = sum(
        1
        for s in part_stats
        if float(s.get("elongation", 1.0) or 1.0) > 4.2 and _terminal_score(s) > 0.42
    )
    slim_terminal_present = any(
        float(s.get("elongation", 1.0) or 1.0) > 3.0
        and float(s.get("area_ratio_vs_support", 0.0) or 0.0) < 0.22
        and _terminal_score(s) > 0.55
        for s in part_stats
    )
    large_compact_present = any(
        float(s.get("elongation", 1.0) or 1.0) < 2.8 and float(s.get("area_ratio_vs_support", 0.0) or 0.0) > 0.20
        for s in part_stats
    )

    if elongated_terminal_count >= 1 and large_compact_present and max_area > 0.18:
        family = "head_dominant"
        source = "geometry"
    elif (elongated_terminal_count >= 2 and compact_terminal_count == 0 and max_area < 0.30) or (
        slim_terminal_present and compact_terminal_count == 0 and max_area < 0.65
    ):
        family = "axial"
        source = "geometry"
    elif compact_terminal_count >= 2:
        family = "multi_terminal"
        source = "geometry"

    return family, {
        "family": family,
        "source": source,
        "compact_terminal_count": int(compact_terminal_count),
        "elongated_terminal_count": int(elongated_terminal_count),
        "slim_terminal_present": bool(slim_terminal_present),
        "max_area_ratio": float(max_area),
    }


def _infer_effective_tool_mode(
    object_name: str,
    instruction: str,
    mask_stats: List[Dict[str, Any]],
    declared_tool_mode: bool,
) -> Tuple[bool, Dict[str, Any]]:
    if declared_tool_mode:
        return True, {"declared": True, "inferred": False, "reason": "declared_tool_mode"}

    ins_l = (instruction or "").lower()
    intent = any(h in ins_l for h in TOOL_INTENT_HINTS)
    part_stats = _part_candidate_stats(mask_stats)
    max_elong = max((float(s.get("elongation", 0.0) or 0.0) for s in part_stats), default=0.0)
    projection_vals = [float(s.get("axis_projection", 0.0) or 0.0) for s in part_stats]
    proj_span = (max(projection_vals) - min(projection_vals)) if len(projection_vals) >= 2 else 0.0
    structured = len(part_stats) >= 2 and (max_elong >= 3.6 or proj_span >= 60.0)
    inferred = bool(intent and structured)
    return inferred, {
        "declared": bool(declared_tool_mode),
        "inferred": bool(inferred),
        "reason": "instruction_plus_geometry" if inferred else "not_tool_like_enough",
        "intent_detected": bool(intent),
        "num_part_candidates": len(part_stats),
        "max_elongation": float(max_elong),
        "axis_projection_span": float(proj_span),
    }


def _repair_tool_role_selection(
    parsed_ids: Dict[str, List[int]],
    *,
    object_name: str,
    instruction: str,
    tool_mode: bool,
    mask_stats: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    out: Dict[str, List[int]] = {
        "whole_ids": sorted(set(int(x) for x in parsed_ids.get("whole_ids", []))),
        "handle_ids": sorted(set(int(x) for x in parsed_ids.get("handle_ids", []))),
        "work_end_ids": sorted(set(int(x) for x in parsed_ids.get("work_end_ids", []))),
        "contact_region_ids": sorted(set(int(x) for x in parsed_ids.get("contact_region_ids", []))),
    }
    if not tool_mode:
        return out, {}

    family, family_dbg = _infer_tool_role_family(mask_stats=mask_stats)
    part_stats = _part_candidate_stats(mask_stats, whole_ids=out["whole_ids"])
    if len(part_stats) < 2:
        return out, {"kind": "tool_role_pair_scoring", "family": family, "reason": "not_enough_part_candidates"}

    proj_vals = [float(s.get("axis_projection", 0.0) or 0.0) for s in part_stats]
    proj_span = max(max(proj_vals) - min(proj_vals), 1.0)
    max_area = max(float(s.get("area_ratio_vs_support", 0.0) or 0.0) for s in part_stats)
    max_diag = max(float(s.get("bbox_diag", 0.0) or 0.0) for s in part_stats)
    max_elong = max(float(s.get("elongation", 1.0) or 1.0) for s in part_stats)
    whole_set = set(out["whole_ids"])
    handle_prior = set(out["handle_ids"])
    work_prior = set(out["work_end_ids"])

    def _term_score(stat: Dict[str, Any]) -> float:
        return abs(float(stat.get("axis_projection", 0.0) or 0.0)) / proj_span

    def _area_score(stat: Dict[str, Any]) -> float:
        return float(stat.get("area_ratio_vs_support", 0.0) or 0.0) / max(max_area, 1e-6)

    def _diag_score(stat: Dict[str, Any]) -> float:
        return float(stat.get("bbox_diag", 0.0) or 0.0) / max(max_diag, 1e-6)

    def _elong_score(stat: Dict[str, Any]) -> float:
        return float(stat.get("elongation", 1.0) or 1.0) / max(max_elong, 1e-6)

    def _compact_score(stat: Dict[str, Any]) -> float:
        elong = float(stat.get("elongation", 1.0) or 1.0)
        return 1.0 / max(elong, 1.0)

    def _small_score(stat: Dict[str, Any]) -> float:
        return 1.0 - min(1.0, _area_score(stat))

    def _connector_penalty(stat: Dict[str, Any]) -> float:
        area_ratio = float(stat.get("area_ratio_vs_support", 0.0) or 0.0)
        elong = float(stat.get("elongation", 1.0) or 1.0)
        return 1.0 if area_ratio < 0.12 and elong > 5.2 else 0.0

    def _handle_score(stat: Dict[str, Any]) -> float:
        base = 0.38 * _elong_score(stat) + 0.20 * _diag_score(stat) + 0.22 * _term_score(stat) + 0.10 * _area_score(stat)
        if family in {"head_dominant", "generic"}:
            base += 0.10 * float(stat.get("depth_valid_ratio", 0.0) or 0.0)
        elif family == "axial":
            base += 0.08 * _small_score(stat)
        return base

    def _work_score(stat: Dict[str, Any]) -> float:
        if family == "head_dominant":
            return 0.32 * _compact_score(stat) + 0.28 * _area_score(stat) + 0.25 * _term_score(stat) - 0.30 * _connector_penalty(stat)
        if family == "axial":
            return 0.28 * _elong_score(stat) + 0.26 * _term_score(stat) + 0.22 * _small_score(stat) - 0.18 * _connector_penalty(stat)
        if family == "multi_terminal":
            return 0.26 * _compact_score(stat) + 0.24 * _area_score(stat) + 0.24 * _term_score(stat) - 0.20 * _connector_penalty(stat)
        return 0.26 * _compact_score(stat) + 0.20 * _area_score(stat) + 0.24 * _term_score(stat) + 0.10 * _elong_score(stat) - 0.28 * _connector_penalty(stat)

    best_pair: Optional[Tuple[int, int]] = None
    best_score = -1e18
    best_geom_pair: Optional[Tuple[int, int]] = None
    best_geom_score = -1e18
    scored_pairs: List[Dict[str, Any]] = []
    pair_geom_scores: Dict[Tuple[int, int], float] = {}
    for hs in part_stats:
        hid = int(hs.get("id"))
        if hid in whole_set:
            continue
        for ws in part_stats:
            wid = int(ws.get("id"))
            if wid in whole_set or wid == hid:
                continue
            separation = abs(float(hs.get("axis_projection", 0.0) or 0.0) - float(ws.get("axis_projection", 0.0) or 0.0)) / proj_span
            opposite = 1.0 if float(hs.get("axis_projection", 0.0) or 0.0) * float(ws.get("axis_projection", 0.0) or 0.0) <= 0.0 else 0.0
            geom_score = _handle_score(hs) + _work_score(ws) + 0.34 * separation + 0.12 * opposite
            prior_adjustment = 0.0
            if hid in handle_prior:
                prior_adjustment += 0.30
            if wid in work_prior:
                prior_adjustment += 0.30
            if hid in work_prior:
                prior_adjustment -= 0.18
            if wid in handle_prior:
                prior_adjustment -= 0.18
            score = geom_score + prior_adjustment
            pair_geom_scores[(hid, wid)] = float(geom_score)
            scored_pairs.append(
                {
                    "handle_id": hid,
                    "work_end_id": wid,
                    "score": float(score),
                    "geom_score": float(geom_score),
                    "prior_adjustment": float(prior_adjustment),
                }
            )
            if score > best_score:
                best_score = score
                best_pair = (hid, wid)
            if geom_score > best_geom_score:
                best_geom_score = geom_score
                best_geom_pair = (hid, wid)

    if best_pair is None:
        return out, {"kind": "tool_role_pair_scoring", "family": family, "reason": "no_valid_pair"}

    selected_pair = best_pair
    geometry_override = False
    if best_geom_pair is not None and best_geom_pair != best_pair:
        selected_geom_score = pair_geom_scores.get(best_pair, -1e18)
        # Trust geometry when it clearly disagrees with a prior-biased label assignment.
        if (best_geom_score - selected_geom_score) >= 0.25:
            selected_pair = best_geom_pair
            geometry_override = True

    repaired = dict(out)
    repaired["handle_ids"] = [int(selected_pair[0])]
    repaired["work_end_ids"] = [int(selected_pair[1])]
    changed = repaired["handle_ids"] != out["handle_ids"] or repaired["work_end_ids"] != out["work_end_ids"]
    return repaired, {
        "kind": "tool_role_pair_scoring",
        "family": family,
        "family_debug": family_dbg,
        "before": out,
        "after": repaired,
        "changed": bool(changed),
        "geometry_override": bool(geometry_override),
        "best_total_pair": None if best_pair is None else {"handle_id": int(best_pair[0]), "work_end_id": int(best_pair[1]), "score": float(best_score)},
        "best_geom_pair": None
        if best_geom_pair is None
        else {"handle_id": int(best_geom_pair[0]), "work_end_id": int(best_geom_pair[1]), "geom_score": float(best_geom_score)},
        "top_pairs": sorted(scored_pairs, key=lambda item: float(item["score"]), reverse=True)[:5],
    }


def _union_from_ids(mask_candidates: List[np.ndarray], ids: List[int]) -> np.ndarray:
    if not ids:
        return np.zeros_like(mask_candidates[0], dtype=np.uint8)
    chosen = [mask_candidates[i] for i in ids if 0 <= i < len(mask_candidates)]
    if not chosen:
        return np.zeros_like(mask_candidates[0], dtype=np.uint8)
    return union_masks(chosen)


def _canonical_non_tool_mask(mask01: np.ndarray, rel_min: float = 0.04, pix_min: int = 40) -> np.ndarray:
    return _keep_significant_components((mask01 > 0).astype(np.uint8), rel_min=rel_min, pix_min=pix_min)


def _mask_signature(mask01: np.ndarray) -> Dict[str, Any]:
    m = (mask01 > 0).astype(np.uint8)
    return {
        "area_px": int(mask_area(m)),
        "bbox_xyxy": None if mask_bbox_xyxy(m) is None else [int(v) for v in mask_bbox_xyxy(m)],  # type: ignore[arg-type]
    }


def _cache_matches_masks(cache_obj: Dict[str, Any], masks: List[np.ndarray]) -> bool:
    sigs = cache_obj.get("mask_signatures")
    if not isinstance(sigs, list) or len(sigs) != len(masks):
        return False
    cur = [_mask_signature(m) for m in masks]
    for a, b in zip(sigs, cur):
        if not isinstance(a, dict):
            return False
        if int(a.get("area_px", -1)) != int(b["area_px"]):
            return False
        if a.get("bbox_xyxy") != b["bbox_xyxy"]:
            return False
    return True


def _synthesize_tool_whole_ids(parsed_ids: Dict[str, List[int]], tool_mode: bool) -> Tuple[Dict[str, List[int]], bool]:
    out: Dict[str, List[int]] = {
        "whole_ids": sorted(set(int(x) for x in parsed_ids.get("whole_ids", []))),
        "handle_ids": sorted(set(int(x) for x in parsed_ids.get("handle_ids", []))),
        "work_end_ids": sorted(set(int(x) for x in parsed_ids.get("work_end_ids", []))),
        "contact_region_ids": sorted(set(int(x) for x in parsed_ids.get("contact_region_ids", []))),
    }
    if not tool_mode or out["whole_ids"] or not out["handle_ids"] or not out["work_end_ids"]:
        return out, False
    out["whole_ids"] = sorted(set(out["handle_ids"] + out["work_end_ids"]))
    return out, True


def _normalize_tool_whole_ids(
    parsed_ids: Dict[str, List[int]],
    *,
    tool_mode: bool,
    mask_stats: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    out: Dict[str, List[int]] = {
        "whole_ids": sorted(set(int(x) for x in parsed_ids.get("whole_ids", []))),
        "handle_ids": sorted(set(int(x) for x in parsed_ids.get("handle_ids", []))),
        "work_end_ids": sorted(set(int(x) for x in parsed_ids.get("work_end_ids", []))),
        "contact_region_ids": sorted(set(int(x) for x in parsed_ids.get("contact_region_ids", []))),
    }
    if not tool_mode or len(out["whole_ids"]) <= 1 or not mask_stats:
        return out, {}

    area_by_id: Dict[int, float] = {}
    for stat in mask_stats:
        try:
            area_by_id[int(stat.get("id"))] = float(stat.get("area_ratio_vs_support", 0.0) or 0.0)
        except Exception:
            continue
    candidate_ids = [wid for wid in out["whole_ids"] if wid in area_by_id]
    if not candidate_ids:
        return out, {}

    best_whole_id = max(candidate_ids, key=lambda mid: area_by_id.get(mid, -1.0))
    normalized = dict(out)
    normalized["whole_ids"] = [int(best_whole_id)]
    if normalized["whole_ids"] == out["whole_ids"]:
        return out, {}
    return normalized, {
        "kind": "normalize_tool_whole_ids",
        "before": out,
        "after": normalized,
        "reason": "keep only the largest support mask as whole_ids for tool role scoring",
    }


def _remove_whole_ids_from_tool_parts(parsed_ids: Dict[str, List[int]], tool_mode: bool) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {
        "whole_ids": sorted(set(int(x) for x in parsed_ids.get("whole_ids", []))),
        "handle_ids": sorted(set(int(x) for x in parsed_ids.get("handle_ids", []))),
        "work_end_ids": sorted(set(int(x) for x in parsed_ids.get("work_end_ids", []))),
        "contact_region_ids": sorted(set(int(x) for x in parsed_ids.get("contact_region_ids", []))),
    }
    if not tool_mode or not out["whole_ids"]:
        return out

    whole_ids = set(out["whole_ids"])
    out["handle_ids"] = [i for i in out["handle_ids"] if i not in whole_ids]
    out["work_end_ids"] = [i for i in out["work_end_ids"] if i not in whole_ids]
    return out


def _build_parts_from_vlm_ids(
    mask_candidates: List[np.ndarray],
    parsed_ids: Dict[str, List[int]],
    tool_mode: bool,
) -> Dict[str, np.ndarray]:
    if tool_mode:
        masks = [keep_largest_cc((m > 0).astype(np.uint8)) for m in mask_candidates if mask_area(m) > 0]
    else:
        masks = [_canonical_non_tool_mask(m, rel_min=0.04, pix_min=40) for m in mask_candidates if mask_area(m) > 0]
    if not masks:
        return {}
    if tool_mode:
        handle = keep_largest_cc(_union_from_ids(masks, parsed_ids.get("handle_ids", [])))
        work_end = keep_largest_cc(_union_from_ids(masks, parsed_ids.get("work_end_ids", [])))
        whole_hint = _union_from_ids(masks, parsed_ids.get("whole_ids", []))
        if mask_area(handle) == 0 or mask_area(work_end) == 0:
            return {}
        whole = compose_tool_whole(handle=handle, work_end=work_end, whole_hint=whole_hint)
        return {"whole": whole, "handle": handle, "work_end": work_end}

    whole = _canonical_non_tool_mask(_union_from_ids(masks, parsed_ids.get("whole_ids", [])), rel_min=0.04, pix_min=80)
    contact = _canonical_non_tool_mask(_union_from_ids(masks, parsed_ids.get("contact_region_ids", [])), rel_min=0.04, pix_min=40)
    if mask_area(whole) == 0 and mask_area(contact) == 0:
        return {}
    if mask_area(whole) == 0:
        whole = contact.copy()
    if mask_area(contact) == 0:
        contact = whole.copy()
    return {"whole": whole, "contact_region": contact}


def _is_degenerate_tool_parts(parts: Dict[str, np.ndarray]) -> bool:
    whole = parts.get("whole")
    handle = parts.get("handle")
    work_end = parts.get("work_end")
    if whole is None or handle is None or work_end is None:
        return True
    if mask_area(whole) == 0 or mask_area(handle) == 0 or mask_area(work_end) == 0:
        return True
    if mask_iou(handle, whole) > 0.95:
        return True
    if mask_iou(work_end, whole) > 0.95:
        return True
    if mask_iou(handle, work_end) > 0.90:
        return True
    return False


def _crop_masks_to_support(masks: List[np.ndarray], pad: int = 12) -> List[np.ndarray]:
    if not masks:
        return []
    support = union_masks([(m > 0).astype(np.uint8) for m in masks])
    bb = mask_bbox_xyxy(support)
    if bb is None:
        return [(m > 0).astype(np.uint8).copy() for m in masks]
    h, w = support.shape[:2]
    x1, y1, x2, y2 = bb
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    out: List[np.ndarray] = []
    for m in masks:
        out.append(((m > 0).astype(np.uint8))[y1 : y2 + 1, x1 : x2 + 1].copy())
    return out


def _sample_keypoints(mask01: np.ndarray) -> List[Tuple[int, int]]:
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return []

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    center = pts.mean(axis=0)
    out: List[Tuple[int, int]] = [(int(round(float(center[0]))), int(round(float(center[1]))))]

    xmin_i = int(np.argmin(xs))
    xmax_i = int(np.argmax(xs))
    ymin_i = int(np.argmin(ys))
    ymax_i = int(np.argmax(ys))
    out.extend(
        [
            (int(xs[xmin_i]), int(ys[xmin_i])),
            (int(xs[xmax_i]), int(ys[xmax_i])),
            (int(xs[ymin_i]), int(ys[ymin_i])),
            (int(xs[ymax_i]), int(ys[ymax_i])),
        ]
    )

    if len(pts) >= 10:
        centered = pts - center
        cov = (centered.T @ centered) / max(1.0, float(len(pts)))
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        proj = centered @ axis
        out.append((int(pts[int(np.argmin(proj)), 0]), int(pts[int(np.argmin(proj)), 1])))
        out.append((int(pts[int(np.argmax(proj)), 0]), int(pts[int(np.argmax(proj)), 1])))

    uniq: List[Tuple[int, int]] = []
    for p in out:
        if p not in uniq:
            uniq.append(p)
    return uniq[:6]


def _draw_keypoint_overlay(crop_bgr: np.ndarray, crop_masks: List[np.ndarray]) -> np.ndarray:
    canvas = crop_bgr.copy()
    if not crop_masks:
        return canvas

    palette = [
        (255, 90, 90),
        (90, 220, 255),
        (90, 255, 90),
        (255, 190, 90),
        (210, 120, 255),
        (255, 255, 90),
    ]

    for idx, mask in enumerate(crop_masks):
        m = (mask > 0).astype(np.uint8)
        if mask_area(m) <= 0:
            continue
        color = palette[idx % len(palette)]

        tint = np.zeros_like(canvas, dtype=np.uint8)
        tint[:, :] = color
        sel = m.astype(bool)
        canvas[sel] = cv2.addWeighted(canvas[sel], 0.65, tint[sel], 0.35, 0.0)

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, color, 2)

        ys, xs = np.where(m > 0)
        cx = int(round(float(xs.mean())))
        cy = int(round(float(ys.mean())))
        cv2.putText(
            canvas,
            f"id {idx}",
            (max(0, cx + 5), max(12, cy - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        for px, py in _sample_keypoints(m):
            cv2.circle(canvas, (int(px), int(py)), 3, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (int(px), int(py)), 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return canvas


def _morph_skeleton(mask01: np.ndarray) -> np.ndarray:
    """Morphological skeletonization for binary masks (0/1)."""
    img = (mask01 > 0).astype(np.uint8).copy()
    if mask_area(img) == 0:
        return img
    skel = np.zeros_like(img, dtype=np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(512):
        eroded = cv2.erode(img, ker)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, ker)
        temp = cv2.subtract(eroded, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return (skel > 0).astype(np.uint8)


def _major_axis_endpoints(mask01: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    ys, xs = np.where(mask01 > 0)
    if xs.size < 10:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mu = pts.mean(axis=0)
    centered = pts - mu
    cov = (centered.T @ centered) / max(1.0, float(len(pts)))
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    proj = centered @ axis
    p1 = pts[int(np.argmin(proj))]
    p2 = pts[int(np.argmax(proj))]
    return (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))


def _skeleton_endpoints(skel01: np.ndarray) -> List[Tuple[int, int]]:
    s = (skel01 > 0).astype(np.uint8)
    if mask_area(s) == 0:
        return []
    neigh = cv2.filter2D(s, cv2.CV_16S, np.ones((3, 3), dtype=np.uint8))
    # neigh includes the center pixel itself.
    ys, xs = np.where((s > 0) & (neigh <= 2))
    out: List[Tuple[int, int]] = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        out.append((int(x), int(y)))
    return out


def _farthest_pair(points: List[Tuple[int, int]]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if len(points) < 2:
        return None
    best = -1.0
    pair: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    for i in range(len(points)):
        x1, y1 = points[i]
        for j in range(i + 1, len(points)):
            x2, y2 = points[j]
            d2 = float((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if d2 > best:
                best = d2
                pair = (points[i], points[j])
    return pair


def _draw_skeleton_overlay(crop_bgr: np.ndarray, crop_masks: List[np.ndarray]) -> np.ndarray:
    canvas = crop_bgr.copy()
    if not crop_masks:
        return canvas

    palette = [
        (255, 90, 90),
        (90, 220, 255),
        (90, 255, 90),
        (255, 190, 90),
        (210, 120, 255),
        (255, 255, 90),
    ]
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for idx, mask in enumerate(crop_masks):
        m = (mask > 0).astype(np.uint8)
        if mask_area(m) <= 0:
            continue
        color = palette[idx % len(palette)]

        tint = np.zeros_like(canvas, dtype=np.uint8)
        tint[:, :] = color
        sel = m.astype(bool)
        canvas[sel] = cv2.addWeighted(canvas[sel], 0.78, tint[sel], 0.22, 0.0)

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, color, 2)

        skel = _morph_skeleton(m)
        skel_sel = skel.astype(bool)
        skel_tint = np.zeros_like(canvas, dtype=np.uint8)
        skel_tint[:, :] = color
        canvas[skel_sel] = cv2.addWeighted(canvas[skel_sel], 0.25, skel_tint[skel_sel], 0.75, 0.0)

        endpoints = _skeleton_endpoints(skel)
        axis_pair = _farthest_pair(endpoints)
        if axis_pair is None:
            axis_pair = _major_axis_endpoints(m)
        if axis_pair is not None:
            cv2.line(canvas, axis_pair[0], axis_pair[1], color, 2, cv2.LINE_AA)

        ys, xs = np.where(m > 0)
        cx = int(round(float(xs.mean())))
        cy = int(round(float(ys.mean())))
        cv2.putText(
            canvas,
            f"id {idx}",
            (max(0, cx + 5), max(12, cy - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        pts = _sample_keypoints(m)
        for p_i, (px, py) in enumerate(pts[:6]):
            cv2.circle(canvas, (int(px), int(py)), 3, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (int(px), int(py)), 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            letter = letters[p_i % len(letters)]
            cv2.putText(
                canvas,
                f"{letter}",
                (int(px) + 4, int(py) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
                cv2.LINE_AA,
            )

    return canvas


_VLM1_VERB_HINTS: List[Tuple[str, str, str]] = [
    ("pick up",  "handle = most stable elongated grip shaft",           "work_end = active tip or head"),
    ("pickup",   "handle = most stable elongated grip shaft",           "work_end = active tip or head"),
    ("put down", "handle = region held during placement",               "work_end = contact surface"),
    ("hammer",   "handle = long swing-shaft",                          "work_end = striking face/head"),
    ("unscrew",  "handle = shaft or body that is rotated",             "work_end = bit or tip"),
    ("tighten",  "handle = wrench or driver body",                     "work_end = jaw or bit"),
    ("loosen",   "handle = wrench or driver body",                     "work_end = jaw or bit"),
    ("screw",    "handle = shaft or body",                             "work_end = tip or bit"),
    ("scrape",   "handle = grip shaft",                                "work_end = blade or flat edge"),
    ("slice",    "handle = grip",                                      "work_end = blade"),
    ("pry",      "handle = longer grip end",                           "work_end = flat lever tip"),
    ("paint",    "handle = shaft",                                     "work_end = bristles or roller"),
    ("brush",    "handle = shaft",                                     "work_end = bristle head"),
    ("drill",    "handle = body or grip",                              "work_end = drill bit"),
    ("drive",    "handle = shaft or grip",                             "work_end = bit or tip"),
    ("strike",   "handle = swing-shaft",                               "work_end = striking face"),
    ("twist",    "handle = shaft or grip body",                        "work_end = acting end"),
    ("turn",     "handle = shaft or grip body",                        "work_end = acting end"),
    ("roll",     "handle = grip shaft",                                "work_end = rolling cylinder"),
    ("cut",      "handle = grip",                                      "work_end = blade or edge"),
    ("hit",      "handle = swing-shaft",                               "work_end = striking head"),
    ("lift",     "handle = most stable graspable region",              "work_end = load-bearing end"),
    ("grab",     "handle = region most suited for a firm grip",        "work_end = functional terminal"),
    ("grasp",    "handle = region most suited for a firm grip",        "work_end = functional terminal"),
    ("hold",     "handle = most comfortable stable grip region",       "work_end = functional terminal"),
    ("use",      "handle = region most naturally suited for grasping", "work_end = action terminal"),
]


def _vlm1_action_hint(instruction: str) -> str:
    ins_l = (instruction or "").lower()
    for verb, h_hint, w_hint in _VLM1_VERB_HINTS:
        if verb in ins_l:
            return f"Action verb '{verb}' detected — {h_hint}; {w_hint}."
    return "handle = most graspable grip region; work_end = functional action terminal."


def _build_prompt_text(
    object_name: str,
    instruction: str,
    mask_stats: List[Dict[str, Any]],
    tool_mode: bool,
    has_depth: bool,
    tool_family: Optional[str] = None,
    geom_prior: Optional[Dict[str, Any]] = None,
) -> str:
    target_schema = (
        '{"handle_ids":[...], "work_end_ids":[...], "whole_ids":[...]} (whole_ids optional)'
        if tool_mode
        else '{"whole_ids":[...], "contact_region_ids":[...]}'
    )

    tool_rules = ""
    if tool_mode:
        by_elong = sorted(mask_stats, key=lambda s: float(s.get("elongation", 0.0)), reverse=True)
        geom_hint = ""
        if len(by_elong) >= 2:
            hid = int(by_elong[0].get("id", -1))
            wid = int(by_elong[1].get("id", -1))
            geom_hint = (
                f"Geometry hint: id {hid} is more elongated than id {wid}. "
            )

        family_hint = ""
        if tool_family == "head_dominant":
            family_hint = "Geometry archetype: one elongated grip-side part and one broader terminal head-like part.\n"
        elif tool_family == "axial":
            family_hint = "Geometry archetype: slim longitudinal tool with elongated terminal candidates.\n"
        elif tool_family == "multi_terminal":
            family_hint = "Geometry archetype: multiple terminal compact/contact candidates.\n"
        prior_text = ""
        if geom_prior and geom_prior.get("handle_id") is not None and geom_prior.get("work_end_id") is not None:
            h_id = int(geom_prior["handle_id"])
            w_id = int(geom_prior["work_end_id"])
            h_stat = next((s for s in mask_stats if int(s.get("id", -1)) == h_id), {})
            w_stat = next((s for s in mask_stats if int(s.get("id", -1)) == w_id), {})
            prior_text = (
                "Geometry prior (shape analysis best guess — confirm or override with visual evidence):\n"
                f"  handle candidate: id {h_id}"
                f"  elongation={h_stat.get('elongation', 0):.2f}"
                f"  terminality={h_stat.get('terminality', 0):.2f}"
                f"  skeleton_px={h_stat.get('skeleton_length_px', 0):.0f}"
                f"  width_cv={h_stat.get('width_cv', 0):.2f}"
                f"  mean_half_width={h_stat.get('mean_half_width_px', 0):.1f}\n"
                f"  work_end candidate: id {w_id}"
                f"  elongation={w_stat.get('elongation', 0):.2f}"
                f"  terminality={w_stat.get('terminality', 0):.2f}"
                f"  area_ratio={w_stat.get('area_ratio_vs_support', 0):.3f}"
                f"  endpoint_count={w_stat.get('endpoint_count', 0)}\n"
                "If the geometry prior matches the visual, echo it. "
                "If visual evidence clearly conflicts (e.g., shape is misleading for this tool type), "
                "output your corrected assignment.\n"
            )

        action_hint_text = f"Action role guidance: {_vlm1_action_hint(instruction)}\n"
        tool_rules = (
            action_hint_text
            + "Part-role definitions (strict):\n"
            "- handle: where human grips the tool.\n"
            "- work_end: functional end that contacts/acts on target.\n"
            "- Never use the whole-mask id as handle/work_end if a dedicated part id exists.\n"
            "- For unknown tools, handle should be the most graspable/stable grip region.\n"
            "- work_end should be the active distal terminal region, not a middle connector fragment.\n"
            "- Prefer handle candidates that are elongated, stable, and easy to grasp.\n"
            "- Prefer work_end candidates that are terminal, action-facing, and geometrically distinct from the handle.\n"
            "- Tiny bridge/neck pieces are usually not the work_end unless they are clearly the only terminal action region.\n"
            + family_hint
            + prior_text
            + "- select handle/work_end only from available SAM IDs in the grid.\n"
            + geom_hint
            + "- Enforce mapping even if color cues are ambiguous.\n"
        )
    else:
        obj_l = (object_name or "").lower().strip()
        if "screw" in obj_l:
            tool_rules = (
                "Target-part definitions (strict):\n"
                "- whole_ids: full object support mask.\n"
                "- contact_region_ids: all actionable terminal regions.\n"
                "- For a screw, contact_region may contain multiple IDs.\n"
                "- Prefer the screw head and the lower/tip region; do not keep only the middle shaft.\n"
            )

    image_notes = (
        "Image 1: RGB crop.\n"
        "Image 2: SAM grid with candidate IDs (id 0, id 1, id 2, ...).\n"
        + ("Image 3: depth colormap aligned to RGB crop.\n" if has_depth else "")
        + "Image 4: skeleton-overlay view (each mask has center line + labeled points A,B,C...).\n"
    )
    valid_id_text = (
        "Valid SAM IDs in this example are exactly: "
        + ", ".join(str(i) for i in range(len(mask_stats)))
        + ".\n"
    )

    # Task-conditioned affordance attribute guidance for structured grounding
    affordance_guidance = ""
    if tool_mode:
        affordance_guidance = (
            "\n=== Structured Affordance Attributes (task-conditioned) ===\n"
            "Beyond region IDs, predict per-region affordance attributes conditioned on the\n"
            "action verb in the task instruction.  These attributes ground each region into\n"
            "a formal manipulation primitive for wrench-bounded constraint synthesis.\n\n"
            "grasp_type (HANDLE) — how the hand grips the handle:\n"
            '  "power"     — full-hand cylindrical grip (hammer, screwdriver, heavy shaft).\n'
            '  "precision" — thumb+2-finger fingertip (thin shaft, pen-like).\n'
            '  "pinch"     — thumb+index tip pinch (very narrow rod, small handle).\n'
            '  "hook"      — hook or overhand grip (trowel, L-shape, hook-handle).\n\n'
            "contact_geometry (WORK_END) — surface type of the contact region:\n"
            '  "point"   — concentrated point contact (screwdriver tip, punch, needle).\n'
            '  "line"    — linear edge contact (scraper blade, knife edge, chisel).\n'
            '  "surface" — broad area contact (spatula face, roller, brush).\n\n'
            "force_mode (WORK_END) — dominant wrench component during task execution:\n"
            '  "push"   — compressive force along tool axis (hammering, pressing).\n'
            '  "pull"   — tensile/dragging force (scraping, raking, pulling).\n'
            '  "torque" — rotational moment (screwdriver, wrench, drill).\n'
            '  "normal" — force normal to surface (painting, coating, brushing).\n\n'
        )
        target_schema = (
            '{"handle_ids":[...], "work_end_ids":[...], "whole_ids":[...], '
            '"grasp_type":"power"|"precision"|"pinch"|"hook", '
            '"contact_geometry":"point"|"line"|"surface", '
            '"force_mode":"push"|"pull"|"torque"|"normal", '
            '"attribute_confidence":{"grasp_type":0.0-1.0,"contact_geometry":0.0-1.0,"force_mode":0.0-1.0}}'
            " (whole_ids, affordance fields optional)"
        )

    return (
        "You are canonicalizing object parts from SAM candidates into a structured "
        "region-level affordance representation.\n"
        f"Task instruction: {instruction}\n"
        f"Object name: {object_name}\n"
        + image_notes
        + valid_id_text
        + tool_rules
        + affordance_guidance
        + "Use visual evidence + skeleton overlay + geometry/depth stats.\n"
        + "Geometry/depth stats JSON:\n"
        + json.dumps(mask_stats, ensure_ascii=True)
        + "\nReturn ONLY JSON with schema: "
        + target_schema
        + "\nIDs must refer to SAM grid IDs."
        + "\nNever output an id outside the valid SAM ID list above."
        + "\nDo not swap handle/work_end."
    )


def _build_revision_prompt_text(
    object_name: str,
    instruction: str,
    first_ids: Dict[str, List[int]],
    tool_mode: bool,
    n_masks: Optional[int] = None,
) -> str:
    target_schema = (
        '{"handle_ids":[...], "work_end_ids":[...], "whole_ids":[...]} (whole_ids optional)'
        if tool_mode
        else '{"whole_ids":[...], "contact_region_ids":[...]}'
    )
    valid_id_text = ""
    if n_masks is not None and n_masks > 0:
        valid_id_text = (
            "Valid SAM IDs are exactly: "
            + ", ".join(str(i) for i in range(n_masks))
            + ". Do not use any other id.\n"
        )
    return (
        "Re-check and correct previous part-ID assignment.\n"
        f"Task instruction: {instruction}\n"
        f"Object name: {object_name}\n"
        f"Previous assignment JSON: {json.dumps(first_ids, ensure_ascii=True)}\n"
        + valid_id_text
        + "Use the same images/stats and check for handle/work_end swap.\n"
        + "Definitions: handle = grip part, work_end = functional contact part.\n"
        + "Return ONLY JSON with schema: "
        + target_schema
    )


def _build_adjudication_prompt_text(
    object_name: str,
    instruction: str,
    first_ids: Dict[str, List[int]],
    revised_ids: Dict[str, List[int]],
    tool_mode: bool,
) -> str:
    schema = '{"pick":"first"} or {"pick":"revised"}'
    rules = ""
    if tool_mode:
        rules = (
            "Role rules:\n"
            "- handle = grip part, usually long slim shaft (larger elongation/bbox_diag).\n"
            "- work_end = functional contact part.\n"
        )
    return (
        "Select which assignment is more correct.\n"
        f"Task instruction: {instruction}\n"
        f"Object name: {object_name}\n"
        f"Assignment FIRST: {json.dumps(first_ids, ensure_ascii=True)}\n"
        f"Assignment REVISED: {json.dumps(revised_ids, ensure_ascii=True)}\n"
        + rules
        + "Return ONLY JSON: "
        + schema
    )


def _parse_pick_choice(raw_text: str) -> Optional[str]:
    parsed = extract_json_block(raw_text)
    if isinstance(parsed, dict):
        pick = str(parsed.get("pick", "")).strip().lower()
        if pick in {"first", "revised"}:
            return pick
    t = (raw_text or "").lower()
    if "first" in t and "revised" not in t:
        return "first"
    if "revised" in t and "first" not in t:
        return "revised"
    return None


def _should_skip_recheck(object_name: str, tool_mode: bool) -> bool:
    return False


def build_openai_client() -> Optional[Any]:
    client, _ = build_openai_client_with_status()
    return client


def _to_chat_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    out: List[Dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        t = str(item.get("type", "")).strip().lower()
        if t == "input_text":
            out.append({"type": "text", "text": str(item.get("text", ""))})
            continue
        if t == "input_image":
            iu = item.get("image_url")
            url = None
            if isinstance(iu, str):
                url = iu
            elif isinstance(iu, dict):
                url = iu.get("url")
            if isinstance(url, str) and url:
                out.append({"type": "image_url", "image_url": {"url": url}})
            continue
        # already chat-compatible types
        if t in {"text", "image_url"}:
            out.append(item)
    return out if out else str(content)


def _query_vlm_content_once(client: Any, model: str, content: Any) -> Tuple[str, str]:
    # Route 1: Responses API (preferred)
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
        )
        raw_text = str(getattr(resp, "output_text", "") or "").strip()
        if not raw_text:
            raw_text = str(resp)
        return raw_text, "responses"
    except Exception as exc_resp:
        # Route 2: Chat Completions API (provider compatibility fallback)
        try:
            cc = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _to_chat_content(content)}],
            )
            txt = ""
            if getattr(cc, "choices", None):
                txt = str(getattr(cc.choices[0].message, "content", "") or "").strip()
            if not txt:
                txt = str(cc)
            return txt, "chat_completions"
        except Exception as exc_chat:
            raise RuntimeError(f"responses_error={exc_resp}; chat_completions_error={exc_chat}") from exc_chat


def query_vlm_content(client: Any, model: str, content: Any) -> Tuple[str, str]:
    try:
        return _query_vlm_content_once(client=client, model=model, content=content)
    except Exception as exc:
        failure = classify_openai_failure(exc)
        should_retry_with_env_proxy = (
            MODEL_CFG["openai_ignore_env_proxy"]
            and failure.get("category") == "network"
        )
        if not should_retry_with_env_proxy:
            raise

        retry_client, retry_status = build_openai_client_with_status(trust_env_override=True)
        if retry_client is None:
            retry_msg = retry_status.get("message") if isinstance(retry_status, dict) else "failed to build retry client"
            raise RuntimeError(
                f"{exc}; env_proxy_retry_client_error={retry_msg}"
            ) from exc

        try:
            raw_text, route = _query_vlm_content_once(client=retry_client, model=model, content=content)
            return raw_text, f"{route}_env_proxy_retry"
        except Exception as retry_exc:
            raise RuntimeError(
                f"{exc}; env_proxy_retry_error={retry_exc}"
            ) from retry_exc


def _try_read_cache(
    cache_path: Optional[str],
    object_name: str,
    masks: List[np.ndarray],
    tool_mode: bool,
    mask_stats: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, List[int]]]]:
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        cache_obj = load_json(cache_path)
    except Exception:
        return None

    if str(cache_obj.get("object_name", object_name)).strip().lower() != object_name.strip().lower():
        return None

    n_masks = int(cache_obj.get("n_masks", -1))
    if n_masks != len(masks):
        return None
    if not _cache_matches_masks(cache_obj, masks):
        return None

    parsed_ids = _normalize_selection_dict(cache_obj.get("parsed_ids"), n_masks=len(masks), tool_mode=tool_mode)
    if parsed_ids is None:
        return None
    parsed_ids, _ = _synthesize_tool_whole_ids(parsed_ids, tool_mode=tool_mode)
    parsed_ids, _ = _normalize_tool_whole_ids(parsed_ids, tool_mode=tool_mode, mask_stats=mask_stats or [])
    parsed_ids = _remove_whole_ids_from_tool_parts(parsed_ids, tool_mode=tool_mode)
    parsed_ids, _ = _repair_tool_role_selection(
        parsed_ids,
        object_name=object_name,
        instruction="",
        tool_mode=tool_mode,
        mask_stats=mask_stats or [],
    )
    parts = _build_parts_from_vlm_ids(mask_candidates=masks, parsed_ids=parsed_ids, tool_mode=tool_mode)
    if not parts:
        return None
    if tool_mode and _is_degenerate_tool_parts(parts):
        return None

    cached_parsed_ids = cache_obj.get("parsed_ids")
    if isinstance(cached_parsed_ids, dict) and cached_parsed_ids != parsed_ids and cache_path:
        try:
            updated_cache = dict(cache_obj)
            updated_cache["parsed_ids"] = parsed_ids
            save_json(cache_path, updated_cache)
        except Exception:
            pass

    return parts, parsed_ids


def canonicalize_by_vlm_prompt(
    object_name: str,
    instruction: str,
    masks: List[np.ndarray],
    full_bgr: np.ndarray,
    depth: Optional[np.ndarray],
    grid_path: Optional[str],
    tool_mode: bool,
    client: Optional[Any],
    cache_path: Optional[str] = None,
    skeleton_overlay_path: Optional[str] = None,
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "enabled": True,
        "openai_available": bool(OPENAI_AVAILABLE),
        "used": False,
        "cache_path": cache_path,
        "cache_hit": False,
    }

    if not masks:
        debug["reason"] = "no_masks"
        return None, debug

    crop_bgr, depth_crop = crop_from_masks(full_bgr=full_bgr, masks=masks, depth=depth)
    if not skeleton_overlay_path or not os.path.exists(skeleton_overlay_path):
        debug.update({"reason": "skeleton_overlay_missing", "skeleton_overlay_path": skeleton_overlay_path})
        return None, debug
    skeleton_overlay = cv2.imread(skeleton_overlay_path, cv2.IMREAD_COLOR)
    if skeleton_overlay is None:
        debug.update({"reason": "skeleton_overlay_not_readable", "skeleton_overlay_path": skeleton_overlay_path})
        return None, debug
    skeleton_source = "skeleton.py"
    skeleton_saved_path: Optional[str] = skeleton_overlay_path
    debug["skeleton_source"] = skeleton_source
    debug["skeleton_overlay_path"] = skeleton_saved_path

    mask_stats = build_mask_geometry_stats(mask_candidates=masks, depth=depth)
    effective_tool_mode, tool_mode_debug = _infer_effective_tool_mode(
        object_name=object_name,
        instruction=instruction,
        mask_stats=mask_stats,
        declared_tool_mode=tool_mode,
    )
    tool_family, tool_family_debug = _infer_tool_role_family(
        object_name=object_name,
        instruction=instruction,
        mask_stats=mask_stats,
    )
    cached = _try_read_cache(
        cache_path=cache_path,
        object_name=object_name,
        masks=masks,
        tool_mode=effective_tool_mode,
        mask_stats=mask_stats,
    )
    if cached is not None:
        parts, parsed_ids = cached
        debug.update(
            {
                "used": True,
                "reason": "ok_from_cache",
                "cache_hit": True,
                "parsed_ids": parsed_ids,
                "effective_tool_mode": bool(effective_tool_mode),
                "tool_mode_debug": tool_mode_debug,
                "tool_family": tool_family,
                "tool_family_debug": tool_family_debug,
                "skeleton_source": skeleton_source,
                "skeleton_overlay_path": skeleton_saved_path,
            }
        )
        return parts, debug

    if client is None:
        debug["reason"] = "no_openai_client"
        return None, debug
    if not grid_path:
        debug["reason"] = "missing_grid_path"
        return None, debug

    grid_bgr = cv2.imread(grid_path, cv2.IMREAD_COLOR)
    if grid_bgr is None:
        debug["reason"] = "grid_not_readable"
        return None, debug

    depth_vis = colorize_depth(depth_crop)

    # Compute geometry-best (handle, work_end) pair as a prior for the VLM.
    # This runs _repair_tool_role_selection with no VLM prior so it is purely
    # geometry-driven, then we embed it in the prompt so the VLM can confirm
    # or consciously override it instead of guessing from scratch.
    geom_prior: Optional[Dict[str, Any]] = None
    if effective_tool_mode and len(mask_stats) >= 2:
        _empty_ids: Dict[str, List[int]] = {
            "whole_ids": [],
            "handle_ids": [],
            "work_end_ids": [],
            "contact_region_ids": [],
        }
        _gp_ids, _gp_debug = _repair_tool_role_selection(
            _empty_ids,
            object_name=object_name,
            instruction=instruction,
            tool_mode=True,
            mask_stats=mask_stats,
        )
        if _gp_ids.get("handle_ids") and _gp_ids.get("work_end_ids"):
            geom_prior = {
                "handle_id": int(_gp_ids["handle_ids"][0]),
                "work_end_id": int(_gp_ids["work_end_ids"][0]),
                "family": tool_family,
            }

    prompt = _build_prompt_text(
        object_name=object_name,
        instruction=instruction,
        mask_stats=mask_stats,
        tool_mode=effective_tool_mode,
        has_depth=(depth_vis is not None),
        tool_family=tool_family,
        geom_prior=geom_prior,
    )

    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": encode_image_to_data_url(crop_bgr)},
        {"type": "input_image", "image_url": encode_image_to_data_url(grid_bgr)},
    ]
    if depth_vis is not None:
        content.append({"type": "input_image", "image_url": encode_image_to_data_url(depth_vis)})
    content.append({"type": "input_image", "image_url": encode_image_to_data_url(skeleton_overlay)})

    raw_text = ""
    revision_raw_text = ""
    revision_error = ""
    adjudication_raw_text = ""
    adjudication_error = ""
    adjudication_pick = ""

    skip_recheck = _should_skip_recheck(object_name=object_name, tool_mode=effective_tool_mode)

    try:
        raw_text, api_route = query_vlm_content(client=client, model=MODEL_CFG["vlm_model"], content=content)
        debug["api_route"] = api_route
    except Exception as exc:
        debug.update(
            {
                "reason": "openai_call_failed",
                "error": str(exc),
                "api_failure": classify_openai_failure(exc),
                "mask_stats": mask_stats,
            }
        )
        return None, debug

    parsed_ids = _parse_vlm_mask_selection(raw_text=raw_text, n_masks=len(masks), tool_mode=effective_tool_mode)
    if parsed_ids is None:
        debug.update({"reason": "parse_failed", "raw_text": raw_text, "mask_stats": mask_stats})
        return None, debug

    parsed_ids, first_pass_whole_synthesized = _synthesize_tool_whole_ids(parsed_ids, tool_mode=effective_tool_mode)
    parsed_ids, whole_id_repair = _normalize_tool_whole_ids(
        parsed_ids,
        tool_mode=effective_tool_mode,
        mask_stats=mask_stats,
    )
    parsed_ids = _remove_whole_ids_from_tool_parts(parsed_ids, tool_mode=effective_tool_mode)
    parsed_ids, role_pair_repair = _repair_tool_role_selection(
        parsed_ids,
        object_name=object_name,
        instruction=instruction,
        tool_mode=effective_tool_mode,
        mask_stats=mask_stats,
    )
    revised_ids = parsed_ids

    if not skip_recheck:
        try:
            rev_prompt = _build_revision_prompt_text(
                object_name=object_name,
                instruction=instruction,
                first_ids=parsed_ids,
                tool_mode=effective_tool_mode,
                n_masks=len(masks),
            )
            rev_content: List[Dict[str, Any]] = [{"type": "input_text", "text": rev_prompt}] + content[1:]
            revision_raw_text, revision_route = query_vlm_content(
                client=client,
                model=MODEL_CFG["vlm_model"],
                content=rev_content,
            )
            debug["revision_api_route"] = revision_route
            parsed_revised = _parse_vlm_mask_selection(
                raw_text=revision_raw_text,
                n_masks=len(masks),
                tool_mode=effective_tool_mode,
            )
            if parsed_revised is not None:
                revised_ids, _ = _synthesize_tool_whole_ids(parsed_revised, tool_mode=effective_tool_mode)
                revised_ids, _ = _normalize_tool_whole_ids(
                    revised_ids,
                    tool_mode=effective_tool_mode,
                    mask_stats=mask_stats,
                )
                revised_ids = _remove_whole_ids_from_tool_parts(revised_ids, tool_mode=effective_tool_mode)
                revised_ids, _ = _repair_tool_role_selection(
                    revised_ids,
                    object_name=object_name,
                    instruction=instruction,
                    tool_mode=effective_tool_mode,
                    mask_stats=mask_stats,
                )
        except Exception as exc:
            revision_error = str(exc)

        try:
            adj_prompt = _build_adjudication_prompt_text(
                object_name=object_name,
                instruction=instruction,
                first_ids=parsed_ids,
                revised_ids=revised_ids,
                tool_mode=effective_tool_mode,
            )
            adj_content: List[Dict[str, Any]] = [{"type": "input_text", "text": adj_prompt}] + content[1:]
            adjudication_raw_text, adjudication_route = query_vlm_content(
                client=client,
                model=MODEL_CFG["vlm_model"],
                content=adj_content,
            )
            debug["adjudication_api_route"] = adjudication_route
            pick = _parse_pick_choice(adjudication_raw_text)
            if pick in {"first", "revised"}:
                adjudication_pick = pick
                revised_ids = parsed_ids if pick == "first" else revised_ids
        except Exception as exc:
            adjudication_error = str(exc)

    revised_ids, revised_whole_synthesized = _synthesize_tool_whole_ids(revised_ids, tool_mode=effective_tool_mode)
    revised_ids, whole_id_repair_final = _normalize_tool_whole_ids(
        revised_ids,
        tool_mode=effective_tool_mode,
        mask_stats=mask_stats,
    )
    revised_ids = _remove_whole_ids_from_tool_parts(revised_ids, tool_mode=effective_tool_mode)
    revised_ids, role_pair_repair_final = _repair_tool_role_selection(
        revised_ids,
        object_name=object_name,
        instruction=instruction,
        tool_mode=effective_tool_mode,
        mask_stats=mask_stats,
    )
    parts = _build_parts_from_vlm_ids(mask_candidates=masks, parsed_ids=revised_ids, tool_mode=effective_tool_mode)
    if not parts:
        debug.update(
            {
                "reason": "invalid_selection",
                "raw_text": raw_text,
                "parsed_ids": revised_ids,
                "mask_stats": mask_stats,
                "revision_raw_text": revision_raw_text,
                "revision_error": revision_error,
                "adjudication_raw_text": adjudication_raw_text,
                "adjudication_error": adjudication_error,
                "adjudication_pick": adjudication_pick,
                "effective_tool_mode": bool(effective_tool_mode),
                "tool_mode_debug": tool_mode_debug,
                "tool_family": tool_family,
                "tool_family_debug": tool_family_debug,
                "whole_id_repair": whole_id_repair_final or whole_id_repair,
                "role_pair_repair": role_pair_repair_final,
            }
        )
        return None, debug

    if effective_tool_mode and _is_degenerate_tool_parts(parts):
        debug.update(
            {
                "reason": "degenerate_tool_parts",
                "raw_text": raw_text,
                "parsed_ids": revised_ids,
                "mask_stats": mask_stats,
                "revision_raw_text": revision_raw_text,
                "revision_error": revision_error,
                "adjudication_raw_text": adjudication_raw_text,
                "adjudication_error": adjudication_error,
                "adjudication_pick": adjudication_pick,
                "effective_tool_mode": bool(effective_tool_mode),
                "tool_mode_debug": tool_mode_debug,
                "tool_family": tool_family,
                "tool_family_debug": tool_family_debug,
                "whole_id_repair": whole_id_repair_final or whole_id_repair,
                "role_pair_repair": role_pair_repair_final,
            }
        )
        return None, debug

    if cache_path:
        try:
            save_json(
                cache_path,
                {
                    "object_name": object_name,
                    "tool_mode": bool(effective_tool_mode),
                    "n_masks": int(len(masks)),
                    "mask_signatures": [_mask_signature(m) for m in masks],
                    "parsed_ids": revised_ids,
                    "first_pass_ids": parsed_ids,
                    "whole_ids_source": (
                        "synthesized_from_tool_parts"
                        if bool(revised_whole_synthesized or first_pass_whole_synthesized)
                        else "vlm_response"
                    ),
                },
            )
        except Exception:
            pass

    debug.update(
        {
            "used": True,
            "reason": "ok",
            "parsed_ids": revised_ids,
            "first_pass_ids": parsed_ids,
            "effective_tool_mode": bool(effective_tool_mode),
            "tool_mode_debug": tool_mode_debug,
            "tool_family": tool_family,
            "tool_family_debug": tool_family_debug,
            "whole_id_repair": whole_id_repair_final or whole_id_repair,
            "skip_recheck": bool(skip_recheck),
            "raw_text": raw_text,
            "revision_raw_text": revision_raw_text,
            "revision_error": revision_error,
            "adjudication_raw_text": adjudication_raw_text,
            "adjudication_error": adjudication_error,
            "adjudication_pick": adjudication_pick,
            "role_pair_repair": role_pair_repair_final or role_pair_repair,
            "mask_stats": mask_stats,
            "whole_ids_source": (
                "synthesized_from_tool_parts"
                if bool(revised_whole_synthesized or first_pass_whole_synthesized)
                else "vlm_response"
            ),
            "whole_ids_synthesized": bool(revised_whole_synthesized or first_pass_whole_synthesized),
            "skeleton_source": skeleton_source,
            "skeleton_overlay_path": skeleton_saved_path,
        }
    )
    return parts, debug


def parse_stage2_points_json(raw_text: str) -> Optional[Dict[str, Any]]:
    parsed = extract_json_block(raw_text)
    if not isinstance(parsed, dict):
        return None

    work = parsed.get("work_end") or {}
    grip = parsed.get("grip") or {}
    pattern = parsed.get("pattern") or ""

    line = work.get("letters") or []
    if isinstance(line, str):
        line = [x.strip().upper() for x in line.split(",") if x.strip()]
    else:
        line = [str(x).strip().upper() for x in line]

    side = work.get("side") or ""
    side = str(side).strip().upper().replace("-", "").replace("_", "")
    if not side and len(line) == 2:
        pair = tuple(sorted([str(line[0]).upper(), str(line[1]).upper()]))
        side_map = {
            ("A", "B"): "AB",
            ("B", "C"): "BC",
            ("C", "D"): "CD",
            ("A", "D"): "DA",
        }
        side = side_map.get(pair, "")
    if side in {"AB", "BC", "CD", "DA"} and len(line) != 2:
        line = [side[0], side[1]]

    letter = grip.get("letter") or ""
    letter = str(letter).strip().upper()
    pattern = str(pattern).strip().upper().replace("-", "_")
    if pattern in {"REVERSE_L", "REVERSEL", "REVERSE_LIKE", "HAMMER", "MALLET", "HEX_KEY"}:
        pattern = "REVERSE_L"
    if pattern == "T_PATTERN":
        pattern = "T"
    if pattern in {"I_PATTERN", "LINE", "STRAIGHT"}:
        pattern = "I"

    out = {
        "pattern": pattern,
        "work_end": {
            "id": work.get("id"),
            "side": side,
            "letters": [x for x in line if x][:2],
        },
        "grip": {
            "id": grip.get("id"),
            "letter": letter,
        },
        "source": "vlm",
    }
    return out


def validate_stage2_pick(
    parsed: Dict[str, Any],
    work_points: Dict[str, Tuple[int, int]],
    handle_points: Dict[str, Tuple[int, int]],
) -> bool:
    pattern = str(parsed.get("pattern", "")).strip().upper()
    if pattern not in {"T", "I", "REVERSE_L"}:
        return False
    line = parsed.get("work_end", {}).get("letters", [])
    side = str(parsed.get("work_end", {}).get("side", "")).strip().upper()
    grip = parsed.get("grip", {}).get("letter", "")
    if len(line) != 2:
        return False
    if side not in {"AB", "BC", "CD", "DA"}:
        return False
    if line[0] not in work_points or line[1] not in work_points:
        return False
    if not grip or grip not in handle_points:
        return False
    middle_letters = [k for k in ["O", "E", "F", "G", "H"] if k in handle_points]
    if middle_letters and grip not in middle_letters:
        return False
    return True


def parse_target_edge_json(raw_text: str) -> Optional[Dict[str, Any]]:
    parsed = extract_json_block(raw_text)
    if not isinstance(parsed, dict):
        return None

    edge = parsed.get("target_edge") or parsed.get("contact_line") or parsed.get("line") or {}
    letters = edge.get("letters") or edge.get("line") or edge.get("points") or []
    if isinstance(letters, str):
        letters = [x.strip().upper() for x in letters.split(",") if x.strip()]
    else:
        letters = [str(x).strip().upper() for x in letters]

    out = {
        "target_edge": {
            "id": edge.get("id"),
            "letters": [x for x in letters if x][:2],
        },
        "source": "vlm_target",
    }
    return out


def validate_target_edge_pick(
    parsed: Dict[str, Any],
    target_points: Dict[str, Tuple[int, int]],
) -> bool:
    line = parsed.get("target_edge", {}).get("letters", [])
    if len(line) != 2:
        return False
    a = str(line[0]).strip().upper()
    b = str(line[1]).strip().upper()
    if not a or not b or a == b:
        return False
    if a not in target_points or b not in target_points:
        return False
    return True


def _valid_stage2_semantic_only(
    parsed: Optional[Dict[str, Any]],
    grip_letters: List[str],
    work_end_letters: List[str],
    side_tokens: List[str],
) -> bool:
    if not isinstance(parsed, dict):
        return False
    pattern = str(parsed.get("pattern", "")).strip().upper()
    side = str(parsed.get("work_end", {}).get("side", "")).strip().upper()
    line = [str(x).strip().upper() for x in parsed.get("work_end", {}).get("letters", [])]
    grip = str(parsed.get("grip", {}).get("letter", "")).strip().upper()
    if pattern not in {"T", "I", "REVERSE_L"}:
        return False
    if side not in side_tokens:
        return False
    if len(line) != 2:
        return False
    if line[0] == line[1]:
        return False
    if line[0] not in work_end_letters or line[1] not in work_end_letters:
        return False
    if {line[0], line[1]} != {side[0], side[1]}:
        return False
    if not grip or grip not in grip_letters:
        return False
    return True


def _valid_target_letters_only(
    parsed: Optional[Dict[str, Any]],
    target_letters: List[str],
) -> bool:
    if not isinstance(parsed, dict):
        return False
    line = [str(x).strip().upper() for x in parsed.get("target_edge", {}).get("letters", [])]
    if len(line) != 2:
        return False
    if line[0] == line[1]:
        return False
    if line[0] not in target_letters or line[1] not in target_letters:
        return False
    return True


def build_target_edge_prompt_text(
    object_name: str,
    instruction: str,
    target_id: int,
    target_letters: List[str],
    tool_touch_hint_global: Optional[Tuple[int, int]] = None,
) -> str:
    hint_text = (
        f"Tool-touch hint in full image coordinates is approximately ({int(tool_touch_hint_global[0])}, {int(tool_touch_hint_global[1])})."
        if tool_touch_hint_global is not None
        else "Tool-touch hint is unavailable; infer from instruction + object geometry only."
    )
    return (
        "Task: choose one actionable target contact/alignment line as two skeleton letters.\n"
        f"Object: {object_name}. Instruction: {instruction}.\n"
        f"Target id: {target_id}. Available letters: {', '.join(target_letters)}.\n"
        f"{hint_text}\n"
        "\n"
        "Decision policy (universal):\n"
        "1) Parse intent words from instruction.\n"
        "2) Decide placement mode:\n"
        "   - BORDER mode: for phrases like along border/edge/rim/perimeter/side/outline.\n"
        "   - MIDDLE mode: for phrases like middle/center/central/through center.\n"
        "   - AUTO mode: if instruction is ambiguous.\n"
        "3) Choose two letters that best represent the intended line.\n"
        "4) Prefer clean, long, usable lines over tiny noisy segments.\n"
        "\n"
        "Geometric rules:\n"
        "- BORDER mode: prefer one full outer side (A-B, B-C, C-D, D-A when available).\n"
        "- MIDDLE mode: prefer central helper axis (E-F or G-H) when available.\n"
        "- AUTO mode with tool hint: choose side/line nearest likely tool approach side.\n"
        "- AUTO mode without tool hint: choose dominant stable line for manipulation.\n"
        "- Never return duplicate letters.\n"
        "- Return exactly two letters from available set.\n"
        "\n"
        "Examples:\n"
        "A) \"scrape along the border\" -> BORDER mode -> choose border side near tool approach.\n"
        "B) \"roll tape on the middle\" -> MIDDLE mode -> choose center axis line.\n"
        "C) \"cut paper along edge\" -> BORDER mode -> choose a long outer side.\n"
        "D) \"press button center\" -> MIDDLE mode -> choose center-aligned line.\n"
        "E) \"spread putty\" (ambiguous) + tool hint at left -> AUTO mode -> choose left-near stable line.\n"
        "F) \"align on boundary\" -> BORDER mode -> boundary side line.\n"
        "\n"
        "Return JSON only:\n"
        '{"target_edge":{"id":<int>,"letters":["<L>","<L>"]},"intent":{"mode":"BORDER|MIDDLE|AUTO","reason":"short text"}}'
    )


def build_stage2_prompt_text(
    object_name: str,
    instruction: str,
    handle_id: int,
    work_end_id: int,
    handle_letters: List[str],
    work_end_letters: List[str],
    handle_middle_letters: List[str],
    available_side_tokens: List[str],
    geometry_hint_text: Optional[str] = None,
    target_relation_hint: Optional[str] = None,
) -> str:
    extra = ""
    if target_relation_hint:
        extra += f"\nScene target hint:\n{target_relation_hint}\n"
    if geometry_hint_text:
        extra += f"\nGeometry hint table (from skeleton coordinates):\n{geometry_hint_text}\n"

    return (
        "Research protocol: the VLM performs semantic affordance selection only; exact edge endpoints are realized deterministically from the selected border side token.\n"
        f"Object: {object_name}. Instruction: {instruction}.\n"
        "You must output exactly three semantic decisions:\n"
        "1) pattern: choose one of {T, I, REVERSE_L}.\n"
        "2) grip: choose one middle-handle letter.\n"
        "3) work_end.side: choose one border side token.\n"
        "\n"
        "Pattern definitions:\n"
        "- T: scraper / roller / cutter / spatula family. Handle axis approaches active work-end side approximately orthogonally.\n"
        "- I: screwdriver / knife / blade-like family. Action axis stays straight from grip toward the active end.\n"
        "- REVERSE_L: hammer / mallet family. Handle axis enters head and active side is a terminal striking side.\n"
        "- REVERSE_L also covers bent tools such as hex key / allen key where the active branch turns away from the main handle axis.\n"
        "\n"
        "Grip rule:\n"
        "- grip MUST be one middle-handle letter only, not an end corner.\n"
        "\n"
        "Work-end rule:\n"
        "- choose one true border side token only, not arbitrary letters and not helper axis.\n"
        "- valid side tokens are exactly the available border sides below.\n"
        "\n"
        "Decision checklist:\n"
        "- infer pattern from tool family + scene geometry + instruction semantics.\n"
        "- pick grip from middle-handle zone.\n"
        "- pick border side token consistent with active contact side.\n"
        "- do not invent letters, points, or side tokens.\n"
        f"Handle id is {handle_id}; letters available: {', '.join(handle_letters)}.\n"
        f"Middle-handle letters allowed for grip: {', '.join(handle_middle_letters)}.\n"
        f"Work_end id is {work_end_id}; letters available: {', '.join(work_end_letters)}.\n"
        f"Available border side tokens: {', '.join(available_side_tokens)}.\n"
        "Images order: [1] full RGB scene, [2] full skeleton overlay, [3] work_end id overlay, [4] handle id overlay.\n"
        + extra
        + "\n"
        "Return JSON only:"
        '{"pattern":"T|I|REVERSE_L","work_end":{"id":<int>,"side":"<SIDE>"},"grip":{"id":<int>,"letter":"<L>"}}'
    )


def ask_vlm2_stage2_points(
    client: Any,
    object_name: str,
    instruction: str,
    scene_rgb: np.ndarray,
    skeleton_overlay: np.ndarray,
    work_id_overlay: np.ndarray,
    handle_id_overlay: np.ndarray,
    handle_id: int,
    work_end_id: int,
    handle_letters: List[str],
    work_end_letters: List[str],
    handle_middle_letters: List[str],
    available_side_tokens: List[str],
    geometry_hint_text: Optional[str] = None,
    target_relation_hint: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], str, Optional[Dict[str, Any]], Optional[str], str]:
    prompt = build_stage2_prompt_text(
        object_name=object_name,
        instruction=instruction,
        handle_id=handle_id,
        work_end_id=work_end_id,
        handle_letters=handle_letters,
        work_end_letters=work_end_letters,
        handle_middle_letters=handle_middle_letters,
        available_side_tokens=available_side_tokens,
        geometry_hint_text=geometry_hint_text,
        target_relation_hint=target_relation_hint,
    )
    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": encode_image_to_data_url(scene_rgb)},
        {"type": "input_image", "image_url": encode_image_to_data_url(skeleton_overlay)},
        {"type": "input_image", "image_url": encode_image_to_data_url(work_id_overlay)},
        {"type": "input_image", "image_url": encode_image_to_data_url(handle_id_overlay)},
    ]
    try:
        raw_text, api_route = query_vlm_content(client=client, model=MODEL_CFG["vlm_model"], content=content)
        parsed = parse_stage2_points_json(raw_text)
    except Exception as exc:
        return None, "", classify_openai_failure(exc), None, prompt

    middle_u = [str(x).strip().upper() for x in handle_middle_letters]
    work_u = [str(x).strip().upper() for x in work_end_letters]
    side_u = [str(x).strip().upper() for x in available_side_tokens]
    grip_u = middle_u or [str(x).strip().upper() for x in handle_letters]
    if _valid_stage2_semantic_only(parsed, grip_letters=grip_u, work_end_letters=work_u, side_tokens=side_u):
        return parsed, raw_text, None, api_route, prompt

    middle_pref_txt = ", ".join(middle_u) if middle_u else "(not available)"
    retry_prompt = (
        prompt
        + "\n\nCorrection rules:\n"
        + "- pattern must be exactly one of: T, I, REVERSE_L.\n"
        + f"- grip letter must be one of these middle-handle letters: {middle_pref_txt}.\n"
        + f"- work_end.side must be exactly one of: {', '.join(side_u)}.\n"
        + "- do not return arbitrary endpoint letters in place of side token.\n"
        + "- Return JSON only in the exact schema."
    )
    retry_content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": retry_prompt},
        {"type": "input_text", "text": f"Previous invalid answer:\n{raw_text}"},
        {"type": "input_image", "image_url": encode_image_to_data_url(scene_rgb)},
        {"type": "input_image", "image_url": encode_image_to_data_url(skeleton_overlay)},
        {"type": "input_image", "image_url": encode_image_to_data_url(work_id_overlay)},
        {"type": "input_image", "image_url": encode_image_to_data_url(handle_id_overlay)},
    ]
    try:
        raw_text2, api_route2 = query_vlm_content(client=client, model=MODEL_CFG["vlm_model"], content=retry_content)
        parsed2 = parse_stage2_points_json(raw_text2)
        if isinstance(parsed2, dict):
            parsed2["source"] = str(parsed2.get("source", "vlm")) + "+retry"
        return parsed2, raw_text2, None, api_route2, retry_prompt
    except Exception as exc:
        return parsed, raw_text, classify_openai_failure(exc), api_route, retry_prompt


def ask_vlm2_target_edge_points(
    client: Any,
    object_name: str,
    instruction: str,
    skeleton_overlay: np.ndarray,
    target_id_overlay: np.ndarray,
    target_id: int,
    target_letters: List[str],
    tool_touch_hint_global: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[Dict[str, Any]], str, Optional[Dict[str, Any]], Optional[str], str]:
    prompt = build_target_edge_prompt_text(
        object_name=object_name,
        instruction=instruction,
        target_id=target_id,
        target_letters=target_letters,
        tool_touch_hint_global=tool_touch_hint_global,
    )
    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": encode_image_to_data_url(skeleton_overlay)},
        {"type": "input_image", "image_url": encode_image_to_data_url(target_id_overlay)},
    ]
    try:
        raw_text, api_route = query_vlm_content(client=client, model=MODEL_CFG["vlm_model"], content=content)
        parsed = parse_target_edge_json(raw_text)
    except Exception as exc:
        return None, "", classify_openai_failure(exc), None, prompt

    target_u = [str(x).strip().upper() for x in target_letters]
    if _valid_target_letters_only(parsed, target_letters=target_u):
        return parsed, raw_text, None, api_route, prompt

    retry_prompt = (
        prompt
        + "\n\nCorrection rules:\n"
        + f"- target_edge letters must be two distinct letters from: {', '.join(target_u)}.\n"
        + "- Do not invent letters outside the allowed set.\n"
        + "- Return JSON only in the exact schema."
    )
    retry_content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": retry_prompt},
        {"type": "input_text", "text": f"Previous invalid answer:\n{raw_text}"},
        {"type": "input_image", "image_url": encode_image_to_data_url(skeleton_overlay)},
        {"type": "input_image", "image_url": encode_image_to_data_url(target_id_overlay)},
    ]
    try:
        raw_text2, api_route2 = query_vlm_content(client=client, model=MODEL_CFG["vlm_model"], content=retry_content)
        parsed2 = parse_target_edge_json(raw_text2)
        if isinstance(parsed2, dict):
            parsed2["source"] = str(parsed2.get("source", "vlm_target")) + "+retry"
        return parsed2, raw_text2, None, api_route2, retry_prompt
    except Exception as exc:
        return parsed, raw_text, classify_openai_failure(exc), api_route, retry_prompt

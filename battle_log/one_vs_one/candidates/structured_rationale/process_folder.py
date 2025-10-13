import json
import os
import re
import ast
import pandas as pd
from typing import Any, Dict, List, Optional

# --- helpers ---------------------------------------------------------------

def parse_rationale(r: Any) -> Dict[str, Any]:
    """Handle dict or stringified dict/JSON for 'rationale' → dict."""
    if isinstance(r, dict):
        return r
    if not isinstance(r, str):
        return {}
    s = r.strip()
    if not s:
        return {}
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    # Try Python literal first (many logs use single quotes)
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # Fallback: try JSON after a light single→double quote swap
    try:
        j = re.sub(r"(?<=[:\s\{\,])'", '"', s)
        j = re.sub(r"'(?=[\s\,\}])", '"', j)
        parsed = json.loads(j)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}

def strip_brackets(x: str) -> str:
    return re.sub(r"(\(.*?\)|\[.*?\])", "", x or "").strip()

def canonicalize_action(original: Optional[str]) -> str:
    """Normalize text like 'Shadow Ball (Move object)' → 'move: shadow ball' or 'Switch: Cinderace' → 'switch: cinderace'."""
    if not original:
        return ""
    o = original.strip()
    base = strip_brackets(o)
    low = o.lower()
    tag = None
    if "move object" in low or low.startswith("move:") or low.startswith("move "):
        tag = "move"
    elif "switch:" in low or "pokemon object" in low:
        tag = "switch"
    if low.startswith("move:"):
        tag = "move"; base = base.split(":", 1)[-1].strip()
    if low.startswith("switch:"):
        tag = "switch"; base = base.split(":", 1)[-1].strip()
    base = re.sub(r"\s+", " ", base).strip().lower()
    return f"{tag + ': ' if tag else ''}{base}"

def normalize_turn_id(turn_key: str) -> str:
    m = re.search(r"(\d+)", turn_key)
    return m.group(1) if m else turn_key

def match_best_from_candidates(best_pa: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Match Best to a candidate by canonicalized player_action."""
    best_norm = canonicalize_action(best_pa)
    for c in candidates:
        if canonicalize_action(c.get("player_action", "")) == best_norm:
            return c
    best_tail = best_norm.split(": ", 1)[-1]
    for c in candidates:
        can_tail = canonicalize_action(c.get("player_action", "")).split(": ", 1)[-1]
        if can_tail == best_tail:
            return c
    return None

# --- core logic (modified) -------------------------------------------------

# Set this True to keep your original label_scores != {} condition
REQUIRE_NONEMPTY_LABEL_SCORES = True

def build_turn_record(turn_key: str, turn_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a single turn-level record in the requested schema, or None if not eligible."""
    if not isinstance(turn_data, dict) or not turn_key.lower().startswith("turn "):
        return None

    candidates = turn_data.get("Candidates", []) or []
    # Filter candidates per your conditions
    filt = [
        c for c in candidates
        if c.get("score", 0) > -1 and (c.get("label_scores", {}) != {} if REQUIRE_NONEMPTY_LABEL_SCORES else True)
    ]
    if len(filt) <= 1:
        # Need candidate size > 1
        return None

    # Build candidate objects
    cands_out = []
    for i, c in enumerate(filt, start=1):
        cands_out.append({
            "action_id": f"cand_{i}",
            "player_action": canonicalize_action(c.get("player_action")),
            "opponent_action": canonicalize_action(c.get("opponent_action")) if c.get("opponent_action") else None,
            "StructuredRationale": parse_rationale(c.get("rationale", "")),
            "label_scores": c.get("label_scores", {}) or {}
        })

    # Best action (try to inherit from its matching candidate)
    best_raw = (turn_data.get("Best") or {})
    best_pa = best_raw.get("player_action", "")
    best_match = match_best_from_candidates(best_pa, filt)
    best_struct = {
        "action_id": None,
        "player_action": canonicalize_action(best_pa),
        "opponent_action": None,
        "StructuredRationale": {},
        "label_scores": best_raw.get("label_scores", {}) or {}
    }
    if best_match:
        best_norm = canonicalize_action(best_match.get("player_action", ""))
        for co in cands_out:
            if co["player_action"] == best_norm:
                best_struct["action_id"] = co["action_id"]
                break
        best_struct["opponent_action"] = canonicalize_action(best_match.get("opponent_action")) if best_match.get("opponent_action") else None
        best_struct["StructuredRationale"] = parse_rationale(best_match.get("rationale", ""))

    return {
        "turn_id": normalize_turn_id(turn_key),
        "Current_Game_states": turn_data.get("Current Game State", ""),
        "candidates": cands_out,
        "Best Action": best_struct
    }

def count_actions_in_file(filename, rows: List[Dict[str, Any]]) -> int:
    """Accumulate turn-level records into `rows` and return the action-count increment (same as your logic)."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_actions = 0
        for _, battle_data in data.items():
            for turn_key, turn_data in battle_data.items():
                if "Turn " in turn_key:
                    # For counting: keep your original per-candidate sum
                    candidates = turn_data.get("Candidates", []) or []
                    filtered_cands = [
                        cand for cand in candidates
                        if cand.get("score", 0) > -1 and (cand.get("label_scores", {}) != {} if REQUIRE_NONEMPTY_LABEL_SCORES else True)
                    ]
                    num_filtered = len(filtered_cands)
                    if num_filtered != 1:
                        total_actions += num_filtered

                    # For saving: store a turn-level record only if candidate size > 1
                    rec = build_turn_record(turn_key, turn_data)
                    if rec is not None:
                        rows.append(rec)

        return total_actions
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return 0

def count_all_actions_in_folder(folder_path):
    total_actions = 0
    rows: List[Dict[str, Any]] = []

    for filename in os.listdir(folder_path):
        if filename.startswith("score_evaluation_Battle_Id__") and filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            actions_in_file = count_actions_in_file(file_path, rows)
            total_actions += actions_in_file
            print(f"{filename}: {actions_in_file} actions")

    # Build a DataFrame where each row is a TURN record in your requested schema
    df = pd.DataFrame(rows, columns=["turn_id", "Current_Game_states", "candidates", "Best Action"])

    # Save outputs
    out_csv = os.path.join(folder_path, "filtered_turns_df.csv")
    out_jsonl = os.path.join(folder_path, "all_filtered_turns.jsonl")

    df.to_csv(out_csv, index=False)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSaved DataFrame with {len(df)} rows to: {out_csv}")
    print(f"Also wrote JSONL with the same rows to: {out_jsonl}")
    print(f"Total actions across all files (your counting logic): {total_actions}")

if __name__ == "__main__":
    count_all_actions_in_folder("../all_turn_data")

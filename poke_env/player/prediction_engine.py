import re
import numpy as np
from typing import Dict, Any, List

import pandas as pd
from numpy import ndarray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import torch
import joblib
from torch import Tensor


def _collect_roster(lines: list[str], start: int):
    """Collect consecutive 'Pokemon:' lines starting at `start`."""
    out = []
    i = start
    while i < len(lines) and lines[i].startswith("Pokemon:"):
        out.append(lines[i])
        i += 1
    return out, i


def _skip_blank(lines: list[str], i: int) -> int:
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    return i


def is_pokemon_fainted(pokemon_status_line: str) -> bool:
    """
    Checks if a Pokémon has fainted by looking for 'Status:fainted' or 'HP:0%'
    in its status line.

    Args:
        pokemon_status_line (str): The string describing the Pokémon's status.

    Returns:
        bool: True if the Pokémon has fainted, False otherwise.
    """
    return "Status:fainted" in pokemon_status_line or "HP:0%" in pokemon_status_line
def _collect_block_base(lines: List[str], i: int):
    out = []
    header = lines[i]
    out.append(header)
    i += 1

    if i < len(lines) and re.match(r"^[\w-]+ vs\. [\w-]+:$", lines[i].strip()):
        vs_line = lines[i]
        out.append(vs_line)
        m = re.match(r"^([\w-]+) vs\. ([\w-]+):$", vs_line.strip())
        ally = m.group(1) if m else None
        oppo = m.group(2) if m else None
        i += 1
    else:
        return out, i

    if i < len(lines) and (" outspeeds " in lines[i]):
        out.append(lines[i])
        i += 1

    j = i
    ally_moves_idx = None
    while j < len(lines):
        if lines[j].strip() in ("Requires switch:", "Current pokemon:"):
            break
        if ally and lines[j].strip() == f"{ally}'s moves:":
            ally_moves_idx = j
            break
        j += 1

    if ally_moves_idx is not None:
        out.append(lines[ally_moves_idx])
        j = ally_moves_idx + 1
        while j < len(lines):
            s = lines[j]
            if s.strip() in ("Requires switch:", "Current pokemon:"):
                break
            if oppo and s.strip().startswith("Opponent moves:"):
                break
            if "moves if" in s and "terastallize" in s:
                break
            if s.strip() == "":
                break
            out.append(s)
            j += 1
        i = j
    else:
        i = j

    k = i
    opp_moves_idx = None
    while k < len(lines):
        s = lines[k].strip()
        if s in ("Requires switch:", "Current pokemon:"):
            break
        if s.startswith("Opponent moves:"):
            opp_moves_idx = k
            break
        k += 1

    if opp_moves_idx is not None:
        out.append(lines[opp_moves_idx])
        k += 1
        while k < len(lines):
            s = lines[k]
            if s.strip() in ("Requires switch:", "Current pokemon:"):
                break
            if s.strip() == "":
                break
            if "moves if" in s and "terastallize" in s:
                break
            out.append(s)
            k += 1
        i = k
    else:
        i = k

    while out and out[-1].strip() == "":
        out.pop()
    return out, i


def compress_game_state_local(state_txt: str) -> str:
    """
    Deterministic, deletion-only compression to the target text format:
      1) Keep roster 'Pokemon:' lines (player then opponent), verbatim.
      2) Keep ONLY base-case parts of each 'Requires switch:' block.
      3) Keep ONLY base-case part of the single 'Current pokemon:' block.
      4) Keep the explanatory paragraph about 'terastallize'.
      5) Keep trailing 'Your current pokemon:' line(s).
    Drop ALL lines with conditional Tera variants (contain both 'moves if' and 'terastallize').
    Preserve wording and order; just delete unwanted lines and normalize blank lines.
    """
    lines = state_txt.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    n = len(lines)
    output: List[str] = []

    player_roster, i = _collect_roster(lines, i)
    output.extend(player_roster)
    output.append("")

    i = _skip_blank(lines, i)

    opp_roster, i = _collect_roster(lines, i)
    if opp_roster:
        output.extend(opp_roster)
        output.append("")

    i = _skip_blank(lines, i)

    blocks_out: List[List[str]] = []
    expl_start = None
    your_current_idx = None

    for idx in range(i, n):
        if lines[idx].startswith("Your current pokemon:"):
            your_current_idx = idx
            break
        if "'terastallize' changes a Pokemon's defensive typing" in lines[idx]:
            expl_start = idx
            break

    scan_end = min(
        your_current_idx if your_current_idx is not None else n,
        expl_start if expl_start is not None else n
    )

    j = i
    while j < scan_end:
        s = lines[j].strip()
        if s in ("Requires switch:", "Current pokemon:"):
            block, j2 = _collect_block_base(lines, j)
            blocks_out.append(block)
            j = j2
            while j < scan_end and lines[j].strip() == "":
                j += 1
        else:
            j += 1

    for b in blocks_out:
        if b:
            output.extend(b)
            output.append("")

    if expl_start is not None:
        end = your_current_idx if your_current_idx is not None else n
        expl_lines = lines[expl_start:end]
        while expl_lines and expl_lines[0].strip() == "":
            expl_lines.pop(0)
        while expl_lines and expl_lines[-1].strip() == "":
            expl_lines.pop()
        if expl_lines:
            output.extend(expl_lines)
            output.append("")

    if your_current_idx is not None:
        tail = lines[your_current_idx:]
        while tail and tail[0].strip() == "":
            tail.pop(0)
        if tail:
            output.extend(tail)

    while output and output[-1].strip() == "":
        output.pop()

    cleaned = []
    prev_blank = False
    for s in output:
        if s.strip() == "":
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(s)
            prev_blank = False

    return "\n".join(cleaned)


def extract_numerical_features(
        game_state_text: str,
        player_action: str,
        opponent_action: str
) -> Dict[str, float]:
    """
    Extracts numerical features, including the threat level for the specific action pair.
    This version ensures a consistent output format even if parsing fails.
    """

    def get_active_pokemon_from_state(text: str) -> str:
        """Helper to find the active Pokémon from the 'Current pokemon:' block."""
        match = re.search(r"[Cc]urrent pokemon:\s*\n(.*?)\s*vs\.", text)
        return match.group(1).strip() if match else None

    numerical_features = {}
    active_pokemon_name = ""

    # 1. Determine the active Pokémon based on the player's action
    if player_action.startswith("switch"):
        active_pokemon_name = player_action.split(" ")[1]

    elif player_action.startswith("move"):
        active_pokemon_name = get_active_pokemon_from_state(game_state_text)


    if not active_pokemon_name:
        print("Warning: Current Pokemon has been fainted, now you need to select another pokemon.")
        numerical_features["my_ko_turns"] = 99
        numerical_features["opponent_ko_turns"] = 1
        numerical_features["threat_level"] = -98  # (99 - 99)
        return numerical_features

    # Initialize with default "infinite" KO turns.
    my_ko_turns = 99
    opponent_ko_turns = 99

    # 2. Find the correct matchup block for the active Pokémon
    # This regex looks for a block starting with 'Requires switch:' or 'Current pokemon:'
    # and captures the content until the next block or the end of the text.
    block_match = re.search(
        # The pattern has been changed in two places below
        r"(?:Requires switch:|[Cc]?urrent pokemon:)\n({} vs\..*?)(?=\n\n(?:Requires switch:|[Cc]?urrent pokemon:)|$)".format(
            re.escape(active_pokemon_name)
        ),
        game_state_text,
        re.DOTALL
    )

    if block_match:
        block_text = block_match.group(1)

        # 3. Extract KO turns for the player's move
        if player_action.startswith("move"):
            move_name = player_action.split(" ", 1)[1]
            # This regex specifically looks for the base move, not conditional (Tera) variants.
            move_ko_match = re.search(
                r"^{}:\s*.*?(\d+|inf) turns to KO opponent's pokemon".format(re.escape(move_name)),
                block_text,
                re.MULTILINE
            )
            if move_ko_match:
                ko_val = move_ko_match.group(1)
                my_ko_turns = int(ko_val) if ko_val!= 'inf' else 99

        # 4. Extract KO turns for the opponent's move
        if opponent_action.startswith("move"):
            move_name = opponent_action.split(" ",1)[1]
            # First, isolate the 'Opponent moves:' subsection to avoid ambiguity
            opp_moves_text_match = re.search(r"Opponent moves:.*", block_text, re.DOTALL)
            if opp_moves_text_match:
                opp_moves_text = opp_moves_text_match.group(0)
                move_ko_match = re.search(
                    r"^{}:\s*.*?(\d+|inf) turns to KO your pokemon".format(re.escape(move_name)),
                    opp_moves_text,
                    re.MULTILINE
                )
                if move_ko_match:
                    ko_val = move_ko_match.group(1)
                    opponent_ko_turns = int(ko_val) if ko_val!= 'inf' else 99
    else:
        if is_pokemon_fainted(active_pokemon_name):
            print(f"{active_pokemon_name} has been fainted.")
            numerical_features["my_ko_turns"] = 99
            numerical_features["opponent_ko_turns"] = 1
            numerical_features["threat_level"] = -98.0  # A high threat (opponent KOs fast, I KO slow)

            # Return the default features immediately
            return numerical_features

    # 5. Populate the feature dictionary and calculate the threat level
    numerical_features['my_ko_turns'] = my_ko_turns
    numerical_features['opponent_ko_turns'] = opponent_ko_turns
    numerical_features['threat_level'] = float(opponent_ko_turns - my_ko_turns)

    return numerical_features

def l2norm_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalizes each row of a 2D matrix."""
    if x.ndim == 1:  # Handle the case of a single vector
        x = x.reshape(1, -1)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)

class PredictionEngine:
    def __init__(
            self,
            model_path: str = "./poke_env/data/static/prediction_engine_model/prediction_engine.pkl",
            scaler_path: str = "./poke_env/data/static/prediction_engine_model/scaler.pkl",
            pca_path: str = "./poke_env/data/static/prediction_engine_model/pca.pkl"
            # <-- Add path for the scaler
    ):
        """
        Initialize the PredictionEngine with a pre-trained model and scaler.

        Args:
            model_path (str): Path to the pre-trained model file (loaded via joblib).
            scaler_path (str): Path to the pre-trained scaler file (loaded via joblib).
        """
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.pca = joblib.load(pca_path)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2").to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model, scaler, or encoder: {e}")

    def get_embeddings(self, game_state_text: str, batch_size: int) -> ndarray | None:
        """
        Encode the game state text using the SentenceTransformer model.

        Args:
            game_state_text (str): The game state text description.

        Returns:
            np.ndarray: Embedding vector (384-dimensional for all-MiniLM-L6-v2).
        """
        if not game_state_text:
            return None

        with torch.no_grad():
            # 1. Encode the text(s)
            embeddings = self.encoder.encode(
                game_state_text,
                batch_size=batch_size,
                show_progress_bar=False,
                device=self.device
            )

            # 2. Convert to float32 and apply L2 normalization
            normalized_embeddings = l2norm_rows(embeddings.astype(np.float32))

        # 3. Return the final result
        return normalized_embeddings

    def predict(self, game_state_text: str, player_action: str, opponent_action: str) -> int:
        """
        Predict the score for a single action pair using pre-fitted objects.
        """
        try:
            # --- 1. Feature Extraction ---
            # A. Get numerical features as a dictionary
            numerical_features = extract_numerical_features(game_state_text, player_action, opponent_action)


            # B. Create an ordered numerical vector
            feature_order = ['my_ko_turns', 'opponent_ko_turns', 'threat_level']
            x_numerical = np.array(list(numerical_features.values()))

            # C. Create the text embedding
            state_embedding = self.get_embeddings(compress_game_state_local(game_state_text), 32)
            action_embedding = self.get_embeddings(player_action, 32)
            x_emb = np.array(np.mean([state_embedding, action_embedding], axis=0))

            # D. Apply the pre-fitted PCA to the text embedding
            x_text_pca = self.pca.transform(x_emb)  # Shape becomes (1, 8)

            # E. Combine into the final feature vector
            x_final = np.hstack([x_numerical.reshape(1, -1), x_text_pca])  # Shape becomes (1, 11)

            # F. Apply the pre-fitted scaler
            x_scaled = self.scaler.transform(x_final)

            # --- 3. Predict ---
            y_pred = self.model.predict(x_scaled)

            score = int(y_pred[0])

            # Clamp to 1-100
            score = max(1, min(100, score))
            return score

        except Exception as e:
            print(f"Prediction failed: {e}, returning default score")
            return 50
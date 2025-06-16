#!/usr/bin/env python3
"""
one_vs_one_round_robin.py
Play exactly one 1-v-1 battle for every OU team file in
poke_env/data/static/teams/.

Both agents use the *same* team each battle, so RNG in play decisions
is the only difference.
"""

import argparse
import asyncio
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm

from common import prompt_algos, PNUMBER1        # your project globals
from poke_env.player.team_util import get_llm_player, load_random_team

# ───────────────────────────── argparse ──────────────────────────────
# Player arguments
parser = argparse.ArgumentParser()
parser.add_argument("--player_prompt_algo", default="minimax", choices=prompt_algos)
parser.add_argument("--player_backend", type=str, default="gpt-4o", choices=["gpt-4o-mini", "gpt-4o", "gpt-4o-2024-05-13", "llama", 'None'])
parser.add_argument("--player_name", type=str, default='pokechamp', choices=['pokechamp', 'pokellmon', 'one_step', 'abyssal', 'max_power', 'random'])
parser.add_argument("--player_device", type=int, default=0)

# Opponent arguments
parser.add_argument("--opponent_prompt_algo", default="io", choices=prompt_algos)
parser.add_argument("--opponent_backend", type=str, default="gpt-4o", choices=["gpt-4o-mini", "gpt-4o", "gpt-4o-2024-05-13", "llama", 'None'])
parser.add_argument("--opponent_name", type=str, default='pokellmon', choices=['pokechamp', 'pokellmon', 'one_step', 'abyssal', 'max_power', 'random'])
parser.add_argument("--opponent_device", type=int, default=0)

# Shared arguments
parser.add_argument("--temperature", type=float, default=0.3)
parser.add_argument("--battle_format", default="gen8ou", choices=["gen8randombattle", "gen8ou", "gen9ou", "gen9randombattle"])
parser.add_argument("--log_dir", type=str, default="./battle_log/one_vs_one")

args = parser.parse_args()
# ───────────────────────── helper to list team files ─────────────────
TEAM_DIR = Path("poke_env/data/static/teams")
def iter_team_ids(fmt_prefix: str = "gen8ou") -> list[int]:
    """
    Return the numeric IDs that appear *after* `fmt_prefix` in every
    team file name like  'gen9ou12.txt'  →  [1, 2, 3 … 12].
    """
    ids = []
    pattern = re.compile(rf"{re.escape(fmt_prefix)}(\d+)$")   # digits at end of stem
    for p in TEAM_DIR.glob(f"{fmt_prefix}[0-9]*.txt"):
        m = pattern.match(p.stem)
        if m:
            ids.append(int(m.group(1)))       # only the trailing number
    return sorted(ids)

# ───────────────────────────── main loop ─────────────────────────────
async def main() -> None:
    # create two LLM players
    player = get_llm_player(args,
                            args.player_backend,
                            args.player_prompt_algo,
                            args.player_name,
                            device=args.player_device,
                            PNUMBER1=PNUMBER1,  # for name uniqueness locally
                            battle_format=args.battle_format)

    opponent = get_llm_player(args,
                              args.opponent_backend,
                              args.opponent_prompt_algo,
                              args.opponent_name,
                              device=args.opponent_device,
                              PNUMBER1=PNUMBER1,  # for name uniqueness locally
                              battle_format=args.battle_format)

    team_ids = iter_team_ids(args.battle_format)
    if not team_ids:
        raise FileNotFoundError(f"No '{args.battle_format}*.txt' files in {TEAM_DIR}")

    pbar = tqdm(total=len(team_ids), desc="Battles")
    tid = 2

    team_str = load_random_team(tid,args.battle_format)      # passes `id` → loads exact file
    print(team_str)
    player.update_team(team_str)
    opponent.update_team(team_str)

    # randomise who challenges whom so they alternate first‐turn priority
    if np.random.rand() > 0.5:
        await player.battle_against(opponent, n_battles=1)
    else:
        await opponent.battle_against(player, n_battles=1)

    pbar.set_description(f"Win-rate {player.win_rate*100:.2f}% (team {tid})")
    pbar.update(1)

    print(f"\nFinal Player-A win-rate: {player.win_rate*100:.2f}% over {len(team_ids)} games")

# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())

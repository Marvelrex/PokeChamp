# import ast  # Legacy: unused in current implementation
from copy import copy  # Updated: deepcopy no longer required
# from copy import copy, deepcopy  # Original import retained for reference
# import datetime  # Legacy: unused
import json
import os
# import random  # Legacy: unused
# import sys  # Legacy: unused

import numpy as np
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
# from poke_env.environment.double_battle import DoubleBattle  # Legacy: unused
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
# from poke_env.environment.side_condition import SideCondition  # Legacy: unused
from poke_env.player.ollama_player import OllamaPlayer
from poke_env.player.player import Player, BattleOrder
# from typing import Callable, Dict, List, Optional, Tuple, Union  # Legacy import breadth
from typing import Callable, Dict, Tuple
from poke_env.environment.move import Move
import time
# import json  # Duplicate legacy import
# import joblib  # Legacy: unused post-pickle loading handled in PredictionEngine
from poke_env.data.gen_data import GenData
from poke_env.player.gpt_player import GPTPlayer
# from poke_env.player.llama_player import LLAMAPlayer  # Legacy backend helper
from poke_env.player.local_simulation import LocalSim, SimNode
from difflib import get_close_matches
# from lightgbm import LGBMRegressor  # Legacy: not directly instanced here

from poke_env.player.prediction_engine import PredictionEngine
from poke_env.player.prompts import get_number_turns_faint, get_status_num_turns_fnt, state_translate, get_gimmick_motivation

DEBUG=False

class LLMPlayer(Player):
    def __init__(self,
                 battle_format,
                 api_key="",
                 backend="gpt-4-1106-preview",
                 temperature=1.0,
                 prompt_algo="io",
                 log_dir=None,
                 team=None,
                 save_replays=None,
                 account_configuration=None,
                 server_configuration=None,
                 K=2,
                 _use_strat_prompt=False,
                 _use_prediction_engine="False",
                 prompt_translate: Callable=state_translate,
                 device=0,
                 llm_backend=None
                 ):

        super().__init__(battle_format=battle_format,
                         team=team,
                         save_replays=save_replays,
                         account_configuration=account_configuration,
                         server_configuration=server_configuration)

        self._reward_buffer: Dict[AbstractBattle, float] = {}
        self._battle_last_action : Dict[AbstractBattle, Dict] = {}
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.backend = backend
        self.temperature = temperature
        self.log_dir = log_dir
        self.api_key = api_key
        self.prompt_algo = prompt_algo
        self.gen = GenData.from_format(battle_format)
        self.genNum = self.gen.gen
        self.prompt_translate = prompt_translate
        self.strategy_prompt = ""
        self.team_str = team
        self.use_strat_prompt = _use_strat_prompt
        self.use_prediction_engine = _use_prediction_engine
        self.prediction_engine = None
        self.slm = None


        with open("./poke_env/data/static/moves/moves_effect.json", "r") as f:
            self.move_effect = json.load(f)
        # only used in old prompting method, replaced by statistcal sets data
        with open(f"./poke_env/data/static/moves/gen8pokemon_move_dict.json", "r") as f:
            self.pokemon_move_dict = json.load(f)
        with open("./poke_env/data/static/abilities/ability_effect.json", "r") as f:
            self.ability_effect = json.load(f)
        # only used is old prompting method
        with open("./poke_env/data/static/abilities/gen8pokemon_ability_dict.json", "r") as f:
            self.pokemon_ability_dict = json.load(f)
        with open("./poke_env/data/static/items/item_effect.json", "r") as f:
            self.item_effect = json.load(f)
        # unused
        # with open(f"./poke_env/data/static/items/gen8pokemon_item_dict.json", "r") as f:
        #     self.pokemon_item_dict = json.load(f)
        self.pokemon_item_dict = {}
        with open(f"./poke_env/data/static/pokedex/gen{self.gen.gen}pokedex.json", "r") as f:
            self._pokemon_dict = json.load(f)

        self.last_plan = ""

        if llm_backend is None:
            backend_l = backend.lower()
            if "gpt" in backend_l:
                self.llm = GPTPlayer(self.api_key)
            elif ("gemma" in backend_l) or ("llama" in backend_l):
                self.llm = OllamaPlayer(model=backend)
            else:
                raise ValueError(f"LLM type not implemented: {backend}")
        else:
            self.llm = llm_backend
        self.llm_value = self.llm
        self.K = K      # for minimax, SC, ToT

        if _use_prediction_engine != "False":
            try:
                if _use_prediction_engine == "high":
                    self.prediction_engine = PredictionEngine("poke_env/data/static/prediction_engine_model/High_Optune_LGBM.pkl",
                                                              "poke_env/data/static/prediction_engine_model/LGBM_scaler.pkl",
                                                              "poke_env/data/static/prediction_engine_model/LGBM_pca.joblib")
                elif _use_prediction_engine == "low":
                    self.prediction_engine = PredictionEngine(
                    "poke_env/data/static/prediction_engine_model/Low_Optune_Ridge.pkl",
                        "poke_env/data/static/prediction_engine_model/Ridge_scaler.pkl",
                "poke_env/data/static/prediction_engine_model/RIDGE_pca.joblib")
            except ModuleNotFoundError as missing_dependency:
                # raise RuntimeError("Failed to load prediction engine") from e
                print(f"Prediction engine disabled: missing optional dependency ({missing_dependency}).")
                self.prediction_engine = None
                self.use_prediction_engine = "False"
            except Exception as e:
                # up to you: either raise or log and keep going
                raise RuntimeError("Failed to load prediction engine") from e

    def get_LLM_action(self, system_prompt, user_prompt, model, temperature=0.7, json_format=False, seed=None, stop=None, max_tokens=200, actions=None, llm=None, battle=None) -> str:
        # stop previously defaulted to [], which could retain state between calls.
        if stop is None:
            stop = []
        else:
            stop = list(stop)
        if llm is None:
            output, _ = self.llm.get_LLM_action(system_prompt, user_prompt, model, temperature, True, seed, stop, max_tokens=max_tokens, actions=actions)
        else:
            output, _ = llm.get_LLM_action(system_prompt, user_prompt, model, temperature, True, seed, stop, max_tokens=max_tokens, actions=actions)
        rationale_text = None
        parsed_json = None
        if output.lstrip().startswith("{"):
            try:
                parsed_json = json.loads(output)
            except Exception:
                # not strict JSON or no thought field
                print("JSON PARSED ERROR:",output)
                pass
        if parsed_json is not None:
            self.log_rationale(
                battle=battle,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                llm_action_json=parsed_json,
                player_name= self.ps_client.account_configuration.username
            )
        return output

    
    def check_all_pokemon(self, pokemon_str: str) -> Pokemon:
        valid_pokemon = None
        if pokemon_str in self._pokemon_dict:
            valid_pokemon = pokemon_str
        else:
            closest = get_close_matches(pokemon_str, self._pokemon_dict.keys(), n=1, cutoff=0.8)
            if len(closest) > 0:
                valid_pokemon = closest[0]
        if valid_pokemon is None:
            return None
        # pokemon = Pokemon(species=pokemon_str, gen=self.genNum)  # Original: used raw user string
        pokemon = Pokemon(species=valid_pokemon, gen=self.genNum)
        return pokemon

    # def getCandidateActionScores(
    #         self,
    #         best_action,
    #         predicted_opp_action,
    #         score,
    #         rationale,
    #         summary_list,
    #         battle,
    #         state_prompt,
    # ):
    #     # ---------- Friendly strings for chosen actions ----------
    #     player_msg = best_action.message.replace("/choose ", "")
    #     if player_msg.startswith("move "):
    #         move_name = player_msg.split(" ", 1)[1]
    #         player_action_desc = f"Move: {move_name.capitalize()}"
    #     elif player_msg.startswith("switch "):
    #         mon_name = player_msg.split(" ", 1)[1]
    #         player_action_desc = f"Switch: {mon_name}"
    #     else:
    #         player_action_desc = player_msg.capitalize()
    #
    #     if predicted_opp_action:
    #         opp_msg = predicted_opp_action.message.replace("/choose ", "")
    #         if opp_msg.startswith("move "):
    #             opp_move = opp_msg.split(" ", 1)[1]
    #             opp_action_desc = f"Move: {opp_move.capitalize()}"
    #         elif opp_msg.startswith("switch "):
    #             opp_mon = opp_msg.split(" ", 1)[1]
    #             opp_action_desc = f"Switch: {opp_mon}"
    #         else:
    #             opp_action_desc = opp_msg.capitalize()
    #     else:
    #         opp_action_desc = "None"
    #
    #     # ---------- Prepare candidate summary list if missing ----------
    #     if not summary_list:
    #         summary_list = [{
    #             "player_action": best_action,
    #             "score": int(score) if isinstance(score, (int, float)) else score,
    #             "rationale": rationale,
    #             "opponent_action": predicted_opp_action,
    #         }]
    #
    #     # ---------- Helpers for JSON-safe candidates ----------
    #     def _to_msg(a):
    #         try:
    #             m = a.message if hasattr(a, "message") else str(a)
    #             return m.replace("/choose ", "")
    #         except Exception:
    #             return str(a)
    #
    #     def _to_int_if_num(x):
    #         try:
    #             if isinstance(x, bool):
    #                 return x
    #             if isinstance(x, (int, float)):
    #                 return int(x)
    #             return int(float(x))
    #         except Exception:
    #             return x
    #
    #     json_safe_summary_list = []
    #     for c in summary_list:
    #         if isinstance(c, dict):
    #             d = dict(c)
    #             if "player_action" in d:
    #                 d["player_action"] = _to_msg(d["player_action"])
    #             if "opponent_action" in d:
    #                 d["opponent_action"] = _to_msg(d["opponent_action"])
    #             d["score"] = _to_int_if_num(d.get("score"))
    #             # rationale may be string or dict; keep as-is but JSON-serializable
    #             r = d.get("rationale")
    #             d["rationale"] = "" if r is None else (
    #                 r if isinstance(r, (dict, list, str, int, float, bool)) else str(r))
    #             # pass through label_scores if present
    #             if "label_scores" in d and not isinstance(d["label_scores"], (dict, type(None))):
    #                 # ensure dict or drop
    #                 try:
    #                     from json import loads as _loads
    #                     d["label_scores"] = _loads(d["label_scores"]) if isinstance(d["label_scores"], str) else None
    #                 except Exception:
    #                     d["label_scores"] = None
    #             json_safe_summary_list.append(d)
    #         else:
    #             json_safe_summary_list.append({
    #                 "player_action": _to_msg(c),
    #                 "score": None,
    #                 "rationale": "",
    #                 "opponent_action": "None",
    #                 "label_scores": None,
    #             })
    #     summary_list = json_safe_summary_list
    #
    #     # ---------- Derive state string ----------
    #     try:
    #         _sp = state_prompt
    #     except NameError:
    #         _sp = ""
    #     state_str = (
    #         _sp.split("Current battle state:\n", 1)[1]
    #         if isinstance(_sp, str) and "Current battle state:" in _sp
    #         else (_sp if isinstance(_sp, str) else "")
    #     )
    #
    #     # ---------- Best entry, include label_scores if we can match ----------
    #     score_val = int(score) if isinstance(score, (int, float)) else score
    #
    #     # Try to find the candidate matching the chosen player action to carry label_scores
    #     def _friendly(msg_no_prefix: str) -> str:
    #         if msg_no_prefix.startswith("move "):
    #             return f"Move: {msg_no_prefix.split(' ', 1)[1].capitalize()}"
    #         if msg_no_prefix.startswith("switch "):
    #             return f"Switch: {msg_no_prefix.split(' ', 1)[1]}"
    #         return msg_no_prefix.capitalize()
    #
    #     best_label_scores = None
    #     # Create comparable forms for matching
    #     chosen_norm = player_action_desc.lower().strip()
    #     for cand in summary_list:
    #         cand_norm = _friendly(cand.get("player_action", "")).lower().strip()
    #         if cand_norm == chosen_norm and "label_scores" in cand:
    #             best_label_scores = cand.get("label_scores")
    #             break
    #
    #     # ---------- Build turn entry ----------
    #     turn_entry = {
    #         "Current Game State": state_str,
    #         "Candidates": summary_list,
    #         "Best": {
    #             "player_action": player_action_desc,
    #             "score": score_val,
    #             "label_scores": best_label_scores,  # may be None if unavailable
    #         }
    #     }
    #
    #     # ---------- File path & write ----------
    #     player_name = getattr(self.ps_client.account_configuration, "username", "player")
    #
    #     battle_id_str = "".join(filter(str.isdigit, battle.battle_tag)) or battle.battle_tag
    #     battle_key = (
    #         f"Battle Id: {int(battle_id_str):03d}"
    #         if battle_id_str.isdigit()
    #         else f"Battle Id: {battle_id_str}"
    #     )
    #
    #     if battle.finished:
    #         if battle.won is None:
    #             outcome_tag = "tie"
    #         else:
    #             outcome_tag = "winner" if battle.won else "loser"
    #     else:
    #         outcome_tag = "inprogress"
    #
    #     safe_key = battle_key.replace(":", "_").replace("/", "_").replace(" ", "_")
    #     file_name = f"score_evaluation_{safe_key}_{player_name}_{outcome_tag}.json"
    #
    #     if self.log_dir is None:
    #         os.makedirs("candidates", exist_ok=True)  # ensure dir
    #         log_path = os.path.join("candidates", file_name)
    #     else:
    #         os.makedirs(os.path.join(self.log_dir, "candidates"), exist_ok=True)
    #         log_path = os.path.join(self.log_dir, "candidates", file_name)
    #
    #         # rename inprogress → final tag within the same candidates dir
    #         if battle.finished and outcome_tag != "inprogress":
    #             tmp_name = f"score_evaluation_{safe_key}_{player_name}_inprogress.json"
    #             tmp_path = (
    #                 os.path.join("candidates", tmp_name)
    #                 if self.log_dir is None
    #                 else os.path.join(self.log_dir, "candidates", tmp_name)
    #             )
    #             if os.path.exists(tmp_path):
    #                 os.replace(tmp_path, log_path)
    #
    #     # Read, update, write atomically
    #     try:
    #         with open(log_path, "r", encoding="utf-8") as f:
    #             data = json.load(f)
    #     except FileNotFoundError:
    #         data = {}
    #     except json.JSONDecodeError:
    #         data = {}
    #
    #     if battle_key not in data:
    #         data[battle_key] = {}
    #
    #     data[battle_key][f"Turn {battle.turn}"] = turn_entry
    #
    #     tmp_out = f"{log_path}.tmp"
    #     with open(tmp_out, "w", encoding="utf-8") as f:
    #         json.dump(data, f, indent=4, ensure_ascii=False)
    #     os.replace(tmp_out, log_path)

    def choose_move(self, battle: AbstractBattle):
        self.current_turn = battle.turn
        self.current_battle_tag = battle.battle_tag
        use_prediction_engine = self.use_prediction_engine
        sim = LocalSim(battle,
                    self.move_effect,
                    self.pokemon_move_dict,
                    self.ability_effect,
                    self.pokemon_ability_dict,
                    self.item_effect,
                    self.pokemon_item_dict,
                    self.gen,
                    self._dynamax_disable,
                    self.strategy_prompt,
                    format=self.format,
                    prompt_translate=self.prompt_translate
        )
        if battle.turn <=1 and self.use_strat_prompt:
            self.strategy_prompt = sim.get_llm_system_prompt(self.format, self.llm, team_str=self.team_str, model='gpt-4o-2024-05-13')
        
        if battle.active_pokemon:
            if battle.active_pokemon.fainted and len(battle.available_switches) == 1:
                next_action = BattleOrder(battle.available_switches[0])
                return next_action
            elif not battle.active_pokemon.fainted and len(battle.available_moves) == 1 and len(battle.available_switches) == 0:
                return self.choose_max_damage_move(battle)
        elif len(battle.available_moves) <= 1 and len(battle.available_switches) == 0:
            return self.choose_max_damage_move(battle)

        system_prompt, state_prompt, state_action_prompt = sim.state_translate(battle) # add lower case
        moves = [move.id for move in battle.available_moves]
        switches = [pokemon.species for pokemon in battle.available_switches]
        actions = [moves, switches]

        gimmick_output_format = ''
        if 'pokellmon' not in self.ps_client.account_configuration.username: # make sure we dont mess with pokellmon original strat
            # gimmick_output_format = f'{f' or {{"dynamax":"<move_name>"}}' if battle.can_dynamax else ''}{f' or {{"terastallize":"<move_name>"}}' if battle.can_tera else ''}'  # Original malformed nesting
            dynamax_fragment = ' or {"dynamax":"<move_name>"}' if battle.can_dynamax else ''
            tera_fragment = ' or {"terastallize":"<move_name>"}' if battle.can_tera else ''
            gimmick_output_format = f"{dynamax_fragment}{tera_fragment}"

        if battle.active_pokemon.fainted or len(battle.available_moves) == 0:

            constraint_prompt_io = '''Choose the most suitable pokemon to switch. Your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}\n'''
            constraint_prompt_cot = '''Choose the most suitable pokemon to switch by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n'''
            constraint_prompt_tot_1 = '''Generate top-k (k<=3) best switch options. Your output MUST be a JSON like:{"option_1":{"action":"switch","target":"<switch_pokemon_name>"}, ..., "option_k":{"action":"switch","target":"<switch_pokemon_name>"}}\n'''
            constraint_prompt_tot_2 = '''Select the best option from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:{"decision":{"action":"switch","target":"<switch_pokemon_name>"}}\n'''
        elif len(battle.available_switches) == 0:
            constraint_prompt_io = f'''Choose the best action and your output MUST be a JSON like: {{"move":"<move_name>"}}{gimmick_output_format}\n'''
            constraint_prompt_cot = '''Choose the best action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "move":"<move_name>"} or {"thought":"<step-by-step-thinking>"}\n'''
            constraint_prompt_tot_1 = '''Generate top-k (k<=3) best action options. Your output MUST be a JSON like: {"option_1":{"action":"<move>", "target":"<move_name>"}, ..., "option_k":{"action":"<move>", "target":"<move_name>"}}\n'''
            constraint_prompt_tot_2 = '''Select the best action from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:"decision":{"action":"<move>", "target":"<move_name>"}\n'''
        else:
            constraint_prompt_io = f'''Choose the best action and your output MUST be a JSON like: {{"move":"<move_name>"}}{gimmick_output_format} or {{"switch":"<switch_pokemon_name>"}}\n'''
            constraint_prompt_cot = '''Choose the best action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "move":"<move_name>"} or {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n'''
            constraint_prompt_tot_1 = '''Generate top-k (k<=3) best action options. Your output MUST be a JSON like: {"option_1":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}, ..., "option_k":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}}\n'''
            constraint_prompt_tot_2 = '''Select the best action from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:"decision":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}\n'''

        state_prompt_io = state_prompt + state_action_prompt + constraint_prompt_io
        state_prompt_cot = state_prompt + state_action_prompt + constraint_prompt_cot
        state_prompt_tot_1 = state_prompt + state_action_prompt + constraint_prompt_tot_1
        state_prompt_tot_2 = state_prompt + state_action_prompt + constraint_prompt_tot_2

        retries = 2
        # Chain-of-thought
        if self.prompt_algo == "io":
            return self.io(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim, actions=actions)
        # Self-consistency with k = 3
        elif self.prompt_algo == "sc":
            return self.sc(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim)

        # Tree of thought, k = 3
        elif self.prompt_algo == "tot":
            llm_output1 = ""
            next_action = None
            for i in range(retries):
                try:
                    llm_output1 = self.get_LLM_action(system_prompt=system_prompt,
                                               user_prompt=state_prompt_tot_1,
                                               model=self.backend,
                                               temperature=self.temperature,
                                               max_tokens=200,
                                               json_format=True,
                                               battle = battle)
                    break
                except:
                    raise ValueError('No valid move', battle.active_pokemon.fainted, len(battle.available_switches))
                    continue

            if llm_output1 == "":
                return self.choose_max_damage_move(battle)

            for i in range(retries):
                try:
                    llm_output2 = self.get_LLM_action(system_prompt=system_prompt,
                                               user_prompt=state_prompt_tot_2.replace("[OPTIONS]", llm_output1),
                                               model=self.backend,
                                               temperature=self.temperature,
                                               max_tokens=100,
                                               json_format=True,
                                               battle = battle)

                    next_action = self.parse_new(llm_output2, battle, sim)
                    # with open(f"{self.log_dir}/output.jsonl", "a") as f:  # Original: assumed log_dir always set
                    #     f.write(json.dumps({"turn": battle.turn,
                    #                         "system_prompt": system_prompt,
                    #                         "user_prompt1": state_prompt_tot_1,
                    #                         "user_prompt2": state_prompt_tot_2,
                    #                         "llm_output1": llm_output1,
                    #                         "llm_output2": llm_output2,
                    #                         "battle_tag": battle.battle_tag
                    #                         }) + "\n")
                    if self.log_dir is not None:
                        os.makedirs(self.log_dir, exist_ok=True)
                        with open(os.path.join(self.log_dir, "output.jsonl"), "a", encoding="utf8") as f:
                            f.write(json.dumps({"turn": battle.turn,
                                                "system_prompt": system_prompt,
                                                "user_prompt1": state_prompt_tot_1,
                                                "user_prompt2": state_prompt_tot_2,
                                                "llm_output1": llm_output1,
                                                "llm_output2": llm_output2,
                                                "battle_tag": battle.battle_tag
                                                }) + "\n")
                    if next_action is not None:     break
                except:
                    raise ValueError('No valid move', battle.active_pokemon.fainted, len(battle.available_switches))
                    continue

            if next_action is None:
                next_action = self.choose_max_damage_move(battle)
            return next_action
        elif self.prompt_algo == "minimax":
            try:
                # Use tree_search with return_opp=True to get player action, opponent action, score, and rationale
                # best_action, predicted_opp_action, score, rationale, summary_list, best_label_scores = self.tree_search(
                #     retries,
                #     battle,
                #     return_opp=True,
                #     use_pred_engine=use_prediction_engine,
                # )
                tree_result = self.tree_search(
                    retries,
                    battle,
                    return_opp=True,
                    use_pred_engine=use_prediction_engine,
                )

                best_action, predicted_opp_action, score, rationale, summary_list, best_label_scores = tree_result

                # Format the chosen player action for readability
                player_msg = best_action.message.replace("/choose ", "")
                if player_msg.startswith("move "):
                    move_name = player_msg.split(" ", 1)[1]
                    player_action_desc = f"Move: {move_name.capitalize()}"
                elif player_msg.startswith("switch "):
                    mon_name = player_msg.split(" ", 1)[1]
                    player_action_desc = f"Switch: {mon_name}"
                else:
                    player_action_desc = player_msg.capitalize()

                # Format the predicted opponent action
                if predicted_opp_action:
                    opp_msg = predicted_opp_action.message.replace("/choose ", "")
                    if opp_msg.startswith("move "):
                        opp_move = opp_msg.split(" ", 1)[1]
                        opp_action_desc = f"Move: {opp_move.capitalize()}"
                    elif opp_msg.startswith("switch "):
                        opp_mon = opp_msg.split(" ", 1)[1]
                        opp_action_desc = f"Switch: {opp_mon}"
                    else:
                        opp_action_desc = opp_msg.capitalize()
                else:
                    opp_action_desc = "None"

                # Ensure summary_list is not empty (in case of early return or fallback)
                if not summary_list:
                    summary_list = [{
                        "player_action": str(best_action.order) if hasattr(best_action, "order") else str(best_action),
                        "score": int(score) if isinstance(score, (int, float)) else score,
                        "rationale": rationale or "",
                        "opponent_action": str(predicted_opp_action.order) if predicted_opp_action else None,
                        "label_scores": best_label_scores or {}
                    }]

                # Make summary_list JSON-serializable (convert any objects to primitive types)
                def _to_msg(action):
                    try:
                        msg = action.message if hasattr(action, "message") else str(action)
                        return msg.replace("/choose ", "")
                    except Exception:
                        return str(action)

                def _to_int_if_num(x):
                    try:
                        return int(x)
                    except Exception:
                        return x

                json_safe_summary_list = []
                for cand in summary_list:
                    if isinstance(cand, dict):
                        entry = dict(cand)
                        if "player_action" in entry:
                            entry["player_action"] = _to_msg(entry["player_action"])
                        if "opponent_action" in entry:
                            entry["opponent_action"] = _to_msg(entry["opponent_action"])
                        entry["score"] = _to_int_if_num(entry.get("score"))
                        entry["rationale"] = "" if entry.get("rationale") is None else str(entry.get("rationale"))
                        # Ensure label_scores values are int
                        if "label_scores" in entry and isinstance(entry["label_scores"], dict):
                            entry["label_scores"] = {k: _to_int_if_num(v) for k, v in entry["label_scores"].items()}
                        json_safe_summary_list.append(entry)
                    else:
                        # Handle any non-dict candidates (should not happen in current logic)
                        json_safe_summary_list.append({
                            "player_action": _to_msg(cand),
                            "score": None,
                            "rationale": "",
                            "opponent_action": "None",
                            "label_scores": {}
                        })
                summary_list = json_safe_summary_list

                # Build the turn entry for logging
                score_val = _to_int_if_num(score)
                # Extract current game state string if available
                try:
                    state_str = state_prompt.split("Current battle state:\n", 1)[1]
                except Exception:
                    state_str = state_prompt if isinstance(state_prompt, str) else ""

                turn_entry = {
                    "Current Game State": state_str,
                    "Candidates": summary_list,
                    "Best": {
                        "player_action": player_action_desc,
                        "score": score_val,
                        "label_scores": {k: _to_int_if_num(v) for k, v in (best_label_scores or {}).items()}
                    }
                }

                # Determine log file path (battle_id and outcome status)
                player_name = getattr(self.ps_client.account_configuration, "username", "player")
                battle_id_str = "".join(filter(str.isdigit, battle.battle_tag)) or battle.battle_tag
                battle_key = f"Battle Id: {int(battle_id_str):03d}" if battle_id_str.isdigit() else f"Battle Id: {battle_id_str}"
                outcome_tag = "inprogress"
                if battle.finished:
                    if battle.won is None:
                        outcome_tag = "tie"
                    else:
                        outcome_tag = "winner" if battle.won else "loser"
                # Prepare file name and path
                safe_key = battle_key.replace(":", "_").replace("/", "_").replace(" ", "_")
                file_name = f"score_evaluation_{safe_key}_{player_name}_{outcome_tag}.json"
                log_dir = self.log_dir or ""
                if log_dir:
                    os.makedirs(os.path.join(log_dir, "candidates"), exist_ok=True)
                log_path = os.path.join(log_dir, "candidates", file_name)

                # If battle finished, rename any in-progress file to final outcome
                if battle.finished and outcome_tag != "inprogress":
                    tmp_inprogress = f"score_evaluation_{safe_key}_{player_name}_inprogress.json"
                    tmp_path = os.path.join(log_dir, "candidates", tmp_inprogress) if log_dir else tmp_inprogress
                    if os.path.exists(tmp_path):
                        os.replace(tmp_path, log_path)

                # Load existing data if file exists
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    data = {}
                if battle_key not in data:
                    data[battle_key] = {}
                data[battle_key][f"Turn {battle.turn}"] = turn_entry

                # Write updated log atomically
                tmp_out = f"{log_path}.tmp"
                with open(tmp_out, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                os.replace(tmp_out, log_path)

                # Return the chosen action to execute
                return best_action

            except Exception as e:
                import traceback
                print("minimax step failed. Using dmg calc")
                print("Exception:", e)
                print("Traceback:", traceback.format_exc())
                return self.choose_max_damage_move(battle)


    def io(self, retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle: Battle, sim, dont_verify=False, actions=None):
        next_action = None
        # cot_prompt = 'In fewer than 3 sentences, let\'s think step by step:'
        state_prompt_io = state_prompt + state_action_prompt + constraint_prompt_cot

        for i in range(retries):
            try:
                llm_output = self.get_LLM_action(system_prompt=system_prompt,
                                            user_prompt=state_prompt_io,
                                            model=self.backend,
                                            temperature=self.temperature,
                                            max_tokens=300,
                                            # stop=["reason"],
                                            json_format=True,
                                            actions=actions, battle=battle)

                # load when llm does heavylifting for parsing
                llm_action_json = json.loads(llm_output)
                next_action = None

                dynamax = "dynamax" in llm_action_json.keys()
                tera = "terastallize" in llm_action_json.keys()
                is_a_move = dynamax or tera

                if "move" in llm_action_json.keys() or is_a_move:
                    if dynamax:
                        llm_move_id = llm_action_json["dynamax"].strip()
                    elif tera:
                        llm_move_id = llm_action_json["terastallize"].strip()
                    else:
                        llm_move_id = llm_action_json["move"].strip()
                    move_list = battle.active_pokemon.moves.values()
                    if dont_verify: # opponent
                        move_list = battle.opponent_active_pokemon.moves.values()
                    for i, move in enumerate(move_list):
                        if move.id.lower().replace(' ', '') == llm_move_id.lower().replace(' ', ''):
                            #next_action = self.create_order(move, dynamax=sim._should_dynamax(battle), terastallize=sim._should_terastallize(battle))
                            next_action = self.create_order(move, dynamax=dynamax, terastallize=tera)
                    if next_action is None and dont_verify:
                        # unseen move so just check if it is in the action prompt
                        if llm_move_id.lower().replace(' ', '') in state_action_prompt:
                            next_action = self.create_order(Move(llm_move_id.lower().replace(' ', ''), self.gen.gen), dynamax=dynamax, terastallize=tera)
                elif "switch" in llm_action_json.keys():
                    llm_switch_species = llm_action_json["switch"].strip()
                    switch_list = battle.available_switches
                    if dont_verify: # opponent prediction
                        observable_switches = []
                        for _, opponent_pokemon in battle.opponent_team.items():
                            if not opponent_pokemon.active:
                                observable_switches.append(opponent_pokemon)
                        switch_list = observable_switches
                    for i, pokemon in enumerate(switch_list):
                        if pokemon.species.lower().replace(' ', '') == llm_switch_species.lower().replace(' ', ''):
                            next_action = self.create_order(pokemon)
                else:
                    raise ValueError('No valid action')
                
                # with open(f"{self.log_dir}/output.jsonl", "a") as f:
                #     f.write(json.dumps({"turn": battle.turn,
                #                         "system_prompt": system_prompt,
                #                         "user_prompt": state_prompt_io,
                #                         "llm_output": llm_output,
                #                         "battle_tag": battle.battle_tag
                #                         }) + "\n")
                
                if next_action is not None:
                    break
            except Exception as e:
                print(f'Exception: {e}', 'passed')
                continue
        if next_action is None:
            try:
                print('No action found', llm_action_json, actions, dont_verify)
            except:
                pass

            # raise ValueError('No valid move', battle.active_pokemon.fainted, len(battle.available_switches))
            next_action = self.choose_max_damage_move(battle)
        return next_action

    def log_rationale(
            self,
            battle: Battle,
            system_prompt: str,
            user_prompt: str,
            llm_action_json: dict,
            player_name: str | None = None,  # ← optional override
    ) -> None:

        try:
            # 1) figure out names
            if player_name is None:
                player_name = getattr(
                    self.ps_client.account_configuration, "username", "player"
                )
            battle_id = getattr(battle, "battle_tag", "unknown").replace(":", "_")

            # 2) split out rationale vs action
            rationale_text = llm_action_json.get("thought")
            response_json = llm_action_json.copy()
            response_json.pop("thought", None)

            # 3) build the log entry
            log_entry = {
                "battle_id": battle.battle_tag,
                "round": battle.turn,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "rationale": rationale_text,
                "response": response_json,
            }

            # 4) ensure directory and file path
            rationale_dir = os.path.join("battle_log", "rationale")
            os.makedirs(rationale_dir, exist_ok=True)
            file_name = f"{battle_id}_{player_name}_rationales.jsonl"
            out_path = os.path.join(rationale_dir, file_name)

            # 5) write one JSONL line
            with open(out_path, "a", encoding="utf8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        except Exception:
            # never break gameplay if logging fails
            pass

    def sc(self, retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim):
        actions = [self.io(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim) for i in range(self.K)]
        action_message = [action.message for action in actions]
        _, counts = np.unique(action_message, return_counts=True)
        index = np.argmax(counts)
        return actions[index]
    
    def estimate_matchup(self, sim: LocalSim, battle: Battle, mon: Pokemon, mon_opp: Pokemon, is_opp: bool=False) -> Tuple[Move, int]:
        hp_remaining = []
        moves = list(mon.moves.keys())
        if is_opp:
            moves = sim.get_opponent_current_moves(mon=mon)
        if battle.active_pokemon.species == mon.species and not is_opp:
            moves = [move.id for move in battle.available_moves]
        for move_id in moves:
            move = Move(move_id, gen=sim.gen.gen)
            t = np.inf
            if move.category == MoveCategory.STATUS:
                # apply stat boosting effects to see if it will KO in fewer turns
                t = get_status_num_turns_fnt(mon, move, mon_opp, sim, boosts=mon._boosts.copy())
            else:
                t = get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())
            hp_remaining.append(t)
            # _, hp2, _, _ = sim.calculate_remaining_hp(battle.active_pokemon, battle.opponent_active_pokemon, move, None)
            # hp_remaining.append(hp2)
        hp_best_index = np.argmin(hp_remaining)
        best_move = moves[hp_best_index]
        best_move_turns = hp_remaining[hp_best_index]
        best_move = Move(best_move, gen=sim.gen.gen)
        best_move = self.create_order(best_move)
        # check special moves: tera/dyna
        # dyna for gen 8
        if sim.battle._data.gen == 8 and sim.battle.can_dynamax:
            for move_id in moves:
                move = Move(move_id, gen=sim.gen.gen).dynamaxed
                if move.category != MoveCategory.STATUS:
                    t = get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())
                    if t < best_move_turns:
                        best_move = self.create_order(move, dynamax=True)
                        best_move_turns = t
        # tera for gen 9
        elif sim.battle._data.gen == 9 and sim.battle.can_tera:
            mon.terastallize()
            for move_id in moves:
                move = Move(move_id, gen=sim.gen.gen)
                if move.category != MoveCategory.STATUS:
                    t = get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())
                    if t < best_move_turns:
                        best_move = self.create_order(move, terastallize=True)
                        best_move_turns = t
            mon.unterastallize()
            
        return best_move, best_move_turns

    def dmg_calc_move(self, battle: AbstractBattle, return_move: bool=False):
        sim = LocalSim(battle, 
                    self.move_effect,
                    self.pokemon_move_dict,
                    self.ability_effect,
                    self.pokemon_ability_dict,
                    self.item_effect,
                    self.pokemon_item_dict,
                    self.gen,
                    self._dynamax_disable,
                    format=self.format
        )
        best_action = None
        best_action_turns = np.inf
        if battle.available_moves and not battle.active_pokemon.fainted:
            # try moves and find hp remaining for opponent
            mon = battle.active_pokemon
            mon_opp = battle.opponent_active_pokemon
            best_action, best_action_turns = self.estimate_matchup(sim, battle, mon, mon_opp)
        if return_move:
            if best_action is None:
                return None, best_action_turns
            return best_action.order, best_action_turns
        if best_action_turns > 4:
            return None, np.inf
        if best_action is not None:
            return best_action, best_action_turns
        return self.choose_random_move(battle), 1
    
    
    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4

    def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon):
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None]
        )
        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            score += self.SPEED_TIER_COEFICIENT
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            score -= self.SPEED_TIER_COEFICIENT

        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT

        return score
    
    def check_timeout(self, start_time, battle):
        if time.time() - start_time > 30:
            print('default due to time')
            move, _ = self.dmg_calc_move(battle)
            return move
        else:
            return None

    def tree_search(self, retries, battle, sim=None, return_opp=False, use_pred_engine="False") -> BattleOrder:
        # generate local simulation
        root = SimNode(battle,
                       self.move_effect,
                       self.pokemon_move_dict,
                       self.ability_effect,
                       self.pokemon_ability_dict,
                       self.item_effect,
                       self.pokemon_item_dict,
                       self.gen,
                       self._dynamax_disable,
                       depth=1,
                       format=self.format,
                       prompt_translate=self.prompt_translate,
                       sim=sim
                       )
        q = [
            root
        ]
        leaf_nodes = []
        # create node and add to q B times
        start_time = time.time()
        while len(q) != 0:
            node = q.pop(0)
            # choose node for expansion
            # generate B actions
            player_actions = []
            system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, action_prompt_switch, action_prompt_move = node.simulation.get_player_prompt(
                return_actions=True)
            # panic_move = self.check_timeout(start_time, battle)
            # if panic_move is not None:
            #     return panic_move
            # end if terminal
            if node.simulation.is_terminal() or node.depth == self.K:
                try:
                        # value estimation for leaf nodes
                    # value_prompt = 'Evaluate the score from 1-100 based on how likely the player is to win. Higher is better. Start at 50 points.' + \
                    #                'Add points based on the effectiveness of current available moves.' + \
                    #                'Award points for each pokemon remaining on the player\'s team, weighted by their strength.' + \
                    #                'Add points for boosted status and opponent entry hazards and subtract points for status effects and player entry hazards. ' + \
                    #                'Subtract points for excessive switching.' + \
                    #                'Subtract points based on the effectiveness of the opponent\'s current moves, especially if they have a faster speed.' + \
                    #                'Remove points for each pokemon remaining on the opponent\'s team, weighted by their strength.\n'
                    # cot_prompt = (
                    #     "Think step-by-step (≤ 7 short sentences in total). "
                    #     "list every pivotal ability, move, "
                    #     "or type interaction that influences the score, each with a brief "
                    #     "description (e.g. *\"Quark-Drive boosts Speed → outspeeds Primarina\"*). "
                    #     'After the explanation, output **one** JSON object exactly in this form:\n'
                    #     '{"thought":"<your brief justification>", "score": <total_points>}\n'
                    # )
                    value_prompt = (
                            "Evaluate the score from 1–100 (integer) based on how likely the player is to win. Start at 50 points. "
                            "Add points for effective current moves; award points for stronger/well-positioned remaining Pokémon; "
                            "add for our boosts and opponent hazards; subtract for our negative status and our hazards; "
                            "subtract for excessive switching; subtract for opponent move effectiveness, especially if they outspeed; "
                            "subtract for the opponent’s remaining team strength. "
                            "Use the fixed labels: EffectiveMoves, TypeMatchup, SpeedControl, TeamAdvantage, BoostsAndStatus, "
                            "HazardsAndField, SwitchingCost, OpponentPressure, Uncertainty, WinPlan. "
                            "Constraints: each label_scores value must be an integer in [-20,20]; the sum of label_scores must be in [-49,50]; "
                            "compute score = 50 + sum(label_scores) and then CLAMP to [1,100]. "
                            "Never output a score > 100 (reduce positive label_scores if needed so the final clamped score ≤ 100).\n"
                    )
                    cot_prompt = (
                        "Think step-by-step. "
                        "List every pivotal ability, move, or type interaction that influences the score, each with a brief description "
                        "(e.g. *\"Quark-Drive boosts Speed → outspeeds Primarina\"*). "
                        "Use the fixed labels exactly; if a label is irrelevant, set its text to 'n/a' and its score to 0. "
                        "After the explanation, output **one** JSON object exactly in this form:\n"
                        "{"
                        "\"thought\": {"
                        "\"EffectiveMoves\": \"<≤1 sentence or 'n/a'>\", "
                        "\"TypeMatchup\": \"<≤1 sentence or 'n/a'>\", "
                        "\"SpeedControl\": \"<≤1 sentence or 'n/a'>\", "
                        "\"TeamAdvantage\": \"<≤1 sentence or 'n/a'>\", "
                        "\"BoostsAndStatus\": \"<≤1 sentence or 'n/a'>\", "
                        "\"HazardsAndField\": \"<≤1 sentence or 'n/a'>\", "
                        "\"SwitchingCost\": \"<≤1 sentence or 'n/a'>\", "
                        "\"OpponentPressure\": \"<≤1 sentence or 'n/a'>\", "
                        "\"Uncertainty\": \"<≤1 sentence or 'n/a'>\", "
                        "\"WinPlan\": \"<≤1 sentence or 'n/a'>\""
                        "}, "
                        "\"label_scores\": {"
                        "\"EffectiveMoves\": <int>, "
                        "\"TypeMatchup\": <int>, "
                        "\"SpeedControl\": <int>, "
                        "\"TeamAdvantage\": <int>, "
                        "\"BoostsAndStatus\": <int>, "
                        "\"HazardsAndField\": <int>, "
                        "\"SwitchingCost\": <int>, "
                        "\"OpponentPressure\": <int>, "
                        "\"Uncertainty\": <int>, "
                        "\"WinPlan\": <int>"
                        "}, "
                        "\"score\": <int>"
                        "}\n"
                    )
                    state_prompt_io = state_prompt + value_prompt + cot_prompt

                    if use_pred_engine != "False":
                        player_action = "default"
                        # Check if the player's action is a move
                        if isinstance(node.action.order, Move):
                            player_action = f"move: {node.action.order.id}"
                        # Check if the player's action is a switch
                        elif isinstance(node.action.order, Pokemon):
                            player_action = f"switch: {node.action.order.species}"

                        opponent_action = "default"
                        # Check if the opponent's action is a move
                        if isinstance(node.action_opp.order, Move):
                            opponent_action = f"move: {node.action_opp.order.id}"
                        # Check if the opponent's action is a switch
                        elif isinstance(node.action_opp.order, Pokemon):
                            opponent_action = f"switch: {node.action_opp.order.species}"
                        try:
                            game_state = state_prompt.split("Current battle state:\n", 1)[1]
                        except Exception:
                            print("Got exception during the game_state generation")
                            game_state = state_prompt if isinstance(state_prompt, str) else ""
                        score = self.prediction_engine.predict(game_state,player_action,opponent_action)
                        node.hp_diff = int(score)
                        node.rationale = "Generated From Prediction Engine"
                        node.label_scores = {}

                    else:
                        llm_output = self.get_LLM_action(system_prompt=system_prompt,
                                                         user_prompt=state_prompt_io,
                                                         model=self.backend,
                                                         temperature=self.temperature,
                                                         max_tokens=500,
                                                         json_format=True,
                                                         llm=self.llm_value,
                                                         battle=battle,
                                                         )
                        # load when llm does heavy lifting for parsing
                        llm_action_json = json.loads(llm_output)
                        node.hp_diff = int(llm_action_json['score'])
                        node.rationale = llm_action_json.get("thought")
                        node.label_scores = llm_action_json.get("label_scores")

                except Exception as e:
                    node.hp_diff = node.simulation.get_hp_diff()
                    print(e)

                leaf_nodes.append(node)
                continue
            # panic_move = self.check_timeout(start_time, battle)
            # if panic_move is not None:
            #     return panic_move
            # estimate opp
            try:
                opp_move, opp_turns = self.estimate_matchup(
                    node.simulation,
                    node.simulation.battle,
                    node.simulation.battle.opponent_active_pokemon,
                    node.simulation.battle.active_pokemon,
                    is_opp=True,
                )
            except Exception:
                opp_move, opp_turns = None, np.inf
            if isinstance(opp_move, Move):
                action_opp = BattleOrder.move_to_order(opp_move, target=0)
            else:
                action_opp = None
            node.action_opp = action_opp
            ##############################
            # generate players' action  #
            ##############################
            # if not node.simulation.battle.active_pokemon.fainted and len(battle.available_moves) > 0:  # Original: referenced root battle state
            if not node.simulation.battle.active_pokemon.fainted and len(node.simulation.battle.available_moves) > 0:
                # get dmg calc move
                dmg_calc_out, dmg_calc_turns = self.dmg_calc_move(node.simulation.battle)
                if dmg_calc_out is not None:
                    if dmg_calc_turns <= opp_turns:
                        try:
                            # ask LLM to use heuristic tool or minimax search
                            tool_prompt = '''Based on the current battle state, evaluate whether to use the damage calculator tool or the minimax tree search method. Consider the following factors:

                                1. Damage calculator advantages:
                                - Quick and efficient for finding optimal damaging moves
                                - Useful when a clear type advantage or high-power move is available
                                - Effective when the opponent's is not switching and current pokemon is likely to KO opponent

                                2. Minimax tree search advantages:
                                - Can model opponent behavior and predict future moves
                                - Useful in complex situations with multiple viable options
                                - Effective when long-term strategy is crucial

                                3. Current battle state:
                                - Remaining Pokémon on each side
                                - Health of active Pokémon
                                - Type matchups
                                - Available moves and their effects
                                - Presence of status conditions or field effects

                                4. Uncertainty level:
                                - How predictable is the opponent's next move?
                                - Are there multiple equally viable options for your next move?

                                Evaluate these factors and decide which method would be more beneficial in the current situation. Output your choice in the following JSON format:
                                First **think step by step** about these factors,  list every pivotal ability, move, or type interaction that influences the score, each with a brief description (e.g. *\"Quark-Drive boosts Speed → outspeeds Primarina\"*). 
                                After the explanation, output **one** JSON object exactly in this form:\n'

                                {"thought":"<your short justification (≤ 4 sentences)>",
                                 "choice":"damage calculator" or "choice":"minimax"} '''

                            state_prompt_io = state_prompt + tool_prompt
                            llm_output = self.get_LLM_action(system_prompt=system_prompt,
                                                             user_prompt=state_prompt_io,
                                                             model=self.backend,
                                                             temperature=0.6,
                                                             max_tokens=100,
                                                             json_format=True,
                                                             battle=battle
                                                             )
                            # load when llm does heavy lifting for parsing
                            try:
                                llm_action_json = json.loads(llm_output)
                            except:
                                print("LLM OUTPUTS JSON PARSED ERROR:", llm_output)
                                llm_action_json = {}
                            rationale = llm_action_json.get("thought", "").strip()
                            if not rationale:
                                rationale = (
                                    f"Heuristic TTK shortcut: my_turns={dmg_calc_turns}, "
                                    f"opp_turns={opp_turns}"
                                )
                            if 'choice' in llm_action_json.keys():
                                if llm_action_json['choice'] != 'minimax':
                                    heuristic_score = -1
                                    # --- FIX: always return 5 values (add summary_list) ---
                                    if return_opp:
                                        summary_list = [{
                                            "player_action": str(getattr(dmg_calc_out, "order", dmg_calc_out)),
                                            "score": heuristic_score,
                                            "rationale": rationale,
                                            "opponent_action": str(
                                                getattr(action_opp, "order", action_opp)) if action_opp else None,
                                            "label_scores": {}
                                        }]
                                        return dmg_calc_out, action_opp, heuristic_score, rationale, summary_list, {}
                                    return dmg_calc_out


                                    # ------------------------------------------------------
                        except:
                            print('defaulting to minimax')

                    player_actions.append(dmg_calc_out)
            # panic_move = self.check_timeout(start_time, battle)
            # if panic_move is not None:
            #     return panic_move
            # get llm switch

            # LLM Suggested Up to 2 switch Pokemons
            if len(node.simulation.battle.available_switches) != 0:  # or opp_turns < dmg_calc_turns):
                state_action_prompt_switch = state_action_prompt + action_prompt_switch + '\nYou can only choose to switch this turn.\n'
                constraint_prompt_io = 'Choose the best action and your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}.\n'
                for i in range(2):
                    action_llm_switch = self.io(retries, system_prompt, state_prompt, constraint_prompt_cot,
                                                constraint_prompt_io, state_action_prompt_switch,
                                                node.simulation.battle, node.simulation)
                    if len(player_actions) == 0:
                        player_actions.append(action_llm_switch)
                    elif action_llm_switch.message != player_actions[-1].message:
                        player_actions.append(action_llm_switch)
                        player_actions.append(action_llm_switch)

            # LLM Suggested Up to 1 Move
            if not node.simulation.battle.active_pokemon.fainted and len(
                    node.simulation.battle.available_moves) > 0:  # and not opp_turns < dmg_calc_turns:
                # get llm move
                state_action_prompt_move = state_action_prompt + action_prompt_move + '\nYou can only choose to move this turn.\n'
                constraint_prompt_io = 'Choose the best action and your output MUST be a JSON like: {"move":"<move_name>"}.\n'
                action_llm_move = self.io(retries, system_prompt, state_prompt, constraint_prompt_cot,
                                          constraint_prompt_io, state_action_prompt_move, node.simulation.battle,
                                          node.simulation)
                if len(player_actions) == 0:
                    player_actions.append(action_llm_move)
                elif action_llm_move.message != player_actions[0].message:
                    player_actions.append(action_llm_move)
            # panic_move = self.check_timeout(start_time, battle)
            # if panic_move is not None:
            #     return panic_move
            ##############################
            # generate opponent's action #
            ##############################
            opponent_actions = []
            tool_is_optimal = False
            # dmg calc suggestion
            # action_opp, opp_turns = self.estimate_matchup(node.simulation, node.simulation.battle, node.simulation.battle.opponent_active_pokemon, node.simulation.battle.active_pokemon, is_opp=True)
            if action_opp is not None:
                tool_is_optimal = True
                opponent_actions.append(self.create_order(action_opp))
            # heuristic matchup switch action
            best_score = np.inf
            best_action = None
            for mon in node.simulation.battle.opponent_team.values():
                if mon.species == node.simulation.battle.opponent_active_pokemon.species:
                    continue
                score = self._estimate_matchup(mon, node.simulation.battle.active_pokemon)
                if score < best_score:
                    best_score = score
                    best_action = mon
            if best_action is not None:
                opponent_actions.append(self.create_order(best_action))
            # panic_move = self.check_timeout(start_time, battle)
            # if panic_move is not None:
            #     return panic_move
            # create opponent prompt from battle sim
            system_prompt_o, state_prompt_o, constraint_prompt_cot_o, constraint_prompt_io_o, state_action_prompt_o = node.simulation.get_opponent_prompt(
                system_prompt)
            action_o = self.io(2, system_prompt_o, state_prompt_o, constraint_prompt_cot_o, constraint_prompt_io_o,
                               state_action_prompt_o, node.simulation.battle, node.simulation, dont_verify=True)
            is_repeat_action_o = np.array(
                [action_o.message == opponent_action.message for opponent_action in opponent_actions]).any()
            if not is_repeat_action_o:
                opponent_actions.append(action_o)
            # panic_move = self.check_timeout(start_time, battle)
            # if panic_move is not None:
            #     return panic_move
            # simulate outcome
            if node.depth < self.K:
                for action_p in player_actions:
                    for action_o in opponent_actions:
                        node_new = copy(node)
                        node_new.simulation.battle = copy(node.simulation.battle)
                        # if not tool_is_optimal:
                        node_new.children = []
                        node_new.depth = node.depth + 1
                        node_new.action = action_p
                        node_new.action_opp = action_o
                        node_new.parent_node = node
                        node_new.parent_action = node.action
                        node.children.append(node_new)
                        node_new.simulation.step(action_p, action_o)
                        q.append(node_new)

        # choose best action according to max or min rule
        def get_tree_action(node: SimNode):
            if len(node.children) == 0:  # leaf node
                summary = {
                    "player_action": str(node.action.order) if node.action else None,
                    "score": node.hp_diff,
                    "rationale": node.rationale,
                    "opponent_action": str(node.action_opp.order) if node.action_opp else None,
                    "label_scores": node.label_scores if hasattr(node, 'label_scores') else {}
                }
                return node.action, node.hp_diff, node.action_opp, node.rationale, [summary], summary["label_scores"]
            score_dict, action_dict, opp_dict, rationale_dict, label_scores_dict = {}, {}, {}, {}, {}
            for child in node.children:
                action_str = str(child.action.order)  # player's action as string key
                # Recursively get action, score, opp action, rationale for child node
                # --- FIX: unpack 5 values and ignore child's summary list ---
                action_obj, score_val, opp_act, rationale_text, _child_summary, child_label_scores = get_tree_action(child)
                # -----------------------------------------------------------
                if action_str in score_dict:
                    # Minimax: opponent will minimize the score for this action
                    if score_val < score_dict[action_str]:
                        score_dict[action_str] = score_val
                        opp_dict[action_str] = opp_act
                        rationale_dict[action_str] = rationale_text
                        label_scores_dict[action_str] = child_label_scores
                else:
                    score_dict[action_str] = score_val
                    action_dict[action_str] = action_obj
                    opp_dict[action_str] = opp_act
                    rationale_dict[action_str] = rationale_text
                    label_scores_dict[action_str] = child_label_scores
            summary_list = []
            for action_str, score_val in score_dict.items():
                summary_list.append({
                    "player_action": action_str,
                    "score": score_val,
                    "rationale": rationale_dict[action_str],
                    "opponent_action": str(opp_dict[action_str].order) if opp_dict[action_str] else None,
                    "label_scores": label_scores_dict.get(action_str, {})
                })
            # Sort candidates by score
            summary_list.sort(key=lambda entry: entry["score"], reverse=True)
            # Choose the player action with the highest worst-case score
            best_action_str = max(score_dict, key=score_dict.get)
            best_player_action = action_dict[best_action_str]
            best_score = score_dict[best_action_str]
            best_opp_action = opp_dict[best_action_str]
            best_rationale = rationale_dict[best_action_str]
            best_label_scores = label_scores_dict.get(best_action_str, {})

            return best_player_action, best_score, best_opp_action, best_rationale, summary_list, best_label_scores

        best_action, best_score, best_opp_action, best_rationale, summary_list, best_label_scores = get_tree_action(
            root)

        if return_opp:
            return best_action, best_opp_action, best_score, best_rationale, summary_list, best_label_scores
        return best_action

    def battle_summary(self):

        beat_list = []
        remain_list = []
        win_list = []
        tag_list = []
        for tag, battle in self.battles.items():
            beat_score = 0
            for mon in battle.opponent_team.values():
                beat_score += (1-mon.current_hp_fraction)

            beat_list.append(beat_score)

            remain_score = 0
            for mon in battle.team.values():
                remain_score += mon.current_hp_fraction

            remain_list.append(remain_score)
            if battle.won:
                win_list.append(1)

            tag_list.append(tag)

        return beat_list, remain_list, win_list, tag_list

    def reward_computing_helper(
        self,
        battle: AbstractBattle,
        *,
        fainted_value: float = 0.0,
        hp_value: float = 0.0,
        number_of_pokemons: int = 6,
        starting_value: float = 0.0,
        status_value: float = 0.0,
        victory_value: float = 1.0,
    ) -> float:
        """A helper function to compute rewards."""

        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value

        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value

        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value

        to_return = current_value - self._reward_buffer[battle] # the return value is the delta
        self._reward_buffer[battle] = current_value

        return to_return

    def choose_max_damage_move(self, battle: Battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        return self.choose_random_move(battle)

import json
import os


def count_actions_in_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Assuming structure: {"Battle Id: XXX": {"Turn 1": {...}, ...}}
        total_actions = 0
        for battle_key, battle_data in data.items():
            for turn_key, turn_data in battle_data.items():
                if "Turn " in turn_key:
                    # Filter and count candidates
                    candidates = turn_data.get("Candidates", [])
                    filtered_cands = [
                        cand for cand in candidates
                        if cand.get("score", 0) > -1 and cand.get("label_scores", {}) != {}
                    ]
                    num_filtered = len(filtered_cands)
                    if num_filtered != 1:
                        total_actions += num_filtered

                    # Do not count best separately to avoid duplication, as it is selected from candidates

        return total_actions
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return 0


def count_all_actions_in_folder(folder_path):
    total_actions = 0
    for filename in os.listdir(folder_path):
        if filename.startswith("score_evaluation_Battle_Id__") and filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            actions_in_file = count_actions_in_file(file_path)
            total_actions += actions_in_file
            print(f"{filename}: {actions_in_file} actions")

    print(f"\nTotal actions across all files: {total_actions}")


if __name__ == "__main__":
    count_all_actions_in_folder(".")
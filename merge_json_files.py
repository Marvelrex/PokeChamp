import json

def load_json_safe(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON decode error in {file_path}: {e}")
        return {}
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

def merge_two_json_files(file1, file2, output_file="merged_output.json"):
    data1 = load_json_safe(file1)
    data2 = load_json_safe(file2)

    merged = {}

    # Merge both dictionaries
    for battle_id, battle_data in data1.items():
        merged[battle_id] = battle_data

    for battle_id, battle_data in data2.items():
        if battle_id in merged:
            merged[battle_id].update(battle_data)
        else:
            merged[battle_id] = battle_data

    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged JSON saved to {output_file}")

# Example usage
merge_two_json_files("july_merged_battles.json", "score_evaluation_merged.json")

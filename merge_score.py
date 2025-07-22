import os
import json

# Set the directory containing your JSON files
input_dir = "./battle_log/one_vs_one/score_evaluation/best_move"
output_file = "merged_filtered_battle_data.json"

merged_data = {}

for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        with open(os.path.join(input_dir, filename), "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping malformed file: {filename}")
                continue

            for battle_id, turns in data.items():
                if battle_id not in merged_data:
                    merged_data[battle_id] = {}

                for turn_key, turn_data in turns.items():
                    candidates = turn_data.get("Candidates", [])
                    best = turn_data.get("Best", {})

                    if len(candidates) <= 1:
                        continue  # filter: only 0 or 1 candidate actions
                    if best.get("score", 0) < 1:
                        continue  # filter: best score < 1

                    merged_data[battle_id][turn_key] = turn_data

# Save merged + filtered data
with open(output_file, "w") as out_f:
    json.dump(merged_data, out_f, indent=4)

print(f"✅ Merged and saved to {output_file}")

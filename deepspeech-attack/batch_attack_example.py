from batch_attack import run_batch_ensemble_attacks
import os
import json

# Example usage: Specify your input, targets, models, and ensembles
input_wavs = [os.path.join('processed_sound', f) for f in os.listdir('processed_sound')]

print(input_wavs)

# List of target sentences
import csv

# Read the 10 sentences from the target_sentences.csv file
all_target_sentences = []
with open("../target_sentences.csv", "r", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        if row:
            s = row[0].strip()
            all_target_sentences.append(s)

print(all_target_sentences)


training_sets = ["an4", "librispeech", "ted"]
versions = ["v1", "v2"]

attack_params = {
    "epsilon": 0.03,
    "alpha": 0.001,
    "PGD_iter": 1000,
    "n_queries": 250
}

device = "cuda"
base_dir = "batch_attack_results"
os.makedirs(base_dir, exist_ok=True)

# Loop over all combinations
for input_wav in input_wavs:
    for target_sentence in all_target_sentences:
        for target_training_set in training_sets:
            for target_version in versions:
                target_model_config = {"training_set": target_training_set, "version": target_version}
                ensemble_model_configs = [
                    {"training_set": ts, "version": v}
                    for ts in training_sets
                    for v in versions
                    if not (ts == target_training_set and v == target_version) and v != target_version
                ]

                uid = f"{target_training_set}-{target_version}-{target_sentence}-{input_wav.split('/')[-1]}"
                print(f"Running attack on {uid}")

                run_batch_ensemble_attacks(
                    input_wav=input_wav,
                    target_sentences=[target_sentence],
                    target_model_configs=[target_model_config],
                    ensemble_model_configs=[ensemble_model_configs],
                    attack_params=attack_params,
                    device=device,
                    output_csv=os.path.join(base_dir, f"{uid}.csv"),
                    adv_output_dir=os.path.join(base_dir, uid)
                )
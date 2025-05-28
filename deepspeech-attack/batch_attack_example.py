from batch_attack import run_batch_ensemble_attacks
import os
import json

# Example usage: Specify your input, targets, models, and ensembles
input_wav = "processed_sound/normal0.wav"

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


# List of possible target model configs
all_target_model_configs = [
    {"training_set": "librispeech", "version": "v2"},
    {"training_set": "ted", "version": "v2"},
    {"training_set": "an4", "version": "v2"}
]

# List of ensemble sets (each is a list of configs)
all_ensemble_model_configs = [
    [
        {"training_set": "librispeech", "version": "v1"},
        {"training_set": "librispeech", "version": "v2"},
        {"training_set": "ted", "version": "v2"},
        {"training_set": "an4", "version": "v1"},
        {"training_set": "an4", "version": "v2"}
    ],
    [
        {"training_set": "librispeech", "version": "v2"},
        {"training_set": "ted", "version": "v2"}
    ]
    # Add more ensemble sets as needed
]

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
for ens_idx, ensemble_model_configs in enumerate(all_ensemble_model_configs):
    for tgt_idx, target_model_config in enumerate(all_target_model_configs):
        for sent_idx, target_sentence in enumerate(all_target_sentences):
            # Use unique output file per combo
            ens_str = "_".join([f"{cfg['training_set']}-{cfg['version']}" for cfg in ensemble_model_configs])
            tgt_str = f"{target_model_config['training_set']}-{target_model_config['version']}"
            sent_str = target_sentence.replace(" ", "_").replace("/", "-")  # Truncate for filename safety
            output_dir = os.path.join(base_dir, ens_str, tgt_str, sent_str)
            os.makedirs(output_dir, exist_ok=True)
            output_csv = os.path.join(
                output_dir,
                f"results_tgt-{tgt_str}_ens-{ens_str}_sent-{sent_str}.csv"
            )
            print(f"Running attack: Target={tgt_str}, Ensemble={ens_str}, Sentence=\"{target_sentence}\"")
            run_batch_ensemble_attacks(
                input_wav=input_wav,
                target_sentences=[target_sentence],
                target_model_configs=[target_model_config],
                ensemble_model_configs=[ensemble_model_configs],
                attack_params=attack_params,
                device=device,
                output_csv=output_csv,
                adv_output_dir=output_dir
            )

from batch_attack import run_batch_ensemble_attacks
import os
import json

# Example usage: Specify your input, targets, models, and ensembles
input_wav = "processed_sound/normal0.wav"

# List of target sentences
all_target_sentences = [
    "HELLO WORLD",
    "TEST PHRASE",
    "HENRY",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "1234567890",
    "!@#$%^&*()_+",
    "QWERTYUIOPASDFGHJKLZXCVBNM",
    "abcdefghijklmnopqrstuvwxyz",
    "the quick brown fox jumps over the lazy dog",
    ""
]

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
output_dir = "batch_attack_results"
os.makedirs(output_dir, exist_ok=True)

# Loop over all combinations
for ens_idx, ensemble_model_configs in enumerate(all_ensemble_model_configs):
    for tgt_idx, target_model_config in enumerate(all_target_model_configs):
        for sent_idx, target_sentence in enumerate(all_target_sentences):
            # Use unique output file per combo
            ens_str = "_".join([f"{cfg['training_set']}-{cfg['version']}" for cfg in ensemble_model_configs])
            tgt_str = f"{target_model_config['training_set']}-{target_model_config['version']}"
            sent_str = target_sentence[:20].replace(" ", "_").replace("/", "-")  # Truncate for filename safety
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

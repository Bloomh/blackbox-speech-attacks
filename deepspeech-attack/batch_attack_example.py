from batch_attack import run_batch_ensemble_attacks

# Example usage
input_wav = "processed_sound/normal0.wav"
target_sentences = ["HELLO WORLD", "TEST PHRASE"]
target_model_configs = [
    {"training_set": "librispeech", "version": "v2"},
    {"training_set": "ted", "version": "v2"}
]
ensemble_model_configs = [
    [
        {"training_set": "librispeech", "version": "v1"},
        {"training_set": "librispeech", "version": "v2"},
        {"training_set": "ted", "version": "v2"},
        {"training_set": "an4", "version": "v1"},
        {"training_set": "an4", "version": "v2"}
    ]
]
attack_params = {
    "epsilon": 0.03,
    "alpha": 0.001,
    "PGD_iter": 2,
    "n_queries": 25
}
output_csv = "batch_attack_results.csv"

# Run the batch attack
run_batch_ensemble_attacks(
    input_wav=input_wav,
    target_sentences=target_sentences,
    target_model_configs=target_model_configs,
    ensemble_model_configs=ensemble_model_configs,
    attack_params=attack_params,
    device="cpu",
    output_csv=output_csv
)

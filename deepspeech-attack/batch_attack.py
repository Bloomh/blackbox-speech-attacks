import os
import json
import pandas as pd
import torchaudio
from attack import Attacker
from black_box import load_surrogate_model

def run_batch_ensemble_attacks(
    input_wav,
    target_sentences,
    target_model_configs,
    ensemble_model_configs,
    attack_params,
    device="cuda",
    output_csv="batch_attack_results.csv",
    adv_output_dir="batch_adv_outputs"
):
    """
    Run ensemble adversarial attacks over all combinations of target models, ensemble models, and target sentences.
    Save results to a pandas DataFrame and CSV.
    """
    # Load audio once
    sound, sample_rate = torchaudio.load(input_wav)
    results = []

    # Pre-load all models/decoders to avoid redundant loads
    model_cache = {}
    def get_model(training_set, version):
        key = (training_set, version)
        if key not in model_cache:
            model, decoder, labels = load_surrogate_model(training_set, version, device)
            model_cache[key] = (model, decoder, labels)
        return model_cache[key]

    # Ensure output directory exists
    os.makedirs(adv_output_dir, exist_ok=True)

    for tgt_cfg in target_model_configs:
        tgt_train = tgt_cfg['training_set']
        tgt_ver = tgt_cfg['version']
        tgt_model, tgt_decoder, tgt_labels = get_model(tgt_train, tgt_ver)

        # Build ensemble configs, always exclude the target model
        for ens_cfg_list in ensemble_model_configs:
            filtered_ens = [cfg for cfg in ens_cfg_list if not (cfg['training_set'] == tgt_train and cfg['version'] == tgt_ver)]
            if not filtered_ens:
                continue
            ens_models = []
            ens_decoders = []
            ens_versions = []
            ens_trainsets = []
            for cfg in filtered_ens:
                model, decoder, _ = get_model(cfg['training_set'], cfg['version'])
                ens_models.append(model)
                ens_decoders.append(decoder)
                ens_versions.append(cfg['version'])
                ens_trainsets.append(cfg['training_set'])
            ensemble_model_info = list(zip(ens_models, ens_decoders, ens_versions))

            for target_sentence in target_sentences:
                # Compose a unique adversarial output filename
                ens_str = "_".join([f"{cfg['training_set']}-{cfg['version']}" for cfg in filtered_ens])
                safe_target = target_sentence.replace(" ", "_").replace("/", "-")
                adv_wav_name = f"adv_{tgt_train}-{tgt_ver}__ens_{ens_str}__tgt_{safe_target}.wav"
                adv_wav_path = os.path.join(adv_output_dir, adv_wav_name)

                attacker = Attacker(
                    target_model=tgt_model,
                    surrogate_model=ens_models[0],  # Not used in ENSEMBLE mode
                    sound=sound,
                    target=target_sentence,
                    target_decoder=tgt_decoder,
                    sample_rate=sample_rate,
                    device=device,
                    surrogate_decoder=ens_decoders[0],
                    surrogate_version=ens_versions[0],
                    target_version=tgt_ver,
                    target_training_set=tgt_train,
                    ensemble_versions=ens_versions,
                    ensemble_training_sets=ens_trainsets,
                    ensemble_model_info=ensemble_model_info,
                    ensemble_weights=None,
                    save=adv_wav_path,
                    adv_output_dir=adv_output_dir
                )
                # Run attack
                out = attacker.attack(
                    epsilon=attack_params.get('epsilon', 0.3),
                    alpha=attack_params.get('alpha', 0.001),
                    attack_type="ENSEMBLE",
                    PGD_round=attack_params.get('PGD_iter', 40),
                    n_queries=attack_params.get('n_queries', 25)
                )
                # Unpack outputs (see Attacker.attack)
                db_difference, l_distance, target_string, final_output_target, target_distances, surrogate_distances, ensemble_lev_dists = out
                # Compose result row
                row = {
                    'adv_wav_path': adv_wav_path,
                    'input_wav': input_wav,
                    'target_sentence': target_sentence,
                    'target_model': tgt_train,
                    'target_version': tgt_ver,
                    'ensemble_models': json.dumps(filtered_ens),
                    'attack_params': json.dumps(attack_params),
                    'target_pred': final_output_target,
                    'target_lev_dists': target_distances,
                    'ensemble_lev_dists': json.dumps(ensemble_lev_dists) if ensemble_lev_dists else None,
                    'max_db_diff': db_difference,
                }
                results.append(row)
    # Build DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved batch attack results to {output_csv}")
    return df

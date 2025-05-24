import sys, os
import argparse
import torchaudio
from attack import Attacker

def to_model_path(training_data, version):
    if version in ("v1", "v2"):
        return f"../models/{training_data}/pretrained_{version}.pth"
    else:
        raise ValueError("Unknown DeepSpeech version")

def load_surrogate_model(training_data, version, device):
    if version == "v1":
        sys.path.insert(0, "../deepspeech.pytorch.v1")
        from model import DeepSpeech as DeepSpeechV1
        from decoder import GreedyDecoder
        model_path = to_model_path(training_data, version)
        model = DeepSpeechV1.load_model(model_path)
        model = model.to(device)
        # v1 labels: check model._labels if available, else use default
        labels = getattr(model, '_labels', " abcdefghijklmnopqrstuvwxyz'")
        decoder = GreedyDecoder(labels)
    elif version == "v2":
        sys.path.insert(0, "../deepspeech.pytorch.v2")
        from deepspeech_pytorch.utils import load_model, load_decoder
        from deepspeech_pytorch.configs.inference_config import TranscribeConfig
        model_path = to_model_path(training_data, version)
        model = load_model(device=device, model_path=model_path, use_half=False)
        cfg = TranscribeConfig
        decoder = load_decoder(labels=model.labels, cfg=cfg.lm)
        labels = model.labels
    else:
        raise ValueError("Unknown DeepSpeech version")
    return model, decoder, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # I/O parameters
    parser.add_argument('--input_wav', type=str, help='input wav. file')
    parser.add_argument('--output_wav', type=str, default='None', help='output adversarial wav. file')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--surrogate_version', type=str, default='v1', choices=['v1', 'v2'], help='Which DeepSpeech version to use as the surrogate (v1 or v2)')
    parser.add_argument('--target_version', type=str, default='v2', choices=['v1', 'v2'], help='Which DeepSpeech version to use as the target/main model (v1 or v2)')
    # attack parameters
    parser.add_argument('--target_sentence', type=str, default="HELLO WORLD", help='Please use uppercase')
    parser.add_argument('--mode', type=str, default="PGD", help='PGD or FGSM or [new -->] NES_GREY or NES_BLACK or ENSEMBLE') # we'll manually have to edit the multi-model inputs below for ensemble attacks
    parser.add_argument('--epsilon', type=float, default=0.25, help='epsilon')
    parser.add_argument('--alpha', type=float, default=1e-3, help='alpha')
    parser.add_argument('--PGD_iter', type=int, default=50, help='PGD iteration times')
    parser.add_argument('--n_queries', type=int, default=25, help='Number of queries for NES attack')
    # plot parameters
    parser.add_argument('--plot_ori_spec', type=str, default="None", help='Path to save the original spectrogram')
    parser.add_argument('--plot_adv_spec', type=str, default="None", help='Path to save the adversarial spectrogram')
    args = parser.parse_args()

    # Load input audio
    sound, sample_rate = torchaudio.load(args.input_wav)
    target_sentence = args.target_sentence.upper()
    if args.output_wav == "None":
        args.output_wav = None

    # Load surrogate model (for attack)
    surrogate_model, surrogate_decoder, _ = load_surrogate_model("librispeech", args.surrogate_version, args.device)
    # Load target model (for evaluation)
    target_training_set = "librispeech"
    target_model, target_decoder, _ = load_surrogate_model(target_training_set, args.target_version, args.device)
    
    # ENSEMBLE INIT
    
    # if the attack mode has been specified as ENSEMBLE, manually modify ensemble here.
    ensemble_versions = ["v1", "v2", "v2", "v1"] # we might have more of each if we get pretrained models on diff datasets
    ensemble_training_sets = ["librispeech", "librispeech", "ted", "an4"]
    

    # Run attack
    attacker = Attacker(
        target_model=target_model,
        surrogate_model=surrogate_model,
        sound=sound,
        target=target_sentence,
        target_decoder=target_decoder,
        surrogate_decoder=surrogate_decoder,
        device=args.device,
        save=args.output_wav,
        surrogate_version=args.surrogate_version,
        target_version=args.target_version,
        target_training_set=target_training_set,
        ensemble_versions=ensemble_versions,
        ensemble_training_sets=ensemble_training_sets,
        ensemble_model_info=[load_surrogate_model(trainset, version, args.device) for trainset, version in zip(ensemble_training_sets, ensemble_versions)],
        ensemble_weights=None
    )
    db_difference, l_distance, target_string, final_output_target, target_distances, surrogate_distances = attacker.attack(epsilon=args.epsilon, alpha=args.alpha, attack_type=args.mode, PGD_round=args.PGD_iter, n_queries=args.n_queries)

    # Plot and save Levenshtein distances over time
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(target_distances, label='Target Model')
    plt.plot(surrogate_distances, label='Surrogate Model')
    plt.xlabel('Attack Step')
    plt.ylabel('Levenshtein Distance')
    plt.title('Levenshtein Distance Over Attack Steps')
    plt.legend()
    plt.tight_layout()
    plt.savefig('levenshtein_plot.png')
    plt.close()

    if args.plot_ori_spec != "None":
        attacker.get_ori_spec(args.plot_ori_spec)
    if args.plot_adv_spec != "None":
        attacker.get_adv_spec(args.plot_adv_spec)

import sys, os
import argparse
import torchaudio
from attack import Attacker

def version_to_model_path(version):
    if version in ("v1", "v2"):
        return f"../models/librispeech/librispeech_pretrained_{version}.pth"
    else:
        raise ValueError("Unknown DeepSpeech version")

def load_surrogate_model(version, device):
    if version == "v1":
        sys.path.insert(0, "../deepspeech.pytorch.v1")
        from model import DeepSpeech as DeepSpeechV1
        from decoder import GreedyDecoder
        model_path = version_to_model_path(version)
        model = DeepSpeechV1.load_model(model_path)
        model = model.to(device)
        # v1 labels: check model._labels if available, else use default
        labels = getattr(model, '_labels', " abcdefghijklmnopqrstuvwxyz'")
        decoder = GreedyDecoder(labels)
    elif version == "v2":
        sys.path.insert(0, "../deepspeech.pytorch.v2")
        from deepspeech_pytorch.utils import load_model, load_decoder
        from deepspeech_pytorch.configs.inference_config import TranscribeConfig
        model_path = version_to_model_path(version)
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
    parser.add_argument('--model_path', type=str, default='None', help='model pth path; please use absolute path')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--surrogate_version', type=str, default='v1', choices=['v1', 'v2'], help='Which DeepSpeech version to use as the surrogate (v1 or v2)')
    # attack parameters
    parser.add_argument('--target_sentence', type=str, default="HELLO WORLD", help='Please use uppercase')
    parser.add_argument('--mode', type=str, default="PGD", help='PGD or FGSM or [new -->] NES_GREY or NES_BLACK')
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
    model, decoder, labels = load_surrogate_model(args.surrogate_version, args.device)

    # Run attack
    attacker = Attacker(model=model, sound=sound, target=target_sentence, decoder=decoder, device=args.device, save=args.output_wav, model_version=args.surrogate_version)
    attacker.attack(epsilon=args.epsilon, alpha=args.alpha, attack_type=args.mode, PGD_round=args.PGD_iter, n_queries=args.n_queries)

    if args.plot_ori_spec != "None":
        attacker.get_ori_spec(args.plot_ori_spec)
    if args.plot_adv_spec != "None":
        attacker.get_adv_spec(args.plot_adv_spec)

    # TODO: use the model_path specified for evaluation of the final success

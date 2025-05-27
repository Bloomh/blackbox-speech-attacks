from stft import STFT, magphase
import torch.nn as nn
import torch
import Levenshtein
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import math
import hashlib

def target_sentence_to_label(sentence, labels="_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "):
    out = []
    for idx, word in enumerate(sentence):
        if word not in labels:
            print(f"[WARNING] Character '{word}' at position {idx} in sentence '{sentence}' not in labels: {labels}. Skipping.")
            continue
        out.append(labels.index(word))
    return torch.IntTensor(out)

def torch_spectrogram(sound, torch_stft):
    real, imag = torch_stft(sound)
    mag, cos, sin = magphase(real, imag)
    mag = torch.log1p(mag)
    mean = mag.mean()
    std = mag.std()
    mag = mag - mean
    mag = mag / std
    mag = mag.permute(0,1,3,2)
    return mag


# Robustly flatten and join any nested list output from decoder to produce a single string
def flatten_to_string(obj):
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, list):
        # Recursively flatten, join non-empty strings with spaces
        flat = []
        for item in obj:
            s = flatten_to_string(item)
            if s:
                flat.append(s)
        return ' '.join(flat)
    else:
        return str(obj)

def decode_model_output(model_version, decoder, out, output_sizes):
    """
    Decodes model output for either v1 or v2 DeepSpeech models.
    Returns the decoded string.
    """
    if model_version == "v1":
        # Transpose output to [T, N, H] as in test.py
        out = out.transpose(0, 1)  # TxNxH
        seq_length = out.size(0)
        batch_size = out.size(1)
        sizes = torch.IntTensor([seq_length] * batch_size)
        decoded_output = decoder.decode(out, sizes)
        return flatten_to_string(decoded_output[0][0])
    else:
        decoded_output, _ = decoder.decode(out, output_sizes)
        return flatten_to_string(decoded_output[0][0])

def run_model(model, model_version, spec, input_sizes):
    """
    Runs the correct forward pass for DeepSpeech v1 or v2 models.
    Returns (out, output_sizes), where output_sizes is None for v1.
    """
    if model_version == "v1":
        out = model(spec)
        output_sizes = None
        return out, output_sizes
    else:
        return model(spec, input_sizes)

def freeze_batchnorm_stats(model):
    """Freeze BatchNorm running statistics to prevent them from being updated during forward passes."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.track_running_stats = False

def unfreeze_batchnorm_stats(model):
    """Unfreeze BatchNorm running statistics."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.track_running_stats = True

class Attacker:
    def __init__(self, target_model, surrogate_model, sound, target, target_decoder, sample_rate=16000, device="cpu", save=None, surrogate_decoder=None, surrogate_version="v2", target_version="v2", target_training_set="librispeech",
                 ensemble_versions=None, ensemble_training_sets=None, ensemble_model_info=None, ensemble_weights=None):
        # print(f"[MODEL LOAD] Target model: training_set={target_training_set}, version={target_version}")
        # print(f"[MODEL LOAD] Surrogate model: version={surrogate_version}")
        if ensemble_training_sets is not None and ensemble_versions is not None:
            for i, (trainset, version) in enumerate(zip(ensemble_training_sets, ensemble_versions)):
                print(f"[MODEL LOAD] Ensemble model {i}: training_set={trainset}, version={version}")
        self.target_distances = []
        self.surrogate_distances = []
        """
        target_model: deepspeech model we are attacking
        sound: raw sound data [-1 to +1] (read from torchaudio.load)
        target: target string
        target_decoder: decoder for target_model
        sample_rate: sample rate of sound
        device: device to run on
        save: path to save adversarial sound
        surrogate_decoder: decoder for surrogate model
        surrogate_version: "v1" or "v2" for surrogate model
        target_version: "v1" or "v2" for target model
        target_training_set: "librispeech" or "ted" or "an4" for target model
        """
        self.surrogate_model = surrogate_model
        self.sound = sound
        self.sample_rate = sample_rate
        self.target_string = target
        self.target = target
        self.__init_target()
        self.target_model = target_model
        self.target_model.to(device)
        self.target_model.eval()
        self.target_decoder = target_decoder
        self.surrogate_decoder = surrogate_decoder
        self.criterion = nn.CTCLoss()
        self.device = device
        self.surrogate_version = surrogate_version
        self.target_version = target_version
        self.target_training_set = target_training_set
        n_fft = int(self.sample_rate * 0.02)
        hop_length = int(self.sample_rate * 0.01)
        win_length = int(self.sample_rate * 0.02)
        self.torch_stft = STFT(n_fft=n_fft , hop_length=hop_length, win_length=win_length ,  window='hamming', center=True, pad_mode='reflect', freeze_parameters=True, device=self.device)
        self.save = save

        self.ensemble_versions=ensemble_versions
        self.ensemble_training_sets=ensemble_training_sets
        self.ensemble_models=[m for m, _, _ in ensemble_model_info]
        for model in self.ensemble_models:
            model.to(device)


        self.ensemble_decoders=[d for _, d, _ in ensemble_model_info]
        self.ensemble_weights = [1 / len(ensemble_model_info) for _ in ensemble_model_info] if not ensemble_weights else ensemble_weights


    def get_ori_spec(self, save=None):
        spec = torch_spectrogram(self.sound.to(self.device), self.torch_stft)
        plt.imshow(spec.cpu().numpy()[0][0])
        if save:
            plt.savefig(save)
            plt.clf()
        else:
            plt.show()

    def get_adv_spec(self, save=None):
        spec = torch_spectrogram(self.perturbed_data.to(self.device), self.torch_stft)
        plt.imshow(spec.cpu().numpy()[0][0])
        if save:
            plt.savefig(save)
            plt.clf()
        else:
            plt.show()

    def _forward_model(self, model, version, spec, input_sizes=None):
        version = str(version).strip().lower()
        if version == "v2":
            return model(spec, input_sizes)
        elif version == "v1":
            out = model(spec)
            output_sizes = torch.IntTensor([out.size(1)] * out.size(0))  # batch size
            return out, output_sizes
        else:
            raise ValueError(f"Unknown model version: {version}")

    # prepare
    def __init_target(self):
        self.target = target_sentence_to_label(self.target)
        self.target = self.target.view(1,-1)
        self.target_lengths = torch.IntTensor([self.target.shape[1]]).view(1,-1)

    # FGSM
    def fgsm_attack(self, sound, epsilon, data_grad):

        # find direction of gradient
        sign_data_grad = data_grad.sign()

        # add noise "epilon * direction" to the ori sound
        perturbed_sound = sound - epsilon * sign_data_grad

        return perturbed_sound

    # PGD
    def pgd_attack(self, sound, ori_sound, eps, alpha, data_grad) :

        adv_sound = sound - alpha * data_grad.sign() # + -> - !!!
        eta = torch.clamp(adv_sound - ori_sound.data, min=-eps, max=eps)
        sound = ori_sound + eta

        return sound

    # estimate gradient using NES black-box approach for audio
    def estimate_gradient(self, sound, target_labels, n_queries=25, true_blackbox=False):
        sigma = 0.05  # noise standard deviation
        grad = torch.zeros_like(sound)
        # print(f"sound shape: {sound.shape}") # Keep commented for now
        num_valid_queries = 0

        for i in range(n_queries):
            print(f"NES gradient estimation: query {i+1}/{n_queries}", end="\r")
            u_i = torch.randn_like(sound)
            u_i = u_i / (torch.norm(u_i) + 1e-8) # Normalize perturbation

            sound_plus = torch.clamp(sound + sigma * u_i, min=-1, max=1)
            sound_minus = torch.clamp(sound - sigma * u_i, min=-1, max=1)

            if true_blackbox:
                # _compute_distance uses self.target_string implicitly
                loss_plus = self._compute_distance(sound_plus)
                loss_minus = self._compute_distance(sound_minus)
            else:
                # _compute_ctc_loss needs target_labels
                loss_plus = self._compute_ctc_loss(sound_plus, target_labels)
                loss_minus = self._compute_ctc_loss(sound_minus, target_labels)

            if isinstance(loss_plus, float) and isinstance(loss_minus, float) and not (math.isnan(loss_plus) or math.isinf(loss_plus) or math.isnan(loss_minus) or math.isinf(loss_minus)):
                grad += (loss_plus - loss_minus) * u_i
                num_valid_queries +=1
            # else: # Optional: print if skipping due to NaN/Inf
                # print(f"\nSkipping query {i+1} due to NaN/Inf in loss values (L+: {loss_plus}, L-: {loss_minus}).")

        print(" " * 70, end="\r")  # Clear the line
        if num_valid_queries == 0:
            print("\nWarning: No valid queries in estimate_gradient. Returning zero gradient.")
            return torch.zeros_like(grad)
        return -grad / (num_valid_queries * 2 * sigma) # Average over valid queries

    # compute CTC loss for audio - this is a helper for gradient estimation
    def _compute_ctc_loss(self, sound, target_labels):
        with torch.no_grad():
            spec = torch_spectrogram(sound, self.torch_stft)
            # Assuming spec is (B,C,Time,Freq) after permute, model needs Time for input_sizes
            input_sizes = torch.IntTensor([spec.size(2)]).int() # Corrected to use Time dimension
            out, output_sizes = self.target_model(spec, input_sizes)
            out = out.transpose(0, 1).log_softmax(2)
            loss = self.criterion(out, target_labels, output_sizes, self.target_lengths)
        return loss.item()

    def _compute_distance(self, sound):
        """Compute score for black-box attack using only final transcription"""
        with torch.no_grad():
            spec = torch_spectrogram(sound, self.torch_stft)
            # Assuming spec is (B,C,Time,Freq) after permute
            input_sizes = torch.IntTensor([spec.size(2)]).int() # Corrected to use Time dimension
            out, output_sizes = self.target_model(spec, input_sizes)
            decoded_output, _ = self.target_decoder.decode(out, output_sizes)
            transcription = decoded_output[0][0]
            distance = Levenshtein.distance(transcription, self.target_string)
        return distance

    # mostly ai generated. just for testing if our gradient estimation was used
    # this is not used in any actual attacks
    def _compare_gradients(self, sound, n_queries=25):
        """Compare NES estimated gradient with actual gradient"""
        target_labels_device = self.target.to(self.device) # Use the label tensor

        print("Computing actual gradient via backpropagation...")
        sound_copy = sound.clone().detach().requires_grad_(True)
        actual_grad = torch.zeros_like(sound) # Default to zero grad
        try:
            spec = torch_spectrogram(sound_copy, self.torch_stft)
            # Assuming spec is (B,C,Time,Freq) after permute
            input_sizes = torch.IntTensor([spec.size(2)]).int() # Corrected to use Time dimension
            out, output_sizes = self.target_model(spec, input_sizes)
            out = out.transpose(0, 1).log_softmax(2)
            if torch.isnan(out).any():
                print("WARNING: NaN values detected in model outputs for actual grad!")
                out = torch.nan_to_num(out)
            loss = self.criterion(out, target_labels_device, output_sizes, self.target_lengths)
            self.target_model.zero_grad()
            loss.backward()
            if sound_copy.grad is not None:
                actual_grad_temp = sound_copy.grad.data.clone()
                # Apply gradient clipping to prevent extreme values
                max_norm = 1.0 # You can adjust this value
                grad_norm = torch.norm(actual_grad_temp)
                if grad_norm > max_norm:
                    actual_grad_temp = actual_grad_temp * (max_norm / grad_norm)
                actual_grad = torch.nan_to_num(actual_grad_temp)
            else:
                print("WARNING: sound_copy.grad is None after backward(). Actual grad will be zeros.")
        except Exception as e:
            print(f"Error computing actual gradient: {e}")
            actual_grad = torch.nan_to_num(torch.zeros_like(sound)) # Ensure it's a tensor

        print("Computing estimated gradient with NES (direct audio perturbation)...")
        estimated_grad_raw = self.estimate_gradient(sound, target_labels_device, n_queries)
        estimated_grad = torch.nan_to_num(estimated_grad_raw)

        # Apply smoothing to both gradients
        # Kernel size 5, stride 1, padding 2 for same-size output
        # Ensure gradients are at least 3D for avg_pool1d: (N, C, L)
        actual_grad_smoothed = torch.nn.functional.avg_pool1d(actual_grad.view(1, 1, -1), kernel_size=5, stride=1, padding=2).view_as(actual_grad)
        estimated_grad_smoothed = torch.nn.functional.avg_pool1d(estimated_grad.view(1, 1, -1), kernel_size=5, stride=1, padding=2).view_as(estimated_grad)

        actual_norm = torch.norm(actual_grad_smoothed)
        estimated_norm = torch.norm(estimated_grad_smoothed)
        print(f"Actual gradient norm (smoothed): {actual_norm.item():.6f}")
        print(f"Estimated gradient norm (smoothed): {estimated_norm.item():.6f}")

        cos_sim = torch.tensor(0.0) # Default value
        if actual_norm > 1e-8 and estimated_norm > 1e-8: # Check for non-zero norms
            cos_sim = torch.nn.functional.cosine_similarity(actual_grad_smoothed.flatten(), estimated_grad_smoothed.flatten(), dim=0)
        else:
            print("WARNING: One or both smoothed gradients have near-zero norm. Cosine similarity set to 0.")

        sign_agreement = torch.tensor(0.0)
        non_zero_mask = (actual_grad_smoothed.abs() > 1e-8) & (estimated_grad_smoothed.abs() > 1e-8)
        if non_zero_mask.sum() > 0:
            sign_agreement = (actual_grad_smoothed[non_zero_mask].sign() == estimated_grad_smoothed[non_zero_mask].sign()).float().mean()
        else:
            print("WARNING: No overlapping significant elements for sign agreement after smoothing.")

        top_sign_agreement = torch.tensor(0.0)
        try:
            if actual_norm > 1e-8:
                k = max(int(0.1 * actual_grad_smoothed.numel()), 1)
                if k > 0 and actual_grad_smoothed.numel() > 0:
                    _, top_indices = torch.topk(torch.abs(actual_grad_smoothed).flatten(), k)
                    if top_indices.numel() > 0:
                        top_actual_signs = torch.gather(actual_grad_smoothed.flatten().sign(), 0, top_indices)
                        top_estimated_signs = torch.gather(estimated_grad_smoothed.flatten().sign(), 0, top_indices)
                        top_sign_agreement = (top_actual_signs == top_estimated_signs).float().mean()
            else:
                 print("WARNING: Actual smoothed gradient norm near zero, skipping top sign agreement.")
        except Exception as e:
            print(f"Error computing top sign agreement: {e}")

        print(f"Gradient comparison (smoothed gradients):")
        print(f"  Cosine similarity: {cos_sim.item():.4f}")
        print(f"  Overall sign agreement: {sign_agreement.item():.4f}")
        print(f"  Top 10% sign agreement: {top_sign_agreement.item():.4f}")

        return cos_sim.item(), sign_agreement.item(), top_sign_agreement.item() # Restored top_sign_agreement

    def attack(self, epsilon=None, alpha=None, attack_type="ENSEMBLE", PGD_round=10, n_queries=1000):
        # Set all models to train mode for backward compatibility with cuDNN RNNs
        self.target_model.train()
        for model in self.ensemble_models:
            model.train()
        # Ensure BatchNorm stats remain frozen
        freeze_batchnorm_stats(self.target_model)
        for model in self.ensemble_models:
            freeze_batchnorm_stats(model)

        # Disable Dropout (make it behave as in eval) for all models
        def disable_dropout(model):
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
                    module.train(False)  # Ensures dropout is off even in train mode by setting to eval mode
        disable_dropout(self.target_model)
        for model in self.ensemble_models:
            disable_dropout(model)

        print("Start attack")
        # Ensure correct model modes for attack
        self.surrogate_model.train()
        # Removed self.target_model.eval() call
        # Freeze BatchNorm running statistics to prevent models from diverging
        freeze_batchnorm_stats(self.target_model)
        for model in self.ensemble_models:
            freeze_batchnorm_stats(model)

        data, target = self.sound.to(self.device), self.target.to(self.device)
        data_raw = data.clone().detach()
        # initial prediction (target model)
        spec = torch_spectrogram(data, self.torch_stft)
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        out, output_sizes = run_model(self.target_model, self.target_version, spec, input_sizes)
        original_output = decode_model_output(self.target_version, self.target_decoder, out, output_sizes)
        print(f"Original prediction (target): {original_output}")


        # TEST GRADIENT ESTIMATION
        if attack_type == "TEST_BB":
            # test how well we estimate gradient using NES
            self._compare_gradients(data, n_queries)
            perturbed_data = data

        # BLACK BOX ATTACKS

        # NES_GREY: we can peek at logits but we don't have access to the model
        if attack_type == "NES_GREY":
            # we'll assume that we only want pgd in this case
            for i in range(PGD_round):
                print(f"NES + PGD processing ...  {i+1} / {PGD_round}", end="\r")
                # estimate gradient using NES
                data_grad = self.estimate_gradient(data, self.target.to(self.device), n_queries)
                # now that ew've estimated gradient, everything should be the same as PGD
                data = self.pgd_attack(data, data_raw, epsilon, alpha, data_grad).detach_()
            perturbed_data = data

        elif attack_type == "NES_BLACK": # can't even use ctc loss because we don't have access to logits
            for i in range(PGD_round):
                print(f"NES + PGD processing ...  {i+1} / {PGD_round}", end="\r")
                data_grad = self.estimate_gradient(data, self.target.to(self.device), n_queries, true_blackbox=True)
                data = self.pgd_attack(data, data_raw, epsilon, alpha, data_grad).detach_()
            perturbed_data = data

        # WHITE-BOX ATTACKS (original code):

        if attack_type == "FGSM":
            data.requires_grad = True
            spec = torch_spectrogram(data, self.torch_stft)
            input_sizes = torch.IntTensor([spec.size(3)]).int()
            out, output_sizes = self._forward_model(self.surrogate_model, self.surrogate_version, spec, input_sizes)
            out = out.transpose(0, 1)  # TxNxH
            out = out.log_softmax(2)
            loss = self.criterion(out, self.target, output_sizes, self.target_lengths)
            self.surrogate_model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            print(f"data grad shape: {data_grad.shape}")

            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

            # Evaluate after FGSM attack
            # Target
            self.target_model.eval()  # Ensure target model is in eval mode for final evaluation
            self.target_model.zero_grad()  # Clear any accumulated gradients
            spec_t = torch_spectrogram(perturbed_data, self.torch_stft)
            input_sizes_t = torch.IntTensor([spec_t.size(3)]).int()
            out_t, output_sizes_t = run_model(self.target_model, self.target_version, spec_t, input_sizes_t)
            final_output_target = decode_model_output(self.target_version, self.target_decoder, out_t, output_sizes_t)
            l_distance = Levenshtein.distance(self.target_string, final_output_target)
            self.target_distances.append(l_distance)
            # Surrogate
            spec_s = torch_spectrogram(perturbed_data.to(self.device), self.torch_stft)
            input_sizes_s = torch.IntTensor([spec_s.size(3)]).int()
            out_s, output_sizes_s = run_model(self.surrogate_model, self.surrogate_version, spec_s, input_sizes_s)
            surrogate_pred = decode_model_output(self.surrogate_version, self.surrogate_decoder, out_s, output_sizes_s)
            surrogate_distance = Levenshtein.distance(self.target_string, surrogate_pred)
            self.surrogate_distances.append(surrogate_distance)

        elif attack_type == "PGD":
            for i in range(PGD_round):
                print(f"PGD processing ...  {i+1} / {PGD_round}", end="\r")
                data.requires_grad = True

                spec = torch_spectrogram(data, self.torch_stft)
                input_sizes = torch.IntTensor([spec.size(3)]).int()
                out, output_sizes = self._forward_model(self.surrogate_model, self.surrogate_version, spec, input_sizes)
                out = out.transpose(0, 1)  # TxNxH
                out = out.log_softmax(2)
                loss = self.criterion(out, self.target, output_sizes, self.target_lengths)

                self.surrogate_model.zero_grad()
                loss.backward()
                data_grad = data.grad.data

                data = self.pgd_attack(data, data_raw, epsilon, alpha, data_grad).detach_()

                # Evaluate after each PGD step
                # Target
                self.target_model.eval()  # Ensure target model is in eval mode for final evaluation
                self.target_model.zero_grad()  # Clear any accumulated gradients
                spec_t = torch_spectrogram(data, self.torch_stft)
                input_sizes_t = torch.IntTensor([spec_t.size(3)]).int()
                out_t, output_sizes_t = run_model(self.target_model, self.target_version, spec_t, input_sizes_t)
                final_output_target = decode_model_output(self.target_version, self.target_decoder, out_t, output_sizes_t)
                l_distance = Levenshtein.distance(self.target_string, final_output_target)
                self.target_distances.append(l_distance)
                # Surrogate
                spec_s = torch_spectrogram(data.to(self.device), self.torch_stft)
                input_sizes_s = torch.IntTensor([spec_s.size(3)]).int()
                out_s, output_sizes_s = run_model(self.surrogate_model, self.surrogate_version, spec_s, input_sizes_s)
                surrogate_pred = decode_model_output(self.surrogate_version, self.surrogate_decoder, out_s, output_sizes_s)
                surrogate_distance = Levenshtein.distance(self.target_string, surrogate_pred)
                self.surrogate_distances.append(surrogate_distance)
            perturbed_data = data

        elif attack_type == "ENSEMBLE":
            # Store per-model losses for plotting
            import csv
            self.ensemble_loss_histories = [[] for _ in self.ensemble_models]
            self.target_loss_history = []

            for i in range(PGD_round):
                print(f"PGD processing ...  {i+1} / {PGD_round}", end="\r")
                data.requires_grad = True

                spec = torch_spectrogram(data, self.torch_stft)
                input_sizes = torch.IntTensor([spec.size(3)]).int()

                # do fgsm calculation on a list (or zip of parallel lists of models) and do a weighted avg on the losses
                losses = []
                for version, model in zip(self.ensemble_versions, self.ensemble_models):
                    out, output_sizes = self._forward_model(model, version, spec, input_sizes)
                    out = out.transpose(0, 1)  # TxNxH
                    out = out.log_softmax(2)
                    losses.append(self.criterion(out, self.target, output_sizes, self.target_lengths))
                    model.zero_grad()
                # Store each model's loss for this round
                for idx, loss_val in enumerate(losses):
                    self.ensemble_loss_histories[idx].append(loss_val.item() if hasattr(loss_val, "item") else float(loss_val))

                # Compute and store target model loss
                with torch.no_grad():
                    out_t, output_sizes_t = self._forward_model(self.target_model, self.target_version, spec, input_sizes)
                    out_t = out_t.transpose(0, 1)
                    out_t = out_t.log_softmax(2)
                    target_loss = self.criterion(out_t, self.target, output_sizes_t, self.target_lengths)
                self.target_loss_history.append(target_loss.item() if hasattr(target_loss, "item") else float(target_loss))

                loss = sum(w * loss for w, loss in zip(self.ensemble_weights, losses))
                loss.backward()
                data_grad = data.grad.data

                # PGD update
                data = self.pgd_attack(data, data_raw, epsilon, alpha, data_grad).detach_()

                # Debug logs for adversarial input and prediction
                with torch.no_grad():
                    self.target_model.eval()
                    self.target_model.zero_grad()
                    spec_t = torch_spectrogram(data, self.torch_stft)
                    input_sizes_t = torch.IntTensor([spec_t.size(3)]).int()
                    out_t, output_sizes_t = run_model(self.target_model, self.target_version, spec_t, input_sizes_t)
                    final_output_target = decode_model_output(self.target_version, self.target_decoder, out_t, output_sizes_t)
                    l_distance = Levenshtein.distance(self.target_string, final_output_target)
                    self.target_distances.append(l_distance)
                    print(f"[PGD ITER {i+1}] Target model adversarial prediction: {final_output_target}")
                    print(f"[PGD ITER {i+1}] Target model Levenshtein Distance: {l_distance}")
                    print(f"[PGD ITER {i+1}] Adversarial input min/max: {data.min().item():.4f}/{data.max().item():.4f} (mean: {data.mean().item():.4f})")

                # Track ensemble model Levenshtein distances at each PGD step
                if not hasattr(self, 'ensemble_lev_dists_hist'):
                    self.ensemble_lev_dists_hist = [[] for _ in self.ensemble_models]
                for idx, (model, version, decoder, training_set) in enumerate(zip(self.ensemble_models, self.ensemble_versions, self.ensemble_decoders, self.ensemble_training_sets)):
                    spec_e = torch_spectrogram(data.to(self.device), self.torch_stft)
                    input_sizes_e = torch.IntTensor([spec_e.size(3)]).int()
                    model.eval()
                    out_e, output_sizes_e = run_model(model, version, spec_e, input_sizes_e)
                    ensemble_pred = decode_model_output(version, decoder, out_e, output_sizes_e)
                    ensemble_distance = Levenshtein.distance(self.target_string, ensemble_pred)
                    self.ensemble_lev_dists_hist[idx].append(ensemble_distance)
                    print(f"[PGD ITER {i+1}] Ensemble {training_set}_{version} prediction: {ensemble_pred}")
                    print(f"[PGD ITER {i+1}] Ensemble {training_set}_{version} Levenshtein Distance: {ensemble_distance}")

            # Plot Levenshtein distances as a line plot for target and each ensemble model
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            # Target model distances
            if hasattr(self, 'target_distances') and len(self.target_distances) > 0:
                plt.plot(self.target_distances, label=f"Target {self.target_training_set}_{self.target_version}", color="red", linewidth=2, linestyle="--")
            # Ensemble model distances (plot as lines, not horizontal)
            if hasattr(self, 'ensemble_lev_dists_hist'):
                for idx, (dists, ts, ver) in enumerate(zip(self.ensemble_lev_dists_hist, self.ensemble_training_sets, self.ensemble_versions)):
                    plt.plot(dists, label=f"Ensemble {ts}_{ver}", linestyle=":", alpha=0.7)
            plt.xlabel('PGD Iteration')
            plt.ylabel('Levenshtein Distance')
            plt.title('Levenshtein Distance to Target Sentence (Adversarial Output)')
            plt.legend()
            plt.tight_layout()
            plt.savefig('levenshtein_distances_lineplot.png')
            plt.close()
            print("Saved Levenshtein distance line plot to levenshtein_distances_lineplot.png")

            # Save target_distances to CSV
            if hasattr(self, 'target_distances') and len(self.target_distances) > 0:
                csv_filename = "target_levenshtein_distances.csv"
                with open(csv_filename, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([f"Target {self.target_training_set}_{self.target_version}"])
                    for dist in self.target_distances:
                        writer.writerow([dist])
                print(f"Saved target Levenshtein distances to {csv_filename}")
            # Save ensemble Levenshtein histories to CSV
            if hasattr(self, 'ensemble_lev_dists_hist'):
                csv_filename = "ensemble_levenshtein_histories.csv"
                with open(csv_filename, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    header = [f"{ts}_{ver}" for ts, ver in zip(self.ensemble_training_sets, self.ensemble_versions)]
                    writer.writerow(header)
                    for row in zip(*self.ensemble_lev_dists_hist):
                        writer.writerow(row)
                print(f"Saved ensemble Levenshtein histories to {csv_filename}")

            # Save ensemble loss histories to CSV
            csv_filename = "ensemble_loss_histories.csv"
            with open(csv_filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                header = [f"{ts}_{ver}" for ts, ver in zip(self.ensemble_training_sets, self.ensemble_versions)] + [f"target_{self.target_training_set}_{self.target_version}"]
                writer.writerow(header)
                for row in zip(*self.ensemble_loss_histories, self.target_loss_history):
                    writer.writerow(row)
            print(f"Saved ensemble loss histories to {csv_filename}")

            avg_loss_history = [sum(losses_at_step)/len(losses_at_step) for losses_at_step in zip(*self.ensemble_loss_histories)]

            plt.figure(figsize=(10, 6))
            for idx, (loss_history, training_set, version) in enumerate(zip(self.ensemble_loss_histories, self.ensemble_training_sets, self.ensemble_versions)):
                plt.plot(loss_history, label=f"{training_set}_{version}", alpha=0.7)
            plt.plot(avg_loss_history, label="Average Ensemble Loss", color="black", linewidth=3)
            plt.plot(self.target_loss_history, label=f"Target {self.target_training_set}_{self.target_version}", color="red", linewidth=2, linestyle="--")
            plt.xlabel("PGD Iteration")
            plt.ylabel("Loss")
            plt.title("Ensemble and Target Model Losses During PGD Attack")
            plt.legend()
            plt.tight_layout()
            plt.savefig("ensemble_losses.png")
            plt.close()
            print("Saved ensemble loss plot to ensemble_losses.png")

        # prediction of adversarial sound (evaluate on target model)
        with torch.no_grad():
            self.target_model.eval()  # Ensure target model is in eval mode for final evaluation
            self.target_model.zero_grad()  # Clear any accumulated gradients
            spec = torch_spectrogram(perturbed_data, self.torch_stft)
            input_sizes = torch.IntTensor([spec.size(3)]).int()
            out, output_sizes = run_model(self.target_model, self.target_version, spec, input_sizes)

            final_output_target = decode_model_output(self.target_version, self.target_decoder, out, output_sizes)
        perturbed_data = perturbed_data.detach()
        abs_ori = 20*np.log10(np.sqrt(np.mean(np.absolute(data_raw.cpu().numpy())**2)))
        abs_after = 20*np.log10(np.sqrt(np.mean(np.absolute(perturbed_data.cpu().numpy())**2)))
        db_difference = abs_after - abs_ori
        print(f"[TARGET {self.target_training_set}_{self.target_version}] Adversarial prediction: {final_output_target}")
        l_distance = Levenshtein.distance(self.target_string, final_output_target)
        print(f"[TARGET {self.target_training_set}_{self.target_version}] Levenshtein Distance: {l_distance}")
        print(f"Max Decibel Difference: {db_difference:.4f}")


        if self.save:
            torchaudio.save(self.save, src=perturbed_data.cpu(), sample_rate=self.sample_rate)
        self.perturbed_data = perturbed_data
        return db_difference, l_distance, self.target_string, final_output_target, self.target_distances, self.surrogate_distances
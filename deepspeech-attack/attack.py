from stft import STFT, magphase
import torch.nn as nn
import torch
import Levenshtein
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import math

def target_sentence_to_label(sentence, labels="_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "):
    out = []
    for word in sentence:
        out.append(labels.index(word))
    return torch.IntTensor(out)

def torch_spectrogram(sound, torch_stft_func):
    """Computes spectrograms.
    Returns:
        spec_for_model (Tensor): Normalized log-mag spectrogram, permuted (B,C,Time,Freq).
        norm_log_mag_unpermuted (Tensor): Normalized log-mag spectrogram, unpermuted (B,C,Freq,Time).
    """
    real, imag = torch_stft_func(sound)
    mag, _, _ = magphase(real, imag)
    log_mag = torch.log1p(mag) # Shape: (B, C, Freq, Time)
    
    # Per-instance, per-channel normalization
    mean = log_mag.mean(dim=(2,3), keepdim=True)
    std = log_mag.std(dim=(2,3), keepdim=True)
    norm_log_mag_unpermuted = (log_mag - mean) / (std + 1e-8)
    
    # Permute for the model: (B,C,Freq,Time) -> (B,C,Time,Freq)
    spec_for_model = norm_log_mag_unpermuted.permute(0,1,3,2)
    return spec_for_model, norm_log_mag_unpermuted


class Attacker:
    def __init__(self, model, sound, target, decoder, sample_rate=16000, device="cpu", save=None):
        """
        model: deepspeech model
        sound: raw sound data [-1 to +1] (read from torchaudio.load)
        label: string
        """
        self.sound = sound.to(device)
        self.sample_rate = sample_rate
        self.target_string = target
        self.target = target
        self.__init_target()
        
        self.model = model
        self.model.to(device)
        self.model.train()
        self.decoder = decoder
        self.criterion = nn.CTCLoss()
        self.device = device
        n_fft = int(self.sample_rate * 0.02)
        hop_length = int(self.sample_rate * 0.01)
        win_length = int(self.sample_rate * 0.02)
        self.torch_stft_func = STFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hamming', center=True, pad_mode='reflect', freeze_parameters=True, device=self.device)
        self.save = save
    
    def get_ori_spec(self, save=None):
        spec_for_model, _ = torch_spectrogram(self.sound, self.torch_stft_func)
        plt.imshow(spec_for_model.cpu().detach().numpy()[0][0])
        if save:
            plt.savefig(save)
            plt.clf()
        else:
            plt.show()

    def get_adv_spec(self, save=None):
        if hasattr(self, 'perturbed_data'):
            spec_for_model, _ = torch_spectrogram(self.perturbed_data, self.torch_stft_func)
            plt.imshow(spec_for_model.cpu().detach().numpy()[0][0])
            if save:
                plt.savefig(save)
                plt.clf()
            else:
                plt.show()
        else:
            print("No perturbed data to visualize.")
    
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
    def estimate_audio_gradient_nes(self, sound, target_labels, n_queries=25, true_blackbox=False, sigma=0.05):
        grad = torch.zeros_like(sound)
        num_valid_queries = 0
        for i in range(n_queries):
            print(f"NES for AudioGrad: query {i+1}/{n_queries}", end="\r")
            u_i = torch.randn_like(sound) / (torch.norm(torch.randn_like(sound)) + 1e-8)
            sound_plus = torch.clamp(sound + sigma * u_i, min=-1, max=1)
            sound_minus = torch.clamp(sound - sigma * u_i, min=-1, max=1)
            loss_plus = self._compute_ctc_loss_from_spec(torch_spectrogram(sound_plus, self.torch_stft_func)[0], target_labels) if not true_blackbox else self._compute_distance(sound_plus)
            loss_minus = self._compute_ctc_loss_from_spec(torch_spectrogram(sound_minus, self.torch_stft_func)[0], target_labels) if not true_blackbox else self._compute_distance(sound_minus)
            if not (math.isnan(loss_plus) or math.isinf(loss_plus) or math.isnan(loss_minus) or math.isinf(loss_minus)):
                grad += (loss_plus - loss_minus) * u_i
                num_valid_queries +=1
        print(" "*70, end="\r")
        if num_valid_queries == 0: return torch.zeros_like(grad)
        return -grad / (num_valid_queries * 2 * sigma)

    # compute CTC loss for audio - this is a helper for gradient estimation
    def _compute_ctc_loss_from_spec(self, spec_for_model, target_labels_for_loss):
        with torch.no_grad():
            input_sizes = torch.IntTensor([spec_for_model.size(2)]).int().to(self.device)
            out, output_sizes = self.model(spec_for_model, input_sizes)
            out = out.transpose(0, 1).log_softmax(2)
            loss = self.criterion(out, target_labels_for_loss, output_sizes, self.target_lengths)
        return loss.item()
    
    def _compute_distance(self, sound):
        """Compute score for black-box attack using only final transcription"""
        with torch.no_grad():
            spec_for_model, _ = torch_spectrogram(sound, self.torch_stft_func)
            # Assuming spec is (B,C,Time,Freq) after permute
            input_sizes = torch.IntTensor([spec_for_model.size(2)]).int() # Corrected to use Time dimension
            out, output_sizes = self.model(spec_for_model, input_sizes)
            decoded_output, _ = self.decoder.decode(out, output_sizes)
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
            spec_for_model, _ = torch_spectrogram(sound_copy, self.torch_stft_func)
            # Assuming spec is (B,C,Time,Freq) after permute
            input_sizes = torch.IntTensor([spec_for_model.size(2)]).int() # Corrected to use Time dimension
            out, output_sizes = self.model(spec_for_model, input_sizes)
            out = out.transpose(0, 1).log_softmax(2)
            if torch.isnan(out).any():
                print("WARNING: NaN values detected in model outputs for actual grad!")
                out = torch.nan_to_num(out)
            loss = self.criterion(out, target_labels_device, output_sizes, self.target_lengths)
            self.model.zero_grad()
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
        estimated_grad_raw = self.estimate_audio_gradient_nes(sound, target_labels_device, n_queries)
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

    def estimate_spectrogram_gradient_nes(self, initial_sound, target_labels_for_loss, n_queries=50, sigma_spec=0.1):
        """ Estimates d(Loss)/d(norm_log_mag_unpermuted) using NES. """
        _, norm_log_mag_unpermuted_target = torch_spectrogram(initial_sound.clone().detach(), self.torch_stft_func)
        # print(f"[NES on Spectrogram] Shape of norm_log_mag_unpermuted_target (B,C,Freq,Time): {norm_log_mag_unpermuted_target.shape}") # Print dimensions
        
        grad_norm_log_mag_accum = torch.zeros_like(norm_log_mag_unpermuted_target)
        num_valid_queries = 0

        for i in range(n_queries):
            print(f"NES for SpecGrad: query {i+1}/{n_queries}", end="\r")
            u_i_spec = torch.randn_like(norm_log_mag_unpermuted_target)
            u_i_spec = u_i_spec / (torch.norm(u_i_spec.flatten(), p=2) + 1e-8)
            
            perturbed_spec_plus = norm_log_mag_unpermuted_target + sigma_spec * u_i_spec
            perturbed_spec_minus = norm_log_mag_unpermuted_target - sigma_spec * u_i_spec
            
            spec_for_loss_plus = perturbed_spec_plus.permute(0,1,3,2)
            spec_for_loss_minus = perturbed_spec_minus.permute(0,1,3,2)
            
            loss_plus = self._compute_ctc_loss_from_spec(spec_for_loss_plus, target_labels_for_loss)
            loss_minus = self._compute_ctc_loss_from_spec(spec_for_loss_minus, target_labels_for_loss)

            if not (math.isnan(loss_plus) or math.isinf(loss_plus) or math.isnan(loss_minus) or math.isinf(loss_minus)):
                directional_derivative_spec = (loss_plus - loss_minus) / (2 * sigma_spec)
                grad_norm_log_mag_accum += directional_derivative_spec * u_i_spec
                num_valid_queries += 1
        
        print(" " * 70, end="\r")
        if num_valid_queries == 0:
            print("\nWarning: No valid queries in estimate_spectrogram_gradient_nes.")
            return torch.zeros_like(norm_log_mag_unpermuted_target)
        return -grad_norm_log_mag_accum / num_valid_queries

    def _compare_spectrogram_gradients(self, sound_tensor, n_queries=100, sigma_spec=0.1):
        print("Comparing Spectrogram Gradients...")
        target_labels_device = self.target.to(self.device)

        # --- True Spectrogram Gradient --- 
        print("Calculating true spectrogram gradient...")
        sound_for_true_grad = sound_tensor.clone().detach()
        _, norm_log_mag_unpermuted_actual = torch_spectrogram(sound_for_true_grad, self.torch_stft_func)
        norm_log_mag_unpermuted_actual.requires_grad_(True)
        
        spec_for_model_actual = norm_log_mag_unpermuted_actual.permute(0,1,3,2)
        input_sizes_actual = torch.IntTensor([spec_for_model_actual.size(2)]).int().to(self.device)
        out_actual, output_sizes_actual = self.model(spec_for_model_actual, input_sizes_actual)
        out_actual_processed = out_actual.transpose(0,1).log_softmax(2)
        
        if torch.isnan(out_actual_processed).any():
            print("Warning: NaN in model output for true spec grad!")
            out_actual_processed = torch.nan_to_num(out_actual_processed)

        loss_actual = self.criterion(out_actual_processed, target_labels_device, output_sizes_actual, self.target_lengths)
        self.model.zero_grad()
        loss_actual.backward()
        true_spec_grad = norm_log_mag_unpermuted_actual.grad.clone() if norm_log_mag_unpermuted_actual.grad is not None else torch.zeros_like(norm_log_mag_unpermuted_actual)
        true_spec_grad = torch.nan_to_num(true_spec_grad)

        # --- Estimated Spectrogram Gradient (NES) --- 
        print("Estimating spectrogram gradient with NES...")
        estimated_spec_grad = self.estimate_spectrogram_gradient_nes(sound_tensor, target_labels_device, n_queries, sigma_spec)
        estimated_spec_grad = torch.nan_to_num(estimated_spec_grad)

        # --- Comparison --- 
        # Smoothing (optional, but can help for noisy high-dim grads)
        # For spectrograms (B,C,F,T), we might pool over F and T if desired, or just flatten
        # Here, let's flatten for direct comparison. For avg_pool, would need to reshape.
        # For simplicity, let's compare flattened versions directly. If too noisy, add specific pooling.

        true_flat = true_spec_grad.flatten()
        est_flat = estimated_spec_grad.flatten()

        true_norm = torch.norm(true_flat, p=2)
        est_norm = torch.norm(est_flat, p=2)
        print(f"True Spec Grad Norm: {true_norm.item():.6f}, Estimated Spec Grad Norm: {est_norm.item():.6f}")

        cos_sim = torch.tensor(0.0, device=self.device)
        if true_norm > 1e-8 and est_norm > 1e-8:
            cos_sim = torch.nn.functional.cosine_similarity(true_flat, est_flat, dim=0)
        else:
            print("Warning: Near-zero norm for one or both spec grads.")

        sign_agreement = torch.tensor(0.0, device=self.device)
        non_zero_mask = (true_flat.abs() > 1e-8) & (est_flat.abs() > 1e-8)
        if non_zero_mask.sum() > 0:
            sign_agreement = (true_flat[non_zero_mask].sign() == est_flat[non_zero_mask].sign()).float().mean()
        else:
            print("Warning: No overlapping significant elements for spec grad sign agreement.")
        
        print(f"Spectrogram Gradient Comparison:")
        print(f"  Cosine Similarity: {cos_sim.item():.4f}")
        print(f"  Sign Agreement: {sign_agreement.item():.4f}")
        return cos_sim.item(), sign_agreement.item()

    def attack(self, epsilon, alpha, attack_type="FGSM", PGD_round=40, n_queries=25):
        print("Start attack")
        data = self.sound # Already on device from __init__
        target_labels_device = self.target.to(self.device) # Already on device
        data_raw = data.clone().detach()

        # Initial prediction
        spec_for_model_orig, _ = torch_spectrogram(data, self.torch_stft_func)
        input_sizes_orig = torch.IntTensor([spec_for_model_orig.size(2)]).int().to(self.device)
        out_orig, output_sizes_orig = self.model(spec_for_model_orig, input_sizes_orig)
        decoded_output_orig, _ = self.decoder.decode(out_orig, output_sizes_orig)
        original_output = decoded_output_orig[0][0] if decoded_output_orig and decoded_output_orig[0] else ""
        print(f"Original prediction: {original_output}")

        perturbed_data = data.clone() # Start with a copy for attacks

        if attack_type == "TEST_BB":
            print("--- Running Spectrogram Gradient Comparison (TEST_BB) ---")
            self._compare_spectrogram_gradients(data, n_queries=n_queries) # data is original sound
            # No actual attack in TEST_BB, perturbed_data remains original sound
        
        elif attack_type == "FGSM": # White-box FGSM
            print("Performing FGSM attack...")
            perturbed_data.requires_grad = True
            spec_for_model, _ = torch_spectrogram(perturbed_data, self.torch_stft_func)
            input_sizes = torch.IntTensor([spec_for_model.size(2)]).int().to(self.device)
            out, output_sizes = self.model(spec_for_model, input_sizes)
            out_processed = out.transpose(0, 1).log_softmax(2)
            loss = self.criterion(out_processed, target_labels_device, output_sizes, self.target_lengths)
            self.model.zero_grad()
            loss.backward()
            audio_grad = perturbed_data.grad.data.clone()
            perturbed_data = self.fgsm_attack(perturbed_data.detach(), epsilon, audio_grad) # Detach before passing to attack func

        elif attack_type == "PGD": # White-box PGD
            print("Performing PGD attack...")
            for i in range(PGD_round):
                print(f"PGD iter {i+1}/{PGD_round}", end="\r")
                perturbed_data.requires_grad = True
                spec_for_model, _ = torch_spectrogram(perturbed_data, self.torch_stft_func)
                input_sizes = torch.IntTensor([spec_for_model.size(2)]).int().to(self.device)
                out, output_sizes = self.model(spec_for_model, input_sizes)
                out_processed = out.transpose(0, 1).log_softmax(2)
                loss = self.criterion(out_processed, target_labels_device, output_sizes, self.target_lengths)
                self.model.zero_grad()
                if perturbed_data.grad is not None: # Zero existing grads before new backward
                    perturbed_data.grad.zero_()
                loss.backward()
                audio_grad = perturbed_data.grad.data.clone()
                perturbed_data = self.pgd_attack(perturbed_data.detach(), data_raw, epsilon, alpha, audio_grad)
            print(" " * 70, end="\r")

        elif attack_type == "NES_GREY": # Black-box (grey-box) audio perturbation PGD
            print("Performing NES_GREY (audio perturbation) PGD attack...")
            for i in range(PGD_round):
                print(f"NES_GREY PGD iter {i+1}/{PGD_round}", end="\r")
                # Use estimate_audio_gradient_nes, true_blackbox=False (default)
                audio_grad = self.estimate_audio_gradient_nes(perturbed_data, target_labels_device, n_queries)
                perturbed_data = self.pgd_attack(perturbed_data, data_raw, epsilon, alpha, audio_grad).detach()
            print(" " * 70, end="\r")

        elif attack_type == "NES_BLACK": # True Black-box audio perturbation PGD
            print("Performing NES_BLACK (audio perturbation, Levenshtein loss) PGD attack...")
            for i in range(PGD_round):
                print(f"NES_BLACK PGD iter {i+1}/{PGD_round}", end="\r")
                # Use estimate_audio_gradient_nes, true_blackbox=True
                audio_grad = self.estimate_audio_gradient_nes(perturbed_data, target_labels_device, n_queries, true_blackbox=True)
                perturbed_data = self.pgd_attack(perturbed_data, data_raw, epsilon, alpha, audio_grad).detach()
            print(" " * 70, end="\r")

        elif attack_type == "NES_SPECTROGRAM_PGD": # Black-box spectrogram perturbation PGD
            print("Performing NES_SPECTROGRAM_PGD attack...")
            for i in range(PGD_round):
                print(f"NES_SPECTROGRAM PGD iter {i+1}/{PGD_round}", end="\r")
                # 1. Estimate d(L)/d(Spec_unpermuted)
                # sigma_spec might need tuning for attacks
                estimated_grad_wrt_norm_log_mag = self.estimate_spectrogram_gradient_nes(perturbed_data, target_labels_device, n_queries, sigma_spec=0.1)
                
                # 2. Propagate this to d(L)/d(Audio)
                sound_for_grad_prop = perturbed_data.clone().detach().requires_grad_(True)
                _, norm_log_mag_actual = torch_spectrogram(sound_for_grad_prop, self.torch_stft_func)
                
                # Zero existing grads before new backward, if any (though sound_for_grad_prop is fresh)
                if sound_for_grad_prop.grad is not None:
                    sound_for_grad_prop.grad.zero_()
                
                norm_log_mag_actual.backward(gradient=estimated_grad_wrt_norm_log_mag)
                audio_grad_for_attack = sound_for_grad_prop.grad
                
                if audio_grad_for_attack is None:
                    print("\nWarning: Audio gradient is None after spec grad propagation. Skipping PGD step.")
                    audio_grad_for_attack = torch.zeros_like(perturbed_data) # Fallback to zero gradient
                else:
                    audio_grad_for_attack = audio_grad_for_attack.clone()
                
                # 3. Use this d(L)/d(Audio) in PGD/FGSM.
                perturbed_data = self.pgd_attack(perturbed_data.detach(), data_raw, epsilon, alpha, audio_grad_for_attack)
            print(" " * 70, end="\r")
        else:
            print(f"Attack type '{attack_type}' is not recognized or fully implemented. Performing no attack.")
            # perturbed_data remains original sound if attack type not matched

        # Final evaluation
        spec_for_model_adv, _ = torch_spectrogram(perturbed_data.detach(), self.torch_stft_func)
        input_sizes_adv = torch.IntTensor([spec_for_model_adv.size(2)]).int().to(self.device)
        out_adv, output_sizes_adv = self.model(spec_for_model_adv, input_sizes_adv)
        decoded_output_adv, _ = self.decoder.decode(out_adv, output_sizes_adv)
        final_output = decoded_output_adv[0][0] if decoded_output_adv and decoded_output_adv[0] else ""
        
        db_difference = 0.0
        if torch.mean(torch.absolute(data_raw)**2) > 1e-12:
            abs_ori = 20*torch.log10(torch.sqrt(torch.mean(torch.absolute(data_raw)**2)) + 1e-9) # Add epsilon for log
            abs_after = 20*torch.log10(torch.sqrt(torch.mean(torch.absolute(perturbed_data.detach())**2)) + 1e-9) # Add epsilon for log
            db_difference = (abs_after-abs_ori).item()
        else:
            print("Warning: Original audio power is near zero, cannot compute dB difference accurately.")

        l_distance = Levenshtein.distance(self.target_string, final_output)
        # Make sure final_output is a string
        final_output_str = str(final_output) if final_output is not None else ""
        print(f"\nFinal Original Prediction: {original_output}")
        print(f"Target Sentence: {self.target_string}")
        print(f"Max Decibel Difference: {db_difference:.4f}")
        print(f"Adversarial Prediction: {final_output_str}")
        print(f"Levenshtein Distance: {l_distance}")
        
        if self.save:
            save_file = f"{self.save}_{attack_type}.wav"
            try:
                torchaudio.save(save_file, src=perturbed_data.cpu().detach(), sample_rate=self.sample_rate)
                print(f"Saved adversarial audio to {save_file}")
            except Exception as e:
                print(f"Error saving audio file {save_file}: {e}")
        self.perturbed_data = perturbed_data.cpu().detach() # Store for potential get_adv_spec
        return db_difference, l_distance, self.target_string, final_output_str
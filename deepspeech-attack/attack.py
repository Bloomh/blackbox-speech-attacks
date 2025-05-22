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

class Attacker:
    def __init__(self, model, sound, target, decoder, sample_rate=16000, device="cpu", save=None, model_version="v2"):
        """
        model: deepspeech model
        sound: raw sound data [-1 to +1] (read from torchaudio.load)
        label: string
        model_version: "v1" or "v2"
        """
        self.sound = sound
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
        self.model_version = model_version
        n_fft = int(self.sample_rate * 0.02)
        hop_length = int(self.sample_rate * 0.01)
        win_length = int(self.sample_rate * 0.02)
        self.torch_stft = STFT(n_fft=n_fft , hop_length=hop_length, win_length=win_length ,  window='hamming', center=True, pad_mode='reflect', freeze_parameters=True, device=self.device)
        self.save = save
    
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

    def _forward_model(self, spec, input_sizes=None):
        version = str(self.model_version).strip().lower()
        if version == "v2":
            return self.model(spec, input_sizes)
        elif version == "v1":
            out = self.model(spec)
            # For v1, output_sizes is typically the time dimension of the output
            output_sizes = torch.IntTensor([out.size(1)] * out.size(0))  # batch size
            return out, output_sizes
        else:
            raise ValueError(f"Unknown model version: {self.model_version}")

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
            out, output_sizes = self.model(spec, input_sizes)
            out = out.transpose(0, 1).log_softmax(2)
            loss = self.criterion(out, target_labels, output_sizes, self.target_lengths)
        return loss.item()
    
    def _compute_distance(self, sound):
        """Compute score for black-box attack using only final transcription"""
        with torch.no_grad():
            spec = torch_spectrogram(sound, self.torch_stft)
            # Assuming spec is (B,C,Time,Freq) after permute
            input_sizes = torch.IntTensor([spec.size(2)]).int() # Corrected to use Time dimension
            out, output_sizes = self.model(spec, input_sizes)
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
            spec = torch_spectrogram(sound_copy, self.torch_stft)
            # Assuming spec is (B,C,Time,Freq) after permute
            input_sizes = torch.IntTensor([spec.size(2)]).int() # Corrected to use Time dimension
            out, output_sizes = self.model(spec, input_sizes)
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

    def attack(self, epsilon, alpha, attack_type="FGSM", PGD_round=40, n_queries=25):
        print("Start attack")
        
        data, target = self.sound.to(self.device), self.target.to(self.device)
        data_raw = data.clone().detach()
        # initial prediction
        spec = torch_spectrogram(data, self.torch_stft)
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        # Model output
        out, output_sizes = self._forward_model(spec, input_sizes)
        if self.model_version == "v1":
            # Transpose output to [T, N, H] as in test.py
            out = out.transpose(0, 1)  # TxNxH
            seq_length = out.size(0)
            batch_size = out.size(1)
            sizes = torch.IntTensor([seq_length] * batch_size)
            decoded_output = self.decoder.decode(out, sizes)
            original_output = flatten_to_string(decoded_output[0][0])
        else:
            decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
            original_output = decoded_output[0][0]
        print(f"Original prediction: {original_output}")
        
        # ...continue with attack logic...

        
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
            out, output_sizes = self._forward_model(spec, input_sizes)
            out = out.transpose(0, 1)  # T x N x H
            out = out.log_softmax(2)
            loss = self.criterion(out, self.target, output_sizes, self.target_lengths)
            
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            
            print(f"data grad shape: {data_grad.shape}")

            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

        elif attack_type == "PGD":
            for i in range(PGD_round):
                print(f"PGD processing ...  {i+1} / {PGD_round}", end="\r")
                data.requires_grad = True
                
                spec = torch_spectrogram(data, self.torch_stft)
                input_sizes = torch.IntTensor([spec.size(3)]).int()
                out, output_sizes = self._forward_model(spec, input_sizes)
                out = out.transpose(0, 1)  # TxNxH
                out = out.log_softmax(2)
                loss = self.criterion(out, self.target, output_sizes, self.target_lengths)
                
                self.model.zero_grad()
                loss.backward()
                data_grad = data.grad.data

                data = self.pgd_attack(data, data_raw, epsilon, alpha, data_grad).detach_()
            perturbed_data = data
            
        # prediction of adversarial sound
        spec = torch_spectrogram(perturbed_data, self.torch_stft)
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        out, output_sizes = self._forward_model(spec, input_sizes)
        if self.model_version == "v1":
            # Transpose output to [T, N, H] as in test.py
            out = out.transpose(0, 1)  # TxNxH
            seq_length = out.size(0)
            batch_size = out.size(1)
            sizes = torch.IntTensor([seq_length] * batch_size)
            decoded_output = self.decoder.decode(out, sizes)
            final_output = decoded_output[0][0]
        else:
            decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
            final_output = decoded_output[0][0]
        
        perturbed_data = perturbed_data.detach()
        abs_ori = 20*np.log10(np.sqrt(np.mean(np.absolute(data_raw.cpu().numpy())**2)))
        abs_after = 20*np.log10(np.sqrt(np.mean(np.absolute(perturbed_data.cpu().numpy())**2)))
        db_difference = abs_after-abs_ori
        
        final_output = flatten_to_string(final_output)
        l_distance = Levenshtein.distance(self.target_string, final_output)
        print(f"Max Decibel Difference: {db_difference:.4f}")
        print(f"Adversarial prediction: {final_output}")
        print(f"Levenshtein Distance {l_distance}")
        if self.save:
            torchaudio.save(self.save, src=perturbed_data.cpu(), sample_rate=self.sample_rate)
        self.perturbed_data = perturbed_data
        return db_difference, l_distance, self.target_string, final_output
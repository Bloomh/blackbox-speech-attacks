from stft import STFT, magphase
import torch.nn as nn
import torch
import Levenshtein
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

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


class Attacker:
    def __init__(self, model, sound, target, decoder, sample_rate=16000, device="cpu", save=None):
        """
        model: deepspeech model
        sound: raw sound data [-1 to +1] (read from torchaudio.load)
        label: string
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
    def estimate_gradient(self, sound, target, n_queries=25, true_blackbox=False):
        sigma = 0.05  # noise standard deviation
        grad = torch.zeros_like(sound)
        print(f"sound shape: {sound.shape}")
        
        for i in range(n_queries):
            print(f"NES gradient estimation: query {i+1}/{n_queries}", end="\r") # show progress on direction guess queries for gradient estimation
            
            # sample random direction
            u_i = torch.randn_like(sound)
            
            # query model at perturbed points
            sound_plus = torch.clamp(sound + sigma * u_i, min=-1, max=1)
            sound_minus = torch.clamp(sound - sigma * u_i, min=-1, max=1)
            
            # get loss for both perturbed points (using CTC loss if we can peek at logits, otherwise use distance if true blackbox)
            if true_blackbox:
                loss_plus = self._compute_distance(sound_plus)
                loss_minus = self._compute_distance(sound_minus)
            else:
                loss_plus = self._compute_ctc_loss(sound_plus)
                loss_minus = self._compute_ctc_loss(sound_minus)
            
            # update gradient estimate (effectively a weighted sum)
            grad += (loss_plus - loss_minus) * u_i
            
        print(" " * 50, end="\r")  # clear the line after NES queries
        return -grad / (2 * n_queries * sigma)

    # compute CTC loss for audio - this is a helper for gradient estimation
    def _compute_ctc_loss(self, sound):
        with torch.no_grad(): # don't track gradient since we're still tied to the model - we wanna be entirely blackbox
            # this is largely the same as PGD below
            spec = torch_spectrogram(sound, self.torch_stft)
            input_sizes = torch.IntTensor([spec.size(3)]).int()
            out, output_sizes = self.model(spec, input_sizes) # effective blackbox query is right here
            out = out.transpose(0, 1)  # T x N x H
            out = out.log_softmax(2)
            loss = self.criterion(out, self.target, output_sizes, self.target_lengths)
        
        return loss.item()
    
    def _compute_distance(self, sound):
        """Compute score for black-box attack using only final transcription"""
        with torch.no_grad():
            # Get transcription from the model
            spec = torch_spectrogram(sound, self.torch_stft)
            input_sizes = torch.IntTensor([spec.size(3)]).int()
            
            # this would be a true blackbox query where we are only working with the final output rather than logits
            out, output_sizes = self.model(spec, input_sizes)
            decoded_output, _ = self.decoder.decode(out, output_sizes)
            transcription = decoded_output[0][0]
            
            # calculate Levenshtein distance to target
            distance = Levenshtein.distance(transcription, self.target_string) 
            
        # distance being high is bad here so we can treat it as a loss of sorts
        return distance

    # mostly ai generated. just for testing if our gradient estimation was used
    # this is not used in any actual attacks
    def _compare_gradients(self, sound, n_queries=25):
        """Compare NES estimated gradient with actual gradient"""
        sound_copy = sound.clone().detach()
        sound_copy.requires_grad = True
        
        # Compute actual gradient with backprop
        spec = torch_spectrogram(sound_copy, self.torch_stft)
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        out, output_sizes = self.model(spec, input_sizes)
        out = out.transpose(0, 1)  # TxNxH
        out = out.log_softmax(2)
        loss = self.criterion(out, self.target, output_sizes, self.target_lengths)
        
        self.model.zero_grad()
        loss.backward()
        actual_grad = sound_copy.grad.data
        
        # Compute estimated gradient with NES
        estimated_grad = self.estimate_gradient(sound, self.target, n_queries)
        
        # Compare gradients
        cos_sim = torch.nn.functional.cosine_similarity(
            actual_grad.flatten(), estimated_grad.flatten(), dim=0)
        
        # Compare signs
        actual_sign = actual_grad.sign()
        estimated_sign = estimated_grad.sign()
        sign_agreement = (actual_sign == estimated_sign).float().mean()
        
        print(f"Gradient comparison:")
        print(f"  Cosine similarity: {cos_sim.item():.4f} (closer to 1 is better)")
        print(f"  Sign agreement: {sign_agreement.item():.4f} (percentage of matching signs)")
        
        return cos_sim.item(), sign_agreement.item()

    def attack(self, epsilon, alpha, attack_type="FGSM", PGD_round=40, n_queries=25):
        print("Start attack")
        
        data, target = self.sound.to(self.device), self.target.to(self.device)
        data_raw = data.clone().detach()
        
        # initial prediction
        spec = torch_spectrogram(data, self.torch_stft)
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        out, output_sizes = self.model(spec, input_sizes)
        decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
        original_output = decoded_output[0][0]
        print(f"Original prediction: {original_output}")
        
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
                data_grad = self.estimate_gradient(data, target, n_queries)
                # now that ew've estimated gradient, everything should be the same as PGD
                data = self.pgd_attack(data, data_raw, epsilon, alpha, data_grad).detach_()
            perturbed_data = data
            
        elif attack_type == "NES_BLACK": # can't even use ctc loss because we don't have access to logits
            for i in range(PGD_round):
                print(f"NES + PGD processing ...  {i+1} / {PGD_round}", end="\r")
                data_grad = self.estimate_gradient(data, target, n_queries, true_blackbox=True)
                data = self.pgd_attack(data, data_raw, epsilon, alpha, data_grad).detach_()
            perturbed_data = data

        # WHITE-BOX ATTACKS (original code):
        
        if attack_type == "FGSM":
            data.requires_grad = True
            
            spec = torch_spectrogram(data, self.torch_stft)
            input_sizes = torch.IntTensor([spec.size(3)]).int()
            out, output_sizes = self.model(spec, input_sizes)
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
                out, output_sizes = self.model(spec, input_sizes)
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
        out, output_sizes = self.model(spec, input_sizes)
        decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
        final_output = decoded_output[0][0]
        
        perturbed_data = perturbed_data.detach()
        abs_ori = 20*np.log10(np.sqrt(np.mean(np.absolute(data_raw.cpu().numpy())**2)))
        abs_after = 20*np.log10(np.sqrt(np.mean(np.absolute(perturbed_data.cpu().numpy())**2)))
        db_difference = abs_after-abs_ori
        l_distance = Levenshtein.distance(self.target_string, final_output)
        print(f"Max Decibel Difference: {db_difference:.4f}")
        print(f"Adversarial prediction: {decoded_output[0][0]}")
        print(f"Levenshtein Distance {l_distance}")
        if self.save:
            torchaudio.save(self.save, src=perturbed_data.cpu(), sample_rate=self.sample_rate)
        self.perturbed_data = perturbed_data
        return db_difference, l_distance, self.target_string, final_output
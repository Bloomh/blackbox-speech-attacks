#!/bin/bash
#SBATCH --job-name=bloom-blackbox
#SBATCH --partition=speech-course-gpu
#SBATCH -G1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source /scratch/hmbloom/speech/miniconda/envs/py310/bin/activate
cd /share/data/lang/users/ttic_31110/hmbloom/blackbox-speech-attacks

source venv/bin/activate

cd deepspeech2-pytorch-adversarial-attack
python3 preprocessing.py --input_folder sound --output_folder processed_sound
python3 main.py --input_wav processed_sound/normal0.wav --output_wav adversarial_sound/HELLO_WORLD.wav --target_sentence HELLO_WORLD --model ../models/librispeech/librispeech_pretrained_v2.pth --device gpu

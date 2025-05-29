#!/bin/bash
#SBATCH --job-name=ensemble_attack
#SBATCH --output=ensemble_attack_%j.log
#SBATCH --partition=speech-course-gpu
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --mem=24G

# Move to project root
cd /share/data/lang/users/ttic_31110/hmbloom/blackbox-speech-attacks || exit 1

# Activate your virtual environment
source venv/bin/activate

# Move to attack code directory
cd deepspeech-attack || exit 1

# Preprocess audio
python3 preprocessing.py --input_folder sound --output_folder processed_sound

# Run ensemble adversarial attack
python3 black_box.py --input_wav processed_sound/normal0.wav --output_wav adversarial_sound/HELLO_WORLD.wav --target_sentence HELLO_WORLD --device cuda --PGD_iter 1000 --mode ENSEMBLE
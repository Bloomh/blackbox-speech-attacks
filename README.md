(cooked rn) instructions to get off the ground @henry:

- to clone properly, run `git clone --recursive [url]` 
    - we have a recursive dependency so to keep things clean I used a submodule and fixed it on an old commit
- (make+activate venv, then) run `pip install -r requirements.txt` for dep installs
    - see version notes in that file for what I did in case we run into more dep issues
- try to run attack
    - move into `deepspeech2-pytorch-adversarial-attack` directory (i followed their readme instructions to setup the outer `deepspeech.pytorch` so j trust ig lol)
    - if preprocessing of original audio file in `deepspeech2-pytorch-adversarial-attack/sound/` hasn't happened yet, run `python3 preprocessing.py --input_folder sound --output_folder processed_sound`
    - run `python3 main.py --input_wav processed_sound/normal0.wav --output_wav to_save.wav --target_sentence HELLO_WORD --model ../models/librispeech/librispeech_pretrained_v2.pth --device cpu` to do the actual attack (I haven't looked into this too much but it outputs a promising looking file). 
        - If you have gpu or mps, change the device obv.
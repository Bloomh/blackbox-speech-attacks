## (cooked rn) instructions to get off the ground @henry:

- to clone properly, run `git clone --recursive [url]` 
    - we have a recursive dependency so to keep things clean I used a submodule and fixed it on an old commit
- (make+activate venv, then) run `pip install -r requirements.txt` for dep installs
    - see version notes in that file for what I did in case we run into more dep issues
- try to run attack
    - move into `deepspeech2-pytorch-adversarial-attack` directory (i followed their readme instructions to setup the outer `deepspeech.pytorch` so j trust ig lol)
    - if preprocessing of original audio file in `deepspeech2-pytorch-adversarial-attack/sound/` hasn't happened yet, run `python3 preprocessing.py --input_folder sound --output_folder processed_sound`
    - run `python3 main.py --input_wav processed_sound/normal0.wav --output_wav adversarial_sound/HELLO_WORLD.wav --target_sentence HELLO_WORLD --model ../models/librispeech/librispeech_pretrained_v2.pth --device cpu` to do the actual attack (I haven't looked into this too much but it outputs a promising looking file). 
        - If you have gpu or mps, change the device obv.

ideally, if you can get this up and running on your machine, I want you to try replicating this exactly on beehhive while I try to see if we can do anything black-boxy here with NES.

right now: with the default 50 iterations on PGD adn their settings, the performance is kinda awful (may be grounds to go for untargeted attack but we'll see):
- you'll notice that we did `HELLO_WORLD` as the target for the original audio that was from https://nicholas.carlini.com/code/audio_adversarial_examples/, saying "without the dataset, the artical is useless" 
- original recognition is already bad - maybe we self record and find a better audio
- the movement towards `HELLO_WORLD` is pretty bad as can be seen below. I guess I can kinda see what the output is going for?
  - (idk how the `_` works here so if you could look into that would be good)
- we should get this working properly asap, but that can be asynchronous with me figuring out NES if needed.
```
Original prediction: WITTOU THE DATRASET THE ARTYGORSE USELESS
Max Decibel Difference: 0.0283
Adversarial prediction: THD E LI LES WHO FULAD
Levenshtein Distance 6
```

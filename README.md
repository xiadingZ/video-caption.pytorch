# pytorch implementation of video captioning

recommend installing pytorch and python packages using Anaconda

## requirements

- cuda
- pytorch (compiled from master, 0.4.0)
- python3
- ffmpeg (can install using anaconda)

### python packages

- tqdm
- pillow
- pretrainedmodels
- nltk

## Data

MSR-VTT. Test video doesn't have captions, so I spilit train-viedo to train/val/test. Extract and put them in `./data/` directory

- train-video: [download link](https://drive.google.com/file/d/1Qi6Gn_l93SzrvmKQQu-drI90L-x8B0ly/view?usp=sharing)
- test-video: [download link](https://drive.google.com/file/d/10fPbEhD-ENVQihrRvKFvxcMzkDlhvf4Q/view?usp=sharing)
- json info of train-video: [download link](https://drive.google.com/file/d/1LcTtsAvfnHhUfHMiI4YkDgN7lF1-_-m7/view?usp=sharing)
- json info of test-video: [download link](https://drive.google.com/file/d/1Kgra0uMKDQssclNZXRLfbj9UQgBv-1YE/view?usp=sharing)

## Options

all default options are defined in opt.py or corresponding code file, change it for your like.

## Usage

### Steps

1. preprocess videos and labels

    this steps take about 3 hours for msr-vtt datasets use one titan XP gpu

```bash
python prepro_feats.py --output_dir data/feats/incepv4 --model inception_v4 --dim_vid 1536 --n_frame_steps 50 --gpu 0,1 --saved_model data/save_cnn/cnn_model_50.pth

python prepro_vocab.py
```

2. Training a model

```bash
python train.py --gpu 5,6,7 --epochs 9001 --batch_size 450 --checkpoint_path data/save7 --feats_dir data/feats/incepv4 --rnn_dropout_p 0.1 --dim_hidden 1024 --dim_word 512 --dim_vid 1536 --model S2VTAttModel
```

3. test

    opt_info.json will be in same directory as saved model.

```bash
python eval.py --recover_opt data/save5/opt_info.json --saved_model data/save5/model_300.pth --batch_size 10 --gpu 1,0
```

## Metrics

I fork the [coco-caption](https://github.com/tylin/coco-caption) and port it to python3, but `meteor` doesn't work. Welcome to talk to me about that.

## welcome pull request

now the score is very low, still don't know the reason, welcome to talk to me about it

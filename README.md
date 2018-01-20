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

all default options are defined in opt["py or corresponding code file, change it for your like.

## Usage

### Optional

you can pretrain cnn model used to extract features of video frames using coco2014 dataset. Download and put them in `data`.

- coco-dataset 2014 (download from ms-coco website)
- word_counts.txt [download link](https://drive.google.com/open?id=1yor3VwdAzN1Ur5Jev9RClegCuBQuCK09)

extract them to `data/mscoco/train2014` and `data/mscoco/annotations` and `data/mscoco/word_counts.txt`. For details,
see the params of code.

```bash
python prepro_coco.py

python finetune_cnn.py --gpu 4,5,6,7 --checkpoint_path data/save_cnn/ --batch_size 200
```

I have a finetuned inception_v4 model, you can [download it](https://drive.google.com/open?id=1yJW5GVJUaKjmm8B4dak9hxC4lYGMMG51)

### Steps

1. preprocess videos and labels

    this steps take about 3 hours for msr-vtt datasets use one titan XP gpu

```bash
python prepro_feats.py --output_dir data/feats/incepv4 --model inception_v4 --dim_vid 1536 --n_frame_steps 50 --gpu 0,1 --saved_model data/save_cnn/cnn_model_50.pth

python prepro_vocab.py
```

2. Training a model

    To train a model, simply run

```bash
python train.py --gpu 5,6,7 --epochs 9001 --batch_size 450 --checkpoint_path data/save7 --feats_dir data/feats/incepv4 --rnn_dropout_p 0.1 --dim_hidden 1024 --dim_word 512 --dim_vid 1536 --model S2VTAttModel
```

3. test

```bash
python eval.py --mode test --model S2VTAttModel --saved_model data/save7/model_best.pth --gpu 2,3,4 --dim_hidden 1024 --dim_vid 1536 --dim_word 512 --feats_dir data/feats/incepv4
```

## Metrics

I fork the [coco-caption](https://github.com/tylin/coco-caption) and port it to python3, but meteor doesn't work. Welcome to talk to me about that.

## welcome pull request

now the score is very low, still don't know the reason, welcome to talk to me about it

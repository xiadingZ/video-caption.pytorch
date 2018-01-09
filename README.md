# pytorch code for video captioning

## requirements
- pytorch (compiled from master, 0.4.0)
- python3
- tqmd
- ffmpeg
- cuda

## Data
MSR-VTT. Test video doesn't have captions, so I spilit train-viedo to train/val/test. Extract and put them in `./data/` directory
- train-video: [download link](https://drive.google.com/file/d/1Qi6Gn_l93SzrvmKQQu-drI90L-x8B0ly/view?usp=sharing)
- test-video: [download link](https://drive.google.com/file/d/10fPbEhD-ENVQihrRvKFvxcMzkDlhvf4Q/view?usp=sharing)
- json info of train-video: [download link](https://drive.google.com/file/d/1LcTtsAvfnHhUfHMiI4YkDgN7lF1-_-m7/view?usp=sharing)
- json info of test-video: [download link](https://drive.google.com/file/d/1Kgra0uMKDQssclNZXRLfbj9UQgBv-1YE/view?usp=sharing)

## Options
all default options are defined in opt.py, change it for your like

## Usage
1. preprocess videos and labels


    this steps take about 2 hours for msr-vtt datasets
```
python prepro_feats


python prepro_labels
```

2. train

```
python train.py --epochs 3001  --batch_size 512 --checkpoint_path save --gpu 0,1
```

3. test

    due to the restriction of pytorch `DataParallel`, if you train the model with *n* gpus, you should use *n* gpus to load it.

```
python eval.py --mode test --model save/model_1700.pth --gpu 0,1
```

## Metrics
I fork the [coco-caption](https://github.com/tylin/coco-caption) and port it to python3, but meteor doesn't work. Welcome to talk to me about that.

# welcome pull request
now the score is very low, still don't know the reason, welcome to talk to me about it

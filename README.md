# pytorch code for video captioning

## requirements
- pytorch (compiled from master, 0.4.0)
- python3
- tqmd
- ffmpeg
- pretrainedmodels
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
python prepro_feats.py --output_dir data/feats/incepv4 --model inception_v4 --dim_vid 1536 --n_frame_steps 50
```
```
python prepro_labels
```
2. Training a model

    To train a model, simply run
```
python train.py --gpu 5,6,7 --epochs 9001 --batch_size 450 --checkpoint_path data/save7 --feats_dir data/feats/incepv4 --rnn_dropout 0.1 --dim_hidden 1024 --dim_word 512 --dim_vid 1536 --model S2VTAttModel
```


3. test

    due to the restriction of pytorch `DataParallel`, if you train the model with *n* gpus, you should use *n* gpus to load it.

```
python eval.py --mode test --model S2VTAttModel --saved_model data/save7/model_best.pth --gpu 2,3,4 --dim_hidden 1024 --dim_vid 1536 --dim_word 512 --feats_dir data/feats/incepv4
```

## Metrics
I fork the [coco-caption](https://github.com/tylin/coco-caption) and port it to python3, but meteor doesn't work. Welcome to talk to me about that.

# welcome pull request
now the score is very low, still don't know the reason, welcome to talk to me about it

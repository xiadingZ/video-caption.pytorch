# pytorch implementation of video captioning

recommend installing pytorch and python packages using Anaconda

## requirements

- cuda
- pytorch 0.4.0
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

UPDATE: MSR-VTT. Test video have released captions. You can download the all captions from  [baiduyun pwd: nxyk](https://pan.baidu.com/s/1RDNygrWtz_PtVH8nh4vG3w). There are also many extracted features in baiduyun.
- **all_videodatainfo_2017.json**: json info of train-video and test-video
- **c3d_feat**: c3d_kintectics(16f) and so on
- **2d_feat**: resnet101 and nasnet
- **audio_feat**: extracted by tensorflow/models/research/audioset/
- **proposal_feat**: extracted by Detectron

## Options

all default options are defined in opt.py or corresponding code file, change them for your like.

## Usage

### (Optional) c3d features
you can use [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) to extract features from video. 

### Steps

1. preprocess videos and labels

```bash
python prepro_feats.py --output_dir data/feats/resnet152 --model resnet152 --n_frame_steps 40  --gpu 4,5

python prepro_vocab.py
```

2. Training a model

```bash

python train.py --gpu 0 --epochs 3001 --batch_size 300 --checkpoint_path data/save --feats_dir data/feats/resnet152 --dim_vid 2048 --model S2VTAttModel  --with_c3d 1 --c3d_feats_dir data/feats/c3d_feats --dim_vid 4096
```

3. test

    opt_info.json will be in same directory as saved model.

```bash
python eval.py --recover_opt data/save/opt_info.json --saved_model data/save/model_1000.pth --batch_size 100 --gpu 1
```

## Metrics

I fork the [coco-caption XgDuan](https://github.com/XgDuan/coco-caption/tree/python3). Thanks to port it to python3.
## Leaderboard ##
[MSR-VTT 2017 Leaderboard](http://ms-multimedia-challenge.com/2017/leaderboard)

|Rank|Team|Organization|BLEU@4|Meteor|CIDEr-D |ROUGE-L|
|-|-|-|-|-|-|-|
|1|RUC+CMU_V2T|RUC & CMU|**0.390**|**0.255**|**0.315**|**0.542**|
|2|TJU_Media|TJU|0.359|0.226|0.249|0.515|
|3|NII|National Institute of Informatics|0.359|0.234|0.231|0.514|
|**our implement**|**model**|**feature**|
|-|s2vt|vgg19|0.2864|0.2055|0.1748|0.4774|
|-|s2vt|resnet101|0.3118|0.2130|0.2002|0.4926|
|-|s2vt|nasnet|0.3003|0.2176|0.2213|0.4931|
|-|s2vt|nasnet+c3d_kinectics|0.3237|0.2227|0.2299|0.5041|
|-|s2vt|nasnet+c3d_kinectics+beam search|0.3349|0.2244|0.2340|0.5034|
|-|s2vt+att|nasnet+c3d_kinectics|0.3226|0.2235|0.2399|0.5011|
|-|s2vt+att|nasnet+c3d_kinectics+beam search|**0.3419**|**0.2278**|**0.2503**|**0.5063**|
## TODO
- lstm
- beam search(you can refer to [Sundrops/video-caption.pytorch](https://github.com/Sundrops/video-caption.pytorch) )
- reinforcement learning
- dataparallel (broken in pytorch 0.4)

## Note
You can see my another repository [video-caption-openNMT.py](https://github.com/xiadingZ/video-caption-openNMT.pytorch). It has higher performence and test score.

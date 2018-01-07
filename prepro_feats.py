import shutil
import subprocess
import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import argparse

import torch
from models.inception import inception_v3
from torch.autograd import Variable

from torchvision import transforms


def extract_frames(video, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            print(" cleanup: " + dst + "/")
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   '-y',  # (optional) overwrite output file if it exists
                                   '-i', video,  # input file
                                   '-vf', "scale=400:300",  # input file
                                   '-qscale:v', "2",  # quality for JPEG
                                   '{0}/%06d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command, stdout=ffmpeg_log, stderr=ffmpeg_log)


def extract_feats(params):
    C, H, W = 3, 299, 299
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        normalize,
    ])
    model = inception_v3(pretrained=True).cuda()

    dir_fc = params['output_dir']
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)

    video_list = glob.glob(os.path.join(params['video_path'], '*.mp4'))
    for video in tqdm(video_list):
        video_id = video.split("/")[-1].split(".")[0]
        dst = video_id
        extract_frames(video, dst)

        image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))
        samples = np.round(np.linspace(0, len(image_list) - 1, params['n_frame_step']))
        image_list = [image_list[int(sample)] for sample in samples]
        images = torch.zeros((len(image_list), C, H, W))
        for iImg in range(len(image_list)):
            img = Image.open(image_list[iImg])
            img = preprocess(img)
            images[iImg] = img
        with torch.no_grad():
            fc_feats, score, aux_logits = model(Variable(images).cuda())
            img_feats = fc_feats.data.cpu().numpy()
        # Save the inception features
        outfile = os.path.join(dir_fc, video_id + '_incep_v3.npy')
        np.save(outfile, img_feats)
        # cleanup
        shutil.rmtree(dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='data/feats/with_norm/', help='directory to store features')

    parser.add_argument("--video_path", dest='video_path', type=str,
                        default='data/train-video', help='path to video dataset')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = vars(args)
    params['n_frame_step'] = 26
    params['dim_image'] = 2048
    extract_feats(params)

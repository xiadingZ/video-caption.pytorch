import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
import numpy as np
import os
import json
import argparse
import opts
from dataloader import CocoDataset
import pretrainedmodels
from pretrainedmodels import utils


C, H, W = 3, 224, 224


def train(dataloader, model, crit, optimizer, lr_scheduler, load_image_fn, params):
    model.train()
    model = nn.DataParallel(model)
    images_path = json.load(open(params.coco_path))

    for epoch in range(params.epochs):
        lr_scheduler.step()
        iteration = 0
        for data in dataloader:
            iteration += 1
            image_ids, image_labels = data['image_ids'], data['labels']
            images = torch.zeros(image_labels.shape[0], C, H, W)
            for i, image_id in enumerate(image_ids):
                image_path = os.path.join(
                    params.coco_dir, images_path[image_id])
                images[i] = load_image_fn(image_path)
            logits = model(Variable(images).cuda())
            loss = crit(logits, Variable(image_labels).cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.data[0]
            torch.cuda.synchronize()

            print("iter %d (epoch %d), train_loss = %.6f" %
                  (iteration, epoch, train_loss))

        if epoch % params.save_checkpoint_every == 0:
            checkpoint_path = os.path.join(
                params.checkpoint_path, 'cnn_model_%d.pth' % (epoch))
            torch.save(model.state_dict(), checkpoint_path)
            print("model saved to %s" % (checkpoint_path))


def main(args):
    global C, H, W
    coco_labels = json.load(open(args.coco_labels))
    num_classes = coco_labels['num_classes']
    if args.model == 'inception_v3':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv3(pretrained='imagenet')

    elif args.model == 'resnet152':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet152(pretrained='imagenet')

    elif args.model == 'inception_v4':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(
            num_classes=1000, pretrained='imagenet')

    else:
        print("doesn't support %s" % (args['model']))

    load_image_fn = utils.LoadTransformImage(model)
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    model = model.cuda()
    dataset = CocoDataset(coco_labels)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.learning_rate_decay_every,
                                                 gamma=args.learning_rate_decay_rate)

    crit = nn.BCEWithLogitsLoss()
    if not os.path.isdir(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)
    train(dataloader, model, crit, optimizer,
          exp_lr_scheduler, load_image_fn, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_path', type=str,
                        default='data/coco_path.json', help='')
    parser.add_argument('--coco_labels', type=str,
                        default='data/coco_labels.json', help='path to processed coco caption json')
    parser.add_argument('--coco_dir', type=str,
                        default='data/mscoco/train2014')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='number of epochs')
    parser.add_argument('--checkpoint_path', type=str,
                        help='path to trained model')
    parser.add_argument("--gpu", dest='gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument("--model", dest="model", type=str, default='inception_v4',
                        help='the CNN model you want to use to extract_feats')
    parser.add_argument('--dim_image', dest='dim_vid', type=int, default=1536,
                        help='dim of frames images extracted by cnn model')

    parser.add_argument('--save_checkpoint_every', type=int, default=10,
                        help='how often to save a model checkpoint (in epoch)?')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                        help='learning rate')

    parser.add_argument('--learning_rate_decay_every', type=int, default=200,
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                        help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                        help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                        help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight_decay')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)

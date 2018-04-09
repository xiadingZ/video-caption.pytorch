import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CocoDataset(Dataset):

    def __init__(self, coco_labels):
        super(CocoDataset, self).__init__()
        self.coco_labels = list(coco_labels['labels'].items())
        self.num_classes = coco_labels['num_classes']

    def __getitem__(self, ix):
        labels = torch.zeros(self.num_classes)
        image_id, labels_ids = self.coco_labels[ix]
        labels[labels_ids] = 1
        data = {}
        data['image_ids'] = image_id
        data['labels'] = labels
        return data

    def __len__(self):
        return len(self.coco_labels)


class VideoDataset(Dataset):

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt, mode):
        super(VideoDataset, self).__init__()
        self.mode = mode  # to load train/val/test data

        # load the json file which contains information about the dataset
        self.captions = json.load(open(opt["caption_json"]))
        info = json.load(open(opt["info_json"]))
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        print('vocab size is ', len(self.ix_to_word))
        self.splits = info['videos']
        print('number of train videos: ', len(self.splits['train']))
        print('number of val videos: ', len(self.splits['val']))
        print('number of test videos: ', len(self.splits['test']))

        self.feats_dir = opt["feats_dir"]
        self.c3d_feats_dir = opt['c3d_feats_dir']
        self.with_c3d = opt['with_c3d']
        print('load feats from %s' % (self.feats_dir))
        # load in the sequence data
        self.max_len = opt["max_len"]
        print('max sequence length in data is', self.max_len)

    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        # which part of data to load
        if self.mode == 'val':
            ix += len(self.splits['train'])
        elif self.mode == 'test':
            ix = ix + len(self.splits['train']) + len(self.splits['val'])
        
        fc_feat = []
        for dir in self.feats_dir:
            fc_feat.append(np.load(os.path.join(dir, 'video%i.npy' % (ix))))
        fc_feat = np.concatenate(fc_feat, axis=1)
        if self.with_c3d == 1:
            c3d_feat = np.load(os.path.join(self.c3d_feats_dir, 'video%i.npy'%(ix)))
            fc_feat = np.concatenate((fc_feat, np.tile(c3d_feat, (fc_feat.shape[0], 1))), axis=1)
        label = torch.zeros(self.max_len)
        mask = torch.zeros(self.max_len)
        captions = self.captions['video%i'%(ix)]['final_captions']
        gts = torch.zeros(len(captions), self.max_len).long()
        for i, cap in enumerate(captions):
            if len(cap) > self.max_len:
                cap = cap[:self.max_len]
                cap[-1] = '<eos>'
            for j, w in enumerate(cap):
                gts[i, j] = self.word_to_ix[w]

        # random select a caption for this video
        cap_ix = random.randint(0, len(captions) - 1)
        label = gts[cap_ix]
        non_zero = (label == 0).nonzero()
        mask[:int(non_zero[0]) + 1] = 1

        data = {}
        data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
        data['labels'] = label
        data['masks'] = mask
        data['gts'] = gts
        data['video_ids'] = 'video%i'%(ix)
        return data

    def __len__(self):
        return len(self.splits[self.mode])

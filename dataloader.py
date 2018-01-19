import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(self, coco_labels):
        super().__init__()
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
        super().__init__()
        self.mode = mode  # to load train/val/test data

        # load the json file which contains information about the dataset
        self.captions = json.load(open(opt.caption_json))
        info = json.load(open(opt.info_json))
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        print('vocab size is ', len(self.ix_to_word))
        self.splits = info['videos']
        print('number of train videos: ', len(self.splits['train']))
        print('number of val videos: ', len(self.splits['val']))
        print('number of test videos: ', len(self.splits['test']))

        self.feats_dir = opt.feats_dir
        print('load feats from %s' % (self.feats_dir))
        # load in the sequence data
        self.max_len = opt.max_len
        print('max sequence length in data is', self.max_len)

    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        # which part of data to load
        if self.mode == 'val':
            ix += len(self.splits['train'])
        elif self.mode == 'test':
            ix = ix + len(self.splits['train']) + len(self.splits['val'])

        fc_feat = np.load(os.path.join(self.feats_dir,
                                       'video' + str(ix) + '.npy'))

        label = torch.zeros(self.max_len)
        mask = torch.zeros(self.max_len)
        captions = self.captions['video' + str(ix)]['final_captions']
        # random select a caption for this video
        cap_ix = random.randint(0, len(captions) - 1)
        cap = captions[cap_ix]
        if len(cap) > self.max_len:
            cap = cap[:self.max_len]
            cap[-1] = '<eos>'

        for i, w in enumerate(cap):
            label[i] = self.word_to_ix[w]
            mask[i] = 1

        data = {}
        data['fc_feats'] = torch.from_numpy(fc_feat)
        data['labels'] = label
        data['masks'] = mask
        data['video_ids'] = 'video' + str(ix)
        return data

    def __len__(self):
        return len(self.splits[self.mode])

import json
import h5py
import os
import numpy as np
import random

import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.mode = opt.mode  # to load train/val/test data

        # load the json file which contains information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.info_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)
        self.splits = self.info['videos']
        print('number of train videos: ', len(self.splits['train']))
        print('number of val videos: ', len(self.splits['val']))
        print('number of test videos: ', len(self.splits['test']))
        # open the hdf5 file
        print('DataLoader loading video features: ', opt.feats_dir)
        print('DataLoader loading h5 file: ', opt.input_label_h5)
        self.h5_label_file = h5py.File(
            self.opt.input_label_h5, 'r', driver='core')

        self.feats_dir = self.opt.feats_dir

        # load in the sequence data
        self.seq_length = self.h5_label_file['labels'].shape[1]
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

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

        label = np.zeros([self.seq_length], dtype='int')
        mask = np.zeros([self.seq_length], dtype='float32')

        # fetch the sequence labels
        ix1 = self.label_start_ix[ix]
        ix2 = self.label_end_ix[ix]
        # random select a caption for this video
        ixl = random.randint(ix1, ix2)
        label = self.h5_label_file['labels'][ixl]

        nonzero_ixs = np.nonzero(label)[0]
        mask[:nonzero_ixs.max() + 2] = 1

        # Used for reward evaluation
        gts = self.h5_label_file['labels'][self.label_start_ix[ix]: self.label_end_ix[ix] + 1]

        # generate mask

        data = {}
        data['fc_feats'] = torch.from_numpy(fc_feat)
        data['labels'] = torch.from_numpy(label)
        data['gts'] = torch.from_numpy(gts)
        data['masks'] = torch.from_numpy(mask)
        data['ix'] = ix
        return data

    def __len__(self):
        return len(self.splits[self.mode])

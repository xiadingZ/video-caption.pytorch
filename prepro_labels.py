import re
import json
import h5py
import argparse
import numpy as np


def build_vocab(vids, params):
    count_thr = params['word_count_threshold']
    # count up the number of words
    counts = {}
    for vid, caps in vids.items():
        for cap in caps['captions']:
            ws = re.sub(r'[.!,;?]', ' ', cap).split()
            for w in ws:
                counts[w] = counts.get(w, 0) + 1
    # cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    total_words = sum(counts.values())
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))
    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('<UNK>')
    for vid, caps in vids.items():
        caps = caps['captions']
        vids[vid]['final_captions'] = []
        for cap in caps:
            ws = re.sub(r'[.!,;?]', ' ', cap).split()
            caption = ['<sos>'] + [w if counts.get(w, 0) > count_thr else '<UNK>' for w in ws] + ['<eos>']
            vids[vid]['final_captions'].append(caption)
    return vocab


def encode_captions(vids, params, wtoi):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    max_length = params['max_length']
    N = len(vids)
    video_ids = list(vids.keys())
    M = sum(len(cap['final_captions']) for cap in vids.values())  # total number of captions
    label_arrays = []
    label_start_ix = np.zeros(N, dtype='int32')
    label_end_ix = np.zeros(N, dtype='int32')
    label_length = np.zeros(M, dtype='int32')
    caption_counter = 0
    counter = 0
    for i in range(N):
        cap = vids[video_ids[i]]
        n = len(cap['final_captions'])

        Li = np.zeros((n, max_length), dtype='int32')
        for j, s in enumerate(cap['final_captions']):
            label_length[caption_counter] = min(max_length, len(s))  # record the length of this sequence
            caption_counter += 1
            for k, w in enumerate(s):
                # ensure last symbol is <eos> (0)
                if k == (max_length - 1):
                    Li[j, k] = wtoi['<eos>']
                    break
                else:
                    Li[j, k] = wtoi[w]

        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(label_arrays, axis=0)  # put all the labels together

    print('encoded captions to array of size ', L.shape)
    return L, label_start_ix, label_end_ix, label_length


def main(params):
    videos = json.load(open(params['input_json'], 'r'))['sentences']
    video_caption = {}
    for i in videos:
        if i['video_id'] not in video_caption.keys():
            video_caption[i['video_id']] = {'captions': []}
        video_caption[i['video_id']]['captions'].append(i['caption'])
    # create the vocab
    vocab = build_vocab(video_caption, params)
    itow = {i + 2: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 2 for i, w in enumerate(vocab)}  # inverse table
    wtoi['<eos>'] = 0
    itow[0] = '<eos>'
    wtoi['<sos>'] = 1
    itow[1] = '<sos>'

    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_start_ix, label_end_ix, label_length = encode_captions(video_caption, params, wtoi)
    # create output h5 file
    f_lb = h5py.File(params['output_h5'] + '_label.h5', "w")
    f_lb.create_dataset("labels", dtype='int32', data=L)
    f_lb.create_dataset("label_start_ix", dtype='int32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='int32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='int32', data=label_length)
    f_lb.close()

    out = {}
    out['ix_to_word'] = itow
    out['word_to_ix'] = wtoi
    out['videos'] = {'train': [], 'val': [], 'test': []}
    videos = json.load(open(params['input_json'], 'r'))['videos']
    for i in videos:
        out['videos'][i['split']].append(int(i['id']))
    json.dump(out, open(params['info_json'], 'w'))
    json.dump(video_caption, open(params['caption_json'], 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--info_json', default='data/info.json', help='info about iw2word and word2ix')
    parser.add_argument('--caption_json', default='data/caption.json', help='caption json file')
    parser.add_argument('--output_h5', default='data/data', help='output h5 file')

    # options
    parser.add_argument('--max_length', default=28, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=1, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)

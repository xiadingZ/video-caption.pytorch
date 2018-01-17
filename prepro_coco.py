import json
import argparse
from pandas.io.json import json_normalize
from nltk.corpus import stopwords
from collections import Counter
from tqdm import tqdm


def main(args):
    coco = json.load(
        open(args.coco_json))['annotations']
    msr = json.load(open(args.msr_caption_json))
    stopWords = set(stopwords.words('english'))
    coco_wordcounts = open(args.coco_wordcounts)
    coco_words = []
    for i in coco_wordcounts:
        w = i.split()[0]
        coco_words.append(w)
    msr_wordcounts = []
    for i in msr.values():
        for j in i['final_captions']:
            msr_wordcounts += j
    msr_wordcounts = Counter(msr_wordcounts).most_common()
    labels = [i for i in msr_wordcounts if i[0]
              not in stopWords and i[0] in coco_words][:args.num_classes]
    for i in tqdm(coco):
        l = []
        for j, w in enumerate(labels):
            if w[0] in i['caption']:
                l.append(j)
        i['labels'] = l
    coco_labels = {}
    for i in tqdm(coco):
        if i['image_id'] in coco_labels:
            coco_labels[i['image_id']] = coco_labels[i['image_id']] + \
                list(set(i['labels']) - set(coco_labels[i['image_id']]))
        else:
            coco_labels[i['image_id']] = i['labels']
    info = {'num_classes': args.num_classes, 'labels': coco_labels}
    with open(args.coco_labels_json, 'w') as f:
        json.dump(info, f)

    coco = json.load(
        open(args.coco_json))['images']
    coco_path = {}
    for i in tqdm(coco):
        coco_path[i['id']] = i['file_name']
    with open(args.coco_path_json, 'w') as f:
        json.dump(coco_path, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_json', type=str,
                        default='data/mscoco/annotations/captions_train2014.json', help='path to coco train json')
    parser.add_argument('-coco_wordcounts', type=str,
                        default='data/mscoco/word_counts.txt', help='word_counts.txt of coco dataset')
    parser.add_argument('--msr_caption_json', type=str,
                        default='data/caption.json', help='path to processed msr vtt caption json')
    parser.add_argument('--num_classes', type=int, default=3000,
                        help='number of classes each image')
    parser.add_argument('--coco_labels_json', type=str, default='data/coco_labels.json',
                        help='path to processed coco train caption json')
    parser.add_argument('--coco_path_json', type=str, default='data/coco_path.json',
                        help='image id and image file name pairs')
    args = parser.parse_args()
    main(args)

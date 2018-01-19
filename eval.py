import json
import os

import opts
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from dataloader import VideoDataset
import misc.utils as utils
from misc.cocoeval import suppress_stdout_stderr, COCOScorer

from pandas.io.json import json_normalize


def convert_data_to_coco_scorer_format(data_frame):
    gts = {}
    for row in zip(data_frame["caption"], data_frame["video_id"]):
        if row[1] in gts:
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
        else:
            gts[row[1]] = []
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
    return gts


def test(model, crit, dataset, vocab, opt):
    model.eval()
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    scorer = COCOScorer()
    gt_dataframe = json_normalize(json.load(open(opt.input_json))['sentences'])
    gts = convert_data_to_coco_scorer_format(gt_dataframe)
    results = []
    samples = {}
    for data in loader:
        # forward the model to get loss
        fc_feats = Variable(data['fc_feats']).cuda()
        labels = Variable(data['labels']).long().cuda()
        with torch.no_grad():
            # forward the model to also get generated samples for each image
            seq_probs, seq_preds = model(
                fc_feats, labels, teacher_forcing_ratio=0)

        sents = utils.decode_sequence(vocab, seq_preds)

        for k, sent in enumerate(sents):
            video_id = 'video' + str(data['ix'][k])
            samples[video_id] = [{'image_id': video_id, 'caption': sent}]

    with suppress_stdout_stderr():
        valid_score = scorer.score(gts, samples, samples.keys())
    results.append(valid_score)
    print(valid_score)

    if not os.path.exists(opt.results_path):
        os.makedirs(opt.results_path)

    with open(os.path.join(opt.results_path, "scores.txt"), 'a') as scores_table:
        scores_table.write(json.dumps(results[0]) + "\n")
    with open(os.path.join(opt.results_path, opt.model.split("/")[-1].split('.')[0] + ".json"), 'w') as prediction_results:
        json.dump({"predictions": samples, "scores": valid_score},
                  prediction_results)


def main(opt):
    dataset = VideoDataset(opt, opt.mode)
    opt.vocab_size = dataset.get_vocab_size()
    opt.seq_length = dataset.seq_length
    if opt.model == 'S2VTModel':
        model = S2VTModel(opt.vocab_size, opt.seq_length, opt.dim_hidden, opt.dim_word,
                          rnn_dropout_p=opt.rnn_dropout_p).cuda()
    elif opt.model == "S2VTAttModel":
        encoder = EncoderRNN(opt.dim_vid, opt.dim_hidden)
        decoder = DecoderRNN(opt.vocab_size, opt.seq_length, opt.dim_hidden, opt.dim_word,
                             rnn_dropout_p=opt.rnn_dropout_p)
        model = S2VTAttModel(encoder, decoder).cuda()
    model = nn.DataParallel(model)
    # Setup the model
    model.load_state_dict(torch.load(opt.saved_model))
    crit = utils.LanguageModelCriterion()

    test(model, crit, dataset, dataset.get_vocab(), opt)


if __name__ == '__main__':
    opt = opts.parse_opt()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    main(opt)

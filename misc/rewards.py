import numpy as np
from collections import OrderedDict
import torch
import sys
sys.path.append("coco-caption")
from pyciderevalcap.ciderD.ciderD import CiderD

CiderD_scorer = None
# CiderD_scorer = CiderD(df='corpus')


def init_cider_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


def get_self_critical_reward(model, fc_feats, data, gen_result):
    batch_size = gen_result.size(0)

    # get greedy decoding baseline
    _, greedy_res = model(fc_feats, mode='inference')

    res = OrderedDict()

    gen_result = gen_result.cpu().data.numpy()
    greedy_res = greedy_res.cpu().data.numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(data['gts'].size(0)):
        gts[i] = [array_to_str(data['gts'][i][j])
                  for j in range(data['gts'].size(1))]

    res = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size] for i in range(2 * batch_size)}
    _, scores = CiderD_scorer.compute_score(gts, res)
    print('Cider scores:', _)

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards

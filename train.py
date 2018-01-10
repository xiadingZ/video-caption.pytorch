import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
import numpy as np
import os

import opts
from models import EncoderRNN, DecoderRNN, Vid2seq, S2VTModel
from dataloader import VideoDataset
import misc.utils as utils
from misc.rewards import init_cider_scorer, get_self_critical_reward


def train(dataloader, model, crit, optimizer, lr_scheduler, opt, rl_crit=None):
    model.train()
    model = nn.DataParallel(model)

    for epoch in range(opt.epochs):
        lr_scheduler.step()

        iteration = 0
        # If start self crit training
        if opt.self_crit_after != -1 and epoch >= opt.self_crit_after:
            sc_flag = True
            init_cider_scorer(opt.cached_tokens)
        else:
            sc_flag = False

        for data in dataloader:
            torch.cuda.synchronize()
            fc_feats = Variable(data['fc_feats']).cuda()
            labels = Variable(data['labels']).long().cuda()
            masks = Variable(data['masks']).cuda()
            optimizer.zero_grad()
            if not sc_flag:
                seq_probs, predicts = model(fc_feats, labels)
                loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
            else:
                gen_result, sample_logprobs = model.sample(fc_feats, vars(opt))
                # print(gen_result)
                reward = get_self_critical_reward(model, fc_feats, data, gen_result)
                loss = rl_crit(sample_logprobs, gen_result, Variable(
                    torch.from_numpy(reward).float().cuda(), requires_grad=False)))
            loss.backward()
            utils.clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            train_loss = loss.data[0]
            torch.cuda.synchronize()
            iteration += 1
            
            if not sc_flag:
                print("iter %d (epoch %d), train_loss = %.6f" % (iteration, epoch, train_loss))
            else:
                print("iter %d (epoch %d), avg_reward = %.3f" % (iteration, epoch,
                                                                 np.mean(reward[:, 0])))

        best_flag = False
        if (epoch % opt.save_checkpoint_every == 0):
            checkpoint_path = os.path.join(opt.checkpoint_path, 'model_%d.pth' % (epoch))
            torch.save(model.state_dict(), checkpoint_path)
            print("model saved to %s" % (checkpoint_path))

            if best_flag:
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                
                 # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break


def main(opt):
    dataset = VideoDataset(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    opt.vocab_size = dataset.vocab_size
    opt.seq_length = dataset.seq_length
    if opt.model == 'S2VTModel':
        model = S2VTModel(opt.vocab_size, opt.seq_length, opt.dim_hidden, opt.dim_word,
                          rnn_dropout_p=opt.rnn_dropout_p).cuda()
    elif opt.model == "Vid2seq":
        encoder = EncoderRNN(opt.dim_vid, opt.dim_hidden)
        decoder = DecoderRNN(opt.vocab_size, opt.seq_length, opt.dim_hidden, use_attention=True, dropout_p=0.2)
        model = Vid2seq(encoder, decoder).cuda()
    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.learning_rate_decay_every,
                                                 gamma=opt.learning_rate_decay_rate)
    if not os.path.isdir(opt.checkpoint_path):
            os.mkdir(opt.checkpoint_path)
    train(dataloader, model, crit, optimizer, exp_lr_scheduler, opt, rl_crit)

    
    # Stop if reaching max epochs
    if epoch >= opt.max_epochs and opt.max_epochs != -1:
        break

if __name__ == '__main__':
    opt = opts.parse_opt()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    main(opt)

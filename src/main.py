import math
import time

import mxnet as mx
from mxnet import gluon, autograd

from Const import *
from Util import Util
from model.Corpus import Corpus
from model.RNNModel import RNNModel

context = mx.cpu(0)
corpus = Corpus(args_data)

train_data = Util.batchify(corpus.train, args_batch_size).as_in_context(context)
val_data = Util.batchify(corpus.valid, args_batch_size).as_in_context(context)
test_data = Util.batchify(corpus.test, args_batch_size).as_in_context(context)

ntokens = len(corpus.dictionary)

model = RNNModel(args_model, ntokens, args_emsize, args_nhid,
                 args_nlayers, args_dropout, args_tied)
model.collect_params().initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': args_lr, 'momentum': 0, 'wd': 0})
loss = gluon.loss.SoftmaxCrossEntropyLoss()


def eval(data_source):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=args_batch_size, ctx=context)
    for i in range(0, data_source.shape[0] - 1, args_bptt):
        data, target = Util.get_batch(data_source, i)
        output, hidden = model(data, hidden)
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal


def train():
    best_val = float("Inf")
    for epoch in range(args_epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=args_batch_size, ctx=context)
        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, args_bptt)):
            data, target = Util.get_batch(train_data, i)
            hidden = Util.detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and bptt size to balance it.
            gluon.utils.clip_global_norm(grads, args_clip * args_bptt * args_batch_size)

            trainer.step(args_batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if ibatch % args_log_interval == 0 and ibatch > 0:
                cur_L = total_L / args_bptt / args_batch_size / args_log_interval
                print('[Epoch %d Batch %d] loss %.2f, perplexity %.2f' % (
                    epoch + 1, ibatch, cur_L, math.exp(cur_L)))
                total_L = 0.0

        val_L = eval(val_data)

        print('[Epoch %d] time cost %.2fs, validation loss %.2f, validation perplexity %.2f' % (
            epoch + 1, time.time() - start_time, val_L, math.exp(val_L)))

        if val_L < best_val:
            best_val = val_L
            test_L = eval(test_data)
            model.save_params(args_save)
            print('test loss %.2f, test perplexity %.2f' % (test_L, math.exp(test_L)))
        else:
            args_lr = args_lr * 0.25
            trainer._init_optimizer('sgd',
                                    {'learning_rate': args_lr,
                                     'momentum': 0,
                                     'wd': 0})
            model.load_params(args_save, context)


def main():
    train()
    model.load_params(args_save, context)
    test_L = eval(test_data)
    print('Best test loss %.2f, test perplexity %.2f' % (test_L, math.exp(test_L)))


if __name__ == '__main__':
    main()

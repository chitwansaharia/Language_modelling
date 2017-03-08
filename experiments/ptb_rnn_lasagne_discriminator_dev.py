from __future__ import absolute_import
import theano
import theano.tensor as T
import sys
import argparse
import numpy as np
import logging
import os
import os, sys
import random
sys.path.insert(0, os.getcwd())
import lasagne
from lasagne.nonlinearities import (sigmoid,linear, )
from lasagne.layers import (ReshapeLayer, DropoutLayer,DenseLayer,
                            ConcatLayer, ElemwiseSumLayer, GaussianNoiseLayer,LSTMLayer, )
import cPickle

theano.config.floatX = 'float32'

parser = argparse.ArgumentParser()
parser.add_argument(
    "-train",
    "--train_sentences",
    help="path to ptb train",
    required=True
)
parser.add_argument(
    "-dev",
    "--dev_sentences",
    help="path to ptb dev",
    required=True
)
parser.add_argument(
    "-c",
    "--check_point",
    help="params save",
    required=False
)

parser.add_argument(
    "-test",
    "--test_sentences",
    help="path to ptb test",
    required=False
)
parser.add_argument(
    "-batch_size",
    "--batch_size",
    help="batch size",
    required=True
)

parser.add_argument(
    "-o",
    "--hidden_dim",
    help="hidden dimension",
    required=True
)
parser.add_argument(
    "-e",
    "--emb_dim",
    help="embedding dimension",
    required=True
)
parser.add_argument(
    "-d",
    "--depth",
    help="num rnn layers",
    required=True
)
parser.add_argument(
    "-exp",
    "--experiment_name",
    help="name of the experiment",
    required=True
)
parser.add_argument(
    "-seed",
    "--seed",
    help="seed for pseudo random number generator",
    default=1337
)
parser.add_argument(
    "-drp"
    "--dropout_rate",
    help="dropout rate",
    default=0.0
)

args = parser.parse_args()
data_path_train = args.train_sentences
data_path_dev = args.dev_sentences
data_path_test = args.test_sentences
np.random.seed(seed=int(args.seed))  # set seed for an experiment
experiment_name = args.experiment_name
batch_size = int(args.batch_size)
hidden_dim = int(args.hidden_dim)
embedding_dim = int(args.emb_dim)
checkpoint=args.check_point
depth = int(args.depth)

if not os.path.exists('log/'):
    os.mkdir('log/')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/%s' % (experiment_name),
    filemode='w'
)

# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


def generate_samples(
    batch_input
):
    """Generate random samples."""
    decoded_batch = f_eval(
        batch_input
    )
    decoded_batch = np.argmax(decoded_batch, axis=2)
    for ind, sentence in enumerate(decoded_batch[:10]):
        logging.info('Src : %s ' % (' '.join([
            ind2word[x] if x != word2ind['<pad>'] else '' for x in batch_input[ind]]
        )))
        logging.info('Sample : %s ' % (' '.join([
            ind2word[x] if x != word2ind['<pad>'] else '' for x in decoded_batch[ind]]
        )))
        logging.info('=======================================================')


def get_perplexity(dataset='train'):
    """Compute perplexity on train/dev/test."""
    if dataset == 'dev':
        dataset = dev_lines
    elif dataset == 'test':
        dataset = test_lines

    perplexities = []

    for j in xrange(0, len(dataset), batch_size):
        inp, op, mask = get_minibatch(
            dataset,
            j,
            batch_size
        )

        decoded_batch_ce = f_ce(
            inp,
            op,
            mask
        )
        perplexities.append(decoded_batch_ce)

    return np.exp(np.mean(perplexities))


def get_minibatch(lines, index, batch_size):
    """Prepare minibatch."""
    lines = [
        line[:40] + ['</s>']
        for line in lines[index: index + (batch_size/2)]
    ]

    lens = [len(line) for line in lines]
    max_len = max(lens)
    input_lines=[]
    target_lines=[]
    for line in lines:
         actual=[word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[:-1]]
         input_lines.append(actual+[word2ind['<pad>']] * (max_len - len(line)))
         random.shuffle(actual)

         target_lines.append(1.0)
         input_lines.append(actual+[word2ind['<pad>']] * (max_len - len(line)))
         target_lines.append(0.0)
    input_lines = np.array(input_lines).astype(np.int32)
    output_score = np.array(target_lines).astype(np.float32)



    mask = np.array(
        [
            ([1] * (l - 1)) + ([0] * (max_len - l))
            for l in lens
        ]
    ).astype(np.float32)
    return input_lines, output_score, mask




train_lines = [line.strip().split() for line in open(data_path_train, 'r')]
dev_lines = [line.strip().split() for line in open(data_path_dev, 'r')]

word2ind = {'<s>': 0, '</s>': 1, '<pad>': 2}
ind2word = {0: '<s>', 1: '</s>', 2: '<pad>'}
ind = 3

for line in train_lines:
    for word in line:
        if word not in word2ind:
            word2ind[word] = ind
            ind2word[ind] = word
            ind += 1

logging.info('Found %d words in vocabulary' % (len(word2ind)))

inp_t = np.random.randint(low=1, high=100, size=(5, 10)).astype(np.int32)
op_t = np.random.rand(5).astype(np.float32)
mask_t = np.float32(np.random.rand(5, 10).astype(np.float32) > 0.5)

x = T.imatrix()
y = T.ivector()
mask = T.fmatrix('Mask')

network = lasagne.layers.InputLayer(shape=(None,None),input_var=x)
network = lasagne.layers.EmbeddingLayer(network,input_size=len(word2ind), output_size=embedding_dim)

for i in xrange(depth-1 ):
    network=LSTMLayer(network, hidden_dim,nonlinearity=lasagne.nonlinearities.tanh)
network=LSTMLayer(network, hidden_dim,nonlinearity=lasagne.nonlinearities.tanh,only_return_final=True)
network=lasagne.layers.DenseLayer(network,num_units=1,nonlinearity=lasagne.nonlinearities.sigmoid)

params = lasagne.layers.get_all_params(network, trainable=True)
logging.info('Model parameters ...')
logging.info('Embedding dim : %d ' % (embedding_dim))
logging.info('RNN Hidden Dim : %d ' % (hidden_dim))
logging.info('Batch size : %s ' % (batch_size))
logging.info('Depth : %s ' % (depth))


final_output=lasagne.layers.get_output(network)
final_output_test=lasagne.layers.get_output(network,deterministic=True)

final_output = T.clip(final_output, 1e-5, 1 - 1e-5)

# Compute cost
def calc_cross_ent(net_output, targets):
    preds = net_output
    targets = T.flatten(targets)
    cost = lasagne.objectives.binary_crossentropy(preds, targets).mean()
    return cost
cost = T.mean(calc_cross_ent(final_output, y))



logging.info('Computation Graph Node Shapes ...')
logging.info('final_output : %s' % (final_output.eval(
    {x: inp_t}).shape,)
)



if checkpoint is not None:
        logger.info('Starting from checkpoint')
        with open(checkpoint, 'rb') as f:
            layer_params=pickle.load(f)
        layer_params=layer_params
        lasagne.layers.set_all_param_values(network,layer_params, trainable=True)

logging.info('Compiling updates ...')
updates = lasagne.updates.rmsprop(cost, params, learning_rate=0.01)

logging.info('Compiling train function ...')
f_train = theano.function(
    inputs=[x, y, mask],
    outputs=cost,
    updates=updates,on_unused_input='warn',allow_input_downcast=True,
)

logging.info('Compiling eval function ...')
f_eval = theano.function(
    inputs=[x],
    outputs=final_output,on_unused_input='warn',allow_input_downcast=True,
)

logging.info('Compiling cross entropy function ...')
f_ce = theano.function(
    inputs=[x, y, mask],
    outputs=cost,on_unused_input='warn',allow_input_downcast=True
)
train_acc = T.mean(T.eq(final_output, y),
                       dtype=theano.config.floatX)
test_acc = T.mean(T.eq(final_output_test, y),
                      dtype=theano.config.floatX)
train_acc_fn = theano.function([[x, y], train_acc)
val_fn = theano.function([[x, y], test_acc)


num_epochs = 2
logging.info('Training network ...')
for i in range(num_epochs):
    costs = []
    np.random.shuffle(train_lines)
    for j in xrange(0, len(train_lines), batch_size):
        inp, op, mask = get_minibatch(
            train_lines,
            j,
            batch_size
        )
        entropy = f_train(
            inp,
            op,
            mask
        )
        costs.append(entropy)
        logging.info('Epoch : %d Minibatch : %d Loss : %.3f Train_acc : %.3f Test_acc : %.3f' % (
            i,
            j,
            entropy,
            train_acc1,
            test_acc1
        ))





    print 'Epoch Loss : %.3f ' % (np.mean(costs))
    train_acc1=train_acc_fn (inp,op)
    test_acc1=val_fn(inp,op)
    logging.info('Epoch : %d   Train_acc : %.3f Test_acc : %.3f' % (
        i,
        train_acc1,
        test_acc1
    ))
_filepath = os.path.join(os.getcwd(),
                                 'check_pointd.pkl')

layer_params=lasagne.layers.get_all_param_values(network, trainable=True)
with open(_filepath, 'wb') as f:
    cPickle.dump(layer_params, f, protocol=cPickle.HIGHEST_PROTOCOL)

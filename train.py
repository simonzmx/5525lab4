import json
import os
import pprint
# import time

import numpy as np
import pandas as pd
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable warning, doesn't enable AVX/FMA
tf.logging.set_verbosity(tf.logging.INFO)
pp = pprint.PrettyPrinter()

flags = tf.flags
# flags.DEFINE_string("model", "LSTM", "choose a model (default: LSTM)")
flags.DEFINE_string("word_vecs_path", "models/wv/word_vectors.npy",
                    "word vectors path (default: models/wv/word_vectors.npy)")

flags.DEFINE_string("x_train", "models/wv/x_train.npy", "docs path (default: models/wv/x_train.npy)")
flags.DEFINE_string("x_test", "models/wv/x_test.npy", "docs path (default: models/wv/x_test.npy)")
flags.DEFINE_string("x_dev", "models/wv/x_dev.npy", "docs path (default: models/wv/x_dev.npy)")

flags.DEFINE_string("y_train", "data/train_labels.csv", "labels path (default: data/train_labels.csv)")
flags.DEFINE_string("y_dev", "data/dev_labels.csv", "labels path (default: data/dev_labels.csv)")
flags.DEFINE_string("y_test", "data/test_labels.csv", "labels path (default: data/test_labels.csv)")

flags.DEFINE_string("model_dir", "models/test", "model path (default: models/test)")  # todo: use timestamp
FLAGS = flags.FLAGS


####################
# Hyper parameters #
####################
PARAMS = {
    'max_length': 256,  # max sentence length
    'hidden_size': 128,  # LSTM output dimension
    'num_epochs': 1,
    'num_layers': 2,
    'lr': 1.1,
    'batch_size': 64,
    'n_class': 5,
    'pool_size': (3,),
    'strides': (3,),
}

label_column = ""
label_to_idx = {}

if PARAMS['n_class'] == 2:
    label_column = 'raw'
    label_to_idx = {
        "negative": 0,
        "positive": 1
    }
elif PARAMS['n_class'] == 5:
    label_column = 'fine_grained'
    label_to_idx = {
        "very negative": 0,
        "negative": 1,
        "neutral": 2,
        "positive": 3,
        "very positive": 4
    }
else:
    raise ValueError('Wrong n_class, can only be 2 or 5.')


def train_input_fn(x, y, batch_size, padding_values_x, label_size, num_epochs, max_length):
    # todo: 1. add unknown words to word vectors
    # todo: 2. truncate or remove sentences exceeding max-length
    # todo: e.g. dataset = dataset.filter(lambda x, y: _filter_max_length((x, y), max_length))

    # truncate
    x = [i[:max_length] for i in x]

    dataset_x = tf.data.Dataset.from_generator(lambda: x, tf.int32)
    dataset_y = tf.data.Dataset.from_generator(lambda: y, tf.int32)

    # "dynamic padding": each batch might have different sequence lengths
    # but the sequence lengths within one batch are the same
    # the data type of padding_values should match that of the dataset! both tf.int32 in this case
    dataset_x = dataset_x.padded_batch(
        batch_size=batch_size, padded_shapes=[None], padding_values=padding_values_x)
    dataset_y = dataset_y.batch(batch_size=batch_size)

    dataset = tf.data.Dataset.zip((dataset_x, dataset_y))
    del dataset_x, dataset_y

    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=len(y), count=num_epochs, seed=61))

    dataset = dataset.map(lambda x1, y1: (x1, tf.reduce_sum(tf.one_hot(y1, label_size), axis=1)))

    return dataset


def eval_input_fn(x, y, batch_size, padding_values_x, label_size, max_length):
    # todo: 1. add unknown words to word vectors
    # todo: 2. truncate or remove sentences exceeding max-length
    # todo: e.g. dataset = dataset.filter(lambda x, y: _filter_max_length((x, y), max_length))

    # truncate
    x = [i[:max_length] for i in x]

    dataset_x = tf.data.Dataset.from_generator(lambda: x, tf.int32)
    dataset_y = tf.data.Dataset.from_generator(lambda: y, tf.int32)

    # "dynamic padding": each batch might have different sequence lengths
    # but the sequence lengths within one batch are the same
    # the data type of padding_values should match that of the dataset! both tf.int32 in this case
    dataset_x = dataset_x.padded_batch(
        batch_size=batch_size, padded_shapes=[None], padding_values=padding_values_x)
    dataset_y = dataset_y.batch(batch_size=batch_size)

    dataset = tf.data.Dataset.zip((dataset_x, dataset_y))
    del dataset_x, dataset_y

    dataset = dataset.map(lambda x1, y1: (x1, tf.reduce_sum(tf.one_hot(y1, label_size), axis=1)))

    return dataset


# def train_input_fn_v2(x, y, batch_size, padding_values_x, label_size, num_epochs, max_length=None):
#     def gen():
#         for doc, label in zip(x, y):
#             yield (doc, label)
#
#     dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32))
#
#     dataset = dataset.padded_batch(
#         batch_size=batch_size, padded_shapes=([max_length], [3]), padding_values=(padding_values_x, -1))
#
#     dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=len(y), count=num_epochs, seed=61))
#     dataset = dataset.map(lambda x1, y1: (x1, tf.reduce_sum(tf.one_hot(y1, label_size), axis=1)),
#                           num_parallel_calls=100*batch_size)
#     dataset = dataset.prefetch(buffer_size=100*batch_size)
#     return dataset


def lstm_model(x, params):
    with tf.name_scope("embedding"):
        w_embed = tf.Variable(
            tf.constant(0.0, shape=[params['vocab_size'], params['embedding_dim']]), trainable=False, name='w_embed')
        w_embed = w_embed.assign(params['embeddings'])
        embed = tf.nn.embedding_lookup(w_embed, x, name='embed')

    # reshape embed to shape [sequence_len, batch_size, embedding_dim]
    embed = tf.transpose(embed, [1, 0, 2])

    with tf.name_scope("lstm"):
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=params['num_layers'], num_units=params['hidden_size'])
        _, (output_states_h, _) = lstm(embed)

    out = tf.layers.AveragePooling1D(output_states_h[-1], params['pool_size'], params['strides'])

    out = tf.layers.dense(out, units=params['label_size'],
                          kernel_initializer=tf.random_normal_initializer,
                          bias_initializer=tf.random_normal_initializer, activation=tf.nn.relu, name='output')

    return out


def model_fn(features, labels, mode, params):
    logits = lstm_model(features, params)

    predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'probabilities': tf.nn.softmax(logits),
                       'class_ids': predicted_classes[:, tf.newaxis],  # todo: why?
                       'logits': logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss_op = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, reduction=tf.losses.Reduction.MEAN)
    acc_op = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')

    tf.summary.scalar('accuracy', acc_op[1])  # todo: why?

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss_op, eval_metric_ops={'accuracy', acc_op})

    # create training op
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'], epsilon=params['epsilon'])
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    logging_hook = tf.train.LoggingTensorHook({'loss': loss_op, 'accuracy': acc_op}, every_n_iter=100)

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op, train_op=train_op, training_hooks=[logging_hook])


def main(_):
    init_w_embed = np.load(FLAGS.word_vecs_path)

    x_train = np.load(FLAGS.x_train).tolist()
    x_test = np.load(FLAGS.x_test).tolist()
    x_dev = np.load(FLAGS.x_dev).tolist()

    y_train = pd.read_csv(FLAGS.y_train)[label_column].map(label_to_idx).values
    y_test = pd.read_csv(FLAGS.y_test)[label_column].map(label_to_idx).values
    y_dev = pd.read_csv(FLAGS.y_dev)[label_column].map(label_to_idx).values

    vocab_size, embedding_dim = init_w_embed.shape

    PARAMS['vocab_size'] = vocab_size
    PARAMS['embedding_dim'] = embedding_dim

    # write params to a txt file
    if os.path.exists(FLAGS.model_dir):
        raise ValueError('model dir exists, change path')

    os.makedirs(FLAGS.model_dir)
    with open(FLAGS.model_dir + '/params.txt', 'w') as f:
        print(PARAMS)
        f.write(json.dumps(PARAMS))

    PARAMS['embeddings'] = init_w_embed
    del init_w_embed

    model = tf.estimator.Estimator(model_fn, model_dir=FLAGS.model_dir, params=PARAMS)

    # Note that the parameters of "input fn" must be passed in model.train()
    # Otherwise the input tensors and the weight tensors would be in different graphs
    model.train(input_fn=lambda: train_input_fn(
        x_train, y_train, PARAMS['batch_size'], PARAMS['embedding_dim'], label_size=PARAMS['n_class'],
        num_epochs=PARAMS['num_epochs'], max_length=PARAMS['max_length']))

    # Predict
    predictions = model.predict(input_fn=lambda: eval_input_fn(
        x_dev, y_dev, PARAMS['batch_size'], PARAMS['embedding_dim'], label_size=PARAMS['n_class'],
        max_length=PARAMS['max_length']))

    np.save(FLAGS.model_dir + '/doc_vecs', predictions)


if __name__ == '__main__':
    tf.app.run()

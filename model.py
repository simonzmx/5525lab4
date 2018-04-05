import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim

# Hyper parameters
WORD_VECTOR_DIM = 300
MAX_LENGTH = 128  # max sentence length

HIDDEN_DIM = 64  # output dimension of LSTM
N_EPOCHS = 1
WINDOW_SIZE = 3
LEARNING_RATE = 0.1
BATCH_SIZE = 32


class LSTMTagger(nn.Module):
    def __init__(self, word_vector_dim, hidden_dim, kernel_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden = self.init_hidden()

        # The LSTM takes word vectors as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(word_vector_dim, hidden_dim)
        self.max_pooling = torch.nn.MaxPool1d(kernel_size)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def init_hidden(self):
        # Before we've done anything, we do not have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, x):
        x, self.hidden = self.lstm(x.view(len(x), 1, -1), self.hidden)
        x = self.max_pooling(x)  # (128, batch_size, 21)
        x_numel = x.numel()
        output_layer = nn.Linear(x_numel, 1)
        x = output_layer(x.view(x_numel))
        return x


def get_sentences_vectors(w2v_model, sen, max_length, wv_dim):
    # generate vectors for a sentence
    # pad the vectors with 0s when i > sentence length
    # cut off the part beyond sequence_length
    u_words = 0
    vectors = np.zeros((max_length, wv_dim))
    for j in range(max_length):
        if j == len(sen):
            break
        # filter words not in vocabulary
        if sen[j] in w2v_model.wv.vocab:
            vectors[j] = w2v_model.wv[sen[j]]
        else:
            u_words += 1
    return vectors, u_words


##########################
# Train a word2vec model #
##########################
sentences = gensim.models.word2vec.LineSentence('train.txt')
word2vec_model = gensim.models.Word2Vec(sentences, size=WORD_VECTOR_DIM, window=5, min_count=1, workers=4)


############################
# Prepare data for pytorch #
############################
# Get word vectors for training data
n_sentences = len(list(sentences))
training_data = np.zeros((n_sentences, MAX_LENGTH, WORD_VECTOR_DIM))
unseen_words = 0
total_words = 0
for i, sentence in enumerate(sentences):
    total_words += len(sentence)
    training_data[i], temp = get_sentences_vectors(word2vec_model, sentence, MAX_LENGTH, WORD_VECTOR_DIM)
    unseen_words += temp

# Get labels for training data
tags = np.zeros(n_sentences)
with open('train_labels.txt', 'r') as f:
    for i, line in enumerate(f):
        tags[i] = line

##################
# Train the LSTM #
##################
model = LSTMTagger(WORD_VECTOR_DIM, HIDDEN_DIM, WINDOW_SIZE)
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# training_data_var = autograd.Variable(torch.DoubleTensor(training_data))

for epoch in range(N_EPOCHS):  # again, normally you would NOT do 300 epochs, it is toy data
    correct = 0
    total_loss = 0
    for i, (sentence, tag) in enumerate(zip(training_data, tags)):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Turn inputs into pytorch Variables.
        sentence = autograd.Variable(torch.FloatTensor(sentence))
        tag = autograd.Variable(torch.FloatTensor([tag]))

        # Step 3. Run our forward pass.
        tag_pred = model(sentence)

        # Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
        loss = loss_function(tag_pred, tag)
        loss.backward()
        optimizer.step()

        if loss.data[0] < 0.04:
            correct += 1

        total_loss += loss.data[0]

        if i % 100 == 0:
            print('Sample {}, loss {}, tag {}, tag_pred {}'.format(i, loss.data[0], tag.data[0], tag_pred.data[0]))

    print('epoch {}, loss {}, accuracy {}'.format(epoch, total_loss, correct / n_sentences))


##############
# Evaluation #
##############
sentences = gensim.models.word2vec.LineSentence('dev.txt')
unseen_words = 0
total_words = 0

n_sentences = len(list(sentences))
evaluation_data = np.zeros((n_sentences, MAX_LENGTH, WORD_VECTOR_DIM))
for i, sentence in enumerate(sentences):
    total_words += len(sentence)
    evaluation_data[i], temp = get_sentences_vectors(word2vec_model, sentence, MAX_LENGTH, WORD_VECTOR_DIM)
    unseen_words += temp

correct = 0
for sentence, tag in zip(evaluation_data, tags):
    # Step 1. Remember that Pytorch accumulates gradients.
    # We need to clear them out before each instance
    model.zero_grad()

    # Also, we need to clear out the hidden state of the LSTM,
    # detaching it from its history on the last instance.
    model.hidden = model.init_hidden()

    # Step 2. Turn inputs into pytorch Variables.
    sentence = autograd.Variable(torch.FloatTensor(sentence))
    tag = autograd.Variable(torch.FloatTensor([tag]))

    # Step 3. Run our forward pass.
    tag_pred = model(sentence)

    # Step 4. Compute the loss, gradients, and update the parameters by
    #  calling optimizer.step()
    loss = loss_function(tag_pred, tag)

    if loss.data[0] < 0.04:
        correct += 1

print("------------------------------")
print("number of sentences: ", n_sentences)
print("Shape of sentences vectors: ", evaluation_data.shape)
print("Number of unseen words: ", unseen_words / total_words)
print('acc {}'.format(correct / n_sentences))

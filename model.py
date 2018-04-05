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
N_EPOCHS = 3
WINDOW_SIZE = 3
LEARNING_RATE = 0.1
BATCH_SIZE = 32

torch.manual_seed(42)


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, kernel_size):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word vectors as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.max_pooling = torch.nn.MaxPool1d(kernel_size)
        self.out_dim = out_dim
        # self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        batch_size = x.size()[1]
        hidden = (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                  autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)))
        x, hidden = self.lstm(x, hidden)
        x = self.max_pooling(x)  # (MAX_LENGTH, batch_size, -1)
        output_layer = nn.Linear(x[-1].data.shape[1], self.out_dim)  # only keep the output of the last LSTM cell
        x = output_layer(x[-1])
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


def train(model, loss_function, optimizer, x, y):
    x = autograd.Variable(x, requires_grad=False)
    y = autograd.Variable(y, requires_grad=False)
    model.zero_grad()
    y_pred = model(x)  # shape: (32,)
    loss = loss_function(y_pred, y)
    loss.backward()
    optimizer.step()
    error = np.abs(y_pred.data.numpy().ravel() - y.data.numpy())
    return loss.data[0], (error < 0.2).sum()


# Train a word2vec model
sentences = gensim.models.word2vec.LineSentence('train.txt')
word2vec_model = gensim.models.Word2Vec(sentences, size=WORD_VECTOR_DIM, window=5, min_count=1, workers=4)


############################
# Prepare data for pytorch #
############################
# Get word vectors for training data
train_size = len(list(sentences))
x_train = np.zeros((train_size, MAX_LENGTH, WORD_VECTOR_DIM))
total_words = 0
for i, sentence in enumerate(sentences):
    total_words += len(sentence)
    x_train[i], _ = get_sentences_vectors(word2vec_model, sentence, MAX_LENGTH, WORD_VECTOR_DIM)

# x_train.shape: (train_size, MAX_LENGTH, WORD_VECTOR_DIM)
x_train = np.swapaxes(x_train, 0, 1)
# x_train.shape: (MAX_LENGTH, train_size, WORD_VECTOR_DIM)

# convert to float tensor
x_train = torch.from_numpy(x_train).float()

# Get labels for training data
y_train = np.zeros(train_size)  # y_train.shape(train_size,)
with open('train_labels.txt', 'r') as f:
    for i, line in enumerate(f):
        y_train[i] = line

# Get output dimension
if len(y_train.shape) == 1:
    output_dim = 1
else:
    output_dim = y_train.shape[1]

# convert y_train to float tensor
y_train = torch.from_numpy(y_train).float()


# Get evaluation data
sentences = gensim.models.word2vec.LineSentence('dev.txt')
unseen_words = 0
total_words = 0

evaluation_size = len(list(sentences))
x_evaluation = np.zeros((evaluation_size, MAX_LENGTH, WORD_VECTOR_DIM))
for i, sentence in enumerate(sentences):
    total_words += len(sentence)
    x_evaluation[i], temp = get_sentences_vectors(word2vec_model, sentence, MAX_LENGTH, WORD_VECTOR_DIM)
    unseen_words += temp

x_evaluation = np.swapaxes(x_evaluation, 0, 1)
x_evaluation = torch.from_numpy(x_evaluation).float()

y_true = np.zeros(evaluation_size)  # y_train.shape(train_size,)
with open('dev_labels.txt', 'r') as f:
    for i, line in enumerate(f):
        y_true[i] = line

print("------------------------------------------")
print("Proportion of unseen words: ", unseen_words / total_words)


##################
# Train the LSTM #
##################
lstm_model = LSTMNet(WORD_VECTOR_DIM, HIDDEN_DIM, output_dim, WINDOW_SIZE)
mse_loss = nn.MSELoss()
sgd = optim.SGD(lstm_model.parameters(), lr=LEARNING_RATE, momentum=0.9)

print("------------- Begin Training -------------")
for epoch in range(N_EPOCHS):
    total_correct = 0
    total_loss = 0.0
    n_batches = train_size // BATCH_SIZE
    for i in range(n_batches):
        start, end = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
        loss_per_batch, correct_per_batch = train(lstm_model,
                                                  mse_loss,
                                                  sgd,
                                                  x_train[:, start:end, :],
                                                  y_train[start:end])
        total_loss += loss_per_batch
        total_correct += correct_per_batch
    y_evaluation = lstm_model(autograd.Variable(x_evaluation, requires_grad=False))
    evaluation_error = np.abs(y_evaluation.data.numpy().ravel() - y_true)
    evaluation_correct = (evaluation_error < 0.2).sum()
    print('epoch {}, loss {}, train accuracy {}, test accuracy {}'.format(
        epoch, total_loss, total_correct / train_size, evaluation_correct / evaluation_size))

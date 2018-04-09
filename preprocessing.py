import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim

# Hyper parameters
WORD_VECTOR_DIM = 300
MAX_LENGTH = 128  # max sentence length

HIDDEN_DIM = 64  # output dimension of LSTM
N_EPOCHS = 30
WINDOW_SIZE = 2
LEARNING_RATE = 0.1
BATCH_SIZE = 32
torch.manual_seed(42)


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, kernel_size, max_length):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.max_length = max_length

        self.lstm = nn.LSTM(input_dim, hidden_dim)
        # self.pooling = torch.nn.MaxPool1d(kernel_size)
        self.pooling = torch.nn.AvgPool1d(kernel_size)

        fc_in_dim = hidden_dim // kernel_size * max_length

        # self.output_layer = nn.Linear(self.hidden_dim, self.out_dim)
        self.output_layer = nn.Linear(fc_in_dim, self.out_dim)

    def hidden_init(self, batch_size):
        return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)))

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.hidden_init(batch_size)

        x = x.view(self.max_length, batch_size, -1)
        x, hidden = self.lstm(x, hidden)
        x = self.pooling(x)  # (MAX_LENGTH, batch_size, -1)

        x = x.view(batch_size, -1)
        x = self.output_layer(x)
        return F.softmax(x, dim=1)


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
    y_pred = model(x)
    loss = loss_function(y_pred, y) / x.data.shape[0]  # normalize the loss
    loss.backward()
    optimizer.step()

    y_pred = np.argmax(y_pred.data.numpy(), axis=1)
    return loss.data[0], np.sum(y_pred == y.data.numpy())


def predict(model, loss_function, x, y):
    x = autograd.Variable(x, requires_grad=False)
    y = autograd.Variable(y, requires_grad=False)

    y_pred = model(x)
    loss = loss_function(y_pred, y) / x.data.shape[0]

    y_pred = np.argmax(y_pred.data.numpy(), axis=1)
    return loss.data[0], np.sum(y_pred == y.data.numpy())


##########################
# Train a word2vec model #
##########################
sentences = gensim.models.word2vec.LineSentence('train.txt')
word2vec_model = gensim.models.Word2Vec(sentences, size=WORD_VECTOR_DIM, window=5, min_count=1, workers=4)
label_to_idx = {"very negative": 0, "negative": 1, "neutral": 2, "positive": 3, "very positive": 4}


#####################
# Get training data #
#####################
train_size = len(list(sentences))
x_train = np.zeros((train_size, MAX_LENGTH, WORD_VECTOR_DIM))
total_words = 0
for i, sentence in enumerate(sentences):
    total_words += len(sentence)
    x_train[i], _ = get_sentences_vectors(word2vec_model, sentence, MAX_LENGTH, WORD_VECTOR_DIM)

# convert to float tensor
x_train = torch.from_numpy(x_train).float()

# Get labels for training data
y_train = np.zeros(train_size).astype(int)
with open('train_labels.txt', 'r') as f:
    for i, line in enumerate(f):
        y_train[i] = label_to_idx[line[:-1]]  # remove "\n"

y_train = torch.LongTensor(y_train)

train_set = Data.TensorDataset(data_tensor=x_train, target_tensor=y_train)
loader = Data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

#######################
# Get evaluation data #
#######################
sentences = gensim.models.word2vec.LineSentence('dev.txt')
unseen_words = 0
total_words = 0

evaluation_size = len(list(sentences))
x_evaluation = np.zeros((evaluation_size, MAX_LENGTH, WORD_VECTOR_DIM))
for i, sentence in enumerate(sentences):
    total_words += len(sentence)
    x_evaluation[i], temp = get_sentences_vectors(word2vec_model, sentence, MAX_LENGTH, WORD_VECTOR_DIM)
    unseen_words += temp

# convert to float tensor
x_evaluation = torch.from_numpy(x_evaluation).float()

y_evaluation = np.zeros(evaluation_size).astype(int)   # (train_size,)
with open('dev_labels.txt', 'r') as f:
    for i, line in enumerate(f):
        y_evaluation[i] = label_to_idx[line[:-1]]

y_evaluation = torch.LongTensor(y_evaluation)

print("------------------------------------------")
print("Proportion of unseen words: ", unseen_words / total_words)


##################
# Train the LSTM #
##################
lstm_model = LSTMNet(WORD_VECTOR_DIM, HIDDEN_DIM, 5, WINDOW_SIZE, MAX_LENGTH)
nll_loss = nn.CrossEntropyLoss()
# adam = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
adam = optim.SGD(lstm_model.parameters(), lr=LEARNING_RATE, momentum=0.9)

print("------------- Begin Training -------------")
for epoch in range(N_EPOCHS):
    correct_train = 0
    loss_train = 0.0
    # n_batches = train_size // BATCH_SIZE
    for i, (data, label) in enumerate(loader):
        loss_per_batch, correct_per_batch = train(lstm_model, nll_loss, adam, data, label)
        correct_train += correct_per_batch
        loss_train += loss_per_batch

    loss_evaluation, correct_evaluation = predict(lstm_model, nll_loss, x_evaluation, y_evaluation)
    print('epoch {}, train loss {}, test loss {}, train acc {} ,test acc {}'.format(
        epoch, loss_train / train_size * 10000, loss_evaluation * 10000,
        correct_train / train_size, correct_evaluation / evaluation_size))

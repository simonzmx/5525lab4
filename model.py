import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim


####################
# Hyper parameters #
####################
WORD_VECTOR_DIM = 256
MAX_LENGTH = 128  # max sentence length
HIDDEN_DIM = 128   # LSTM output dimension
N_EPOCHS = 20
WINDOW_SIZE = 2
LEARNING_RATE = 1.2
BATCH_SIZE = 32
N_CLASS = 5
label_to_idx = {"very negative": 0, "negative": 1, "neutral": 2, "positive": 3, "very positive": 4}
torch.manual_seed(0)


#########
# Model #
#########
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, kernel_size, max_length):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.max_length = max_length

        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.pooling = torch.nn.AvgPool1d(kernel_size)

        fc_in_dim = hidden_dim // kernel_size * max_length

        self.fc = nn.Linear(fc_in_dim, self.out_dim)

    def hidden_init(self, batch_size):
        h0 = torch.Tensor(1, batch_size, self.hidden_dim)
        c0 = torch.Tensor(1, batch_size, self.hidden_dim)
        nn.init.xavier_normal(h0)
        nn.init.xavier_normal(c0)
        return autograd.Variable(h0), autograd.Variable(c0)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.hidden_init(batch_size)

        x = x.view(self.max_length, batch_size, -1)
        x, hidden = self.lstm(x, hidden)
        x = self.pooling(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
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


def train(model, loss_f, opt, x, y):
    x = autograd.Variable(x, requires_grad=False)
    y = autograd.Variable(y, requires_grad=False)

    model.zero_grad()
    y_pred = model(x)
    loss = loss_f(y_pred, y)
    loss.backward()
    opt.step()

    y_pred = np.argmax(y_pred.data.numpy(), axis=1)
    return loss.data[0], np.sum(y_pred == y.data.numpy())


def predict(model, loss_f, x, y):
    x = autograd.Variable(x, requires_grad=False)
    y = autograd.Variable(y, requires_grad=False)

    y_pred = model(x)
    loss = loss_f(y_pred, y)
    y_pred = np.argmax(y_pred.data.numpy(), axis=1)
    return loss.data[0], np.sum(y_pred == y.data.numpy())


def plot_performance(n_epochs, train_array, eval_array, y_label):
    fig = plt.figure(figsize=(15, 5))
    x_axis = range(n_epochs)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(x_axis, train_array, marker="o", label="train")
    ax.plot(x_axis, eval_array, marker="o", label="eval")
    plt.xticks(x_axis)
    plt.xlabel("epoch")
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(y_label + ".png")


##########################
# Train a word2vec model #
##########################
sentences = gensim.models.word2vec.LineSentence('train.txt')
word2vec_model = gensim.models.Word2Vec(sentences, size=WORD_VECTOR_DIM, window=5, min_count=1, workers=4)


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

eval_size = len(list(sentences))
x_eval = np.zeros((eval_size, MAX_LENGTH, WORD_VECTOR_DIM))
for i, sentence in enumerate(sentences):
    total_words += len(sentence)
    x_eval[i], temp = get_sentences_vectors(word2vec_model, sentence, MAX_LENGTH, WORD_VECTOR_DIM)
    unseen_words += temp

# convert to float tensor
x_eval = torch.from_numpy(x_eval).float()

y_eval = np.zeros(eval_size).astype(int)   # (train_size,)
with open('dev_labels.txt', 'r') as f:
    for i, line in enumerate(f):
        y_eval[i] = label_to_idx[line[:-1]]

y_eval = torch.LongTensor(y_eval)

print("---------------------------------------------------------------------------------")
print("Proportion of unseen words in evaluation set: ", unseen_words / total_words * 100)


##################
# Train the LSTM #
##################
lstm_model = LSTMNet(WORD_VECTOR_DIM, HIDDEN_DIM, N_CLASS, WINDOW_SIZE, MAX_LENGTH)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(lstm_model.parameters(), lr=LEARNING_RATE)

print("------------------------------- Begin Training ---------------------------------")

loss_train = np.zeros(N_EPOCHS)
loss_eval = np.zeros(N_EPOCHS)
acc_train = np.zeros(N_EPOCHS)
acc_eval = np.zeros(N_EPOCHS)

for e in range(N_EPOCHS):
    correct_train = 0
    loss_train_e = 0.0
    for data, label in loader:
        loss_per_batch, correct_per_batch = train(lstm_model, loss_function, optimizer, data, label)
        correct_train += correct_per_batch
        loss_train_e += loss_per_batch

    loss_eval[e], correct_eval = predict(lstm_model, loss_function, x_eval, y_eval)
    loss_train[e] = loss_train_e / len(loader)  # len(loader): number of batches
    acc_train[e] = correct_train / train_size
    acc_eval[e] = correct_eval / eval_size

    print('epoch {}, train loss {}, test loss {}, train acc {} ,test acc {}'.format(
        e, loss_train[e], loss_eval[e], acc_train[e], acc_eval[e]))


##########################
# Plot loss and accuracy #
##########################
plot_performance(N_EPOCHS, loss_train, loss_eval, "loss")
plot_performance(N_EPOCHS, acc_train, acc_eval, "acc")

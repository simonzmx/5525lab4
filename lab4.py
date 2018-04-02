import numpy as np
import tensorflow, gensim
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import pandas


def data_preparation(file_name, dic_name, sl_name, split_file):
    # load sentences and splits
    df = pandas.read_csv(file_name, delimiter='\t')
    split_df = pandas.read_csv(split_file, delimiter=',')

    # split the dataset
    train = df.loc[split_df['splitset_label'] == 1]
    dev = df.loc[split_df['splitset_label'] == 3]
    test = df.loc[split_df['splitset_label'] == 2]

    #load dictionary
    dic = pandas.read_csv(dic_name, delimiter='|', header=None)
    dic.columns = ['phrase', 'index']
    #load sentiment labels
    senlab = pandas.read_csv(sl_name, delimiter='|')

    # generate a dataframe consists of 'sentence' 'sentence ids' 'sentence sentiment labels'
    sen_sen_df = get_sentiment_sentences_dataframe(df, dic, senlab)

    return train, dev, test, sen_sen_df


def train_w2v(train):
    # gensim.models.Word2Vec requires lists of list of word which needs to be done by LineSentence
    train[['sentence']].to_csv('train.txt', header=None, index=None)
    sentences = gensim.models.word2vec.LineSentence('train.txt')

    # train the word2vec model
    # the model gives out a 100d vector representing a word
    wv_model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)

    return wv_model


def get_vector_sequence(wv_model, sentence, sequence_length):
    # generate a vector sequence from a sentence
    vector_sequence = []

    # cut off the part beyond sequence_length
    for i in range(sequence_length):
        if i > len(sentence):
            # pad the sequence with 0's vector
            vector_sequence.append(np.zeros(100))
        else:
            vector_sequence.append(wv_model.wv(sentence[i]))
    return vector_sequence


def get_sentimental_label(sentence, dic, senlab, ucount):
    # get a sentence's sentiment label
    # from 'dictionary.txt' and 'sentiment_labels.txt'
    if (not dic['index'].loc[dic['phrase'] == sentence].values):
        ucount += 1
        return(0.5)
    index = dic['index'].loc[dic['phrase'] == sentence].values[0]
    label = senlab['sentiment values'].loc[senlab['phrase ids'] == index].values[0]
    return label


def get_sentiment_sentences_dataframe(df, dic, senlab):
    labels = []
    ucount = 0
    for i in range(df.shape[0]):
        labels.append(get_sentimental_label(df['sentence'][i], dic, senlab, ucount))
        print(str(i) + '               ' + str(labels[i]))
    df['labels'] = labels
    print('Number of unknown sentences' + str(ucount))
    return df


def train_LSTM(train, dev, wv_model):
    LSTM_model = Sequential()


train, dev, test, sen_sen_df = data_preparation('stanfordSentimentTreebank/datasetSentences.txt',
                                                'stanfordSentimentTreebank/dictionary.txt',
                                                'stanfordSentimentTreebank/sentiment_labels.txt',
                                                'stanfordSentimentTreebank/datasetSplit.txt')
wv_model = train_w2v(train)

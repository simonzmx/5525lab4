import csv
import os
import time

import dill
import numpy as np
import pandas as pd

import warnings; warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models.word2vec import Word2Vec, LineSentence


WORD_VECTOR_DIM = 256
WV_PATH = 'models/wv'


def data_preparation(file_name, dic_name, sl_name, split_file):
    # load sentences and splits
    df = pd.read_csv(file_name, delimiter='\t')
    df_split = pd.read_csv(split_file, delimiter=',')

    # clean sentences
    df['sentence'] = df['sentence'].apply(clean_sentence)

    # load dictionary
    dic = pd.read_csv(dic_name, delimiter='|', header=None)
    dic.columns = ['phrase', 'index']

    # load sentiment labels
    s_values = pd.read_csv(sl_name, delimiter='|')

    # generate a data frame consists of 'sentence' 'sentence id' 'label'
    df = get_sentiment_sentences_dataframe(df, dic, s_values)

    # split the data set
    df_train = df.loc[df_split['splitset_label'] == 1]
    df_test = df.loc[df_split['splitset_label'] == 2]
    df_dev = df.loc[df_split['splitset_label'] == 3]

    return df_train, df_dev, df_test


def get_sentimental_label(sentence, dic, s_values):
    u_count = 0

    index = dic['index'].loc[dic['phrase'] == sentence].values
    if not index:
        u_count += 1
        print(sentence)
        return 0.5, u_count
    index = index[0]
    label = s_values['sentiment values'].loc[s_values['phrase ids'] == index].values[0]
    return label, u_count


def get_sentiment_sentences_dataframe(df, dic, s_values):
    labels = [None] * df.shape[0]
    u_count = 0

    for index, row in df.iterrows():
        labels[index], u_count_temp = get_sentimental_label(row['sentence'], dic, s_values)
        u_count += u_count_temp

    print('Number of unmatched sentences: ' + str(u_count))

    df['score'] = labels
    df['fine_grained'] = pd.cut(df['score'],
                                [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                include_lowest=True,
                                labels=["very negative", "negative", "neutral", "positive", "very positive"])
    df['raw'] = pd.cut(df['score'],
                       [0, 0.5, 1.0],
                       include_lowest=True,
                       labels=["negative", "positive"])
    return df


def clean_sentence(sentence):
    # todo: need further cleaning
    replace_dict = {"-LRB-": "(",
                    "-RRB-": ")",
                    "Ã©": "é",
                    "Ã³": "ó",
                    "Ã­": "í",
                    "Ã¼": "ü",
                    "Ã¡": "á",
                    "Ã¦": "æ",
                    "Â": "",
                    "Ã ": "à",
                    "Ã¢": "â",
                    "Ã±": "ñ",
                    "Ã¯": "ï",
                    "Ã´": "ô",
                    "Ã¨": "è",
                    "Ã¶": "ö",
                    "Ã£": "ã",
                    "Ã»": "û",
                    "Ã§": "ç"}

    for key, value in replace_dict.items():
        sentence = sentence.replace(key, value)
    return sentence


def main():
    start = time.time()

    # print("Start preparing data.")
    # train, dev, test = data_preparation('raw_data/datasetSentences.txt',
    #                                     'raw_data/dictionary.txt',
    #                                     'raw_data/sentiment_labels.txt',
    #                                     'raw_data/datasetSplit.txt')
    #
    # train[['sentence']].to_csv('data/train.txt', header=None, index=None, sep='\t', quoting=csv.QUOTE_NONE)
    # dev[['sentence']].to_csv('data/dev.txt', header=None, index=None, sep='\t', quoting=csv.QUOTE_NONE)
    # test[['sentence']].to_csv('data/test.txt', header=None, index=None, sep='\t', quoting=csv.QUOTE_NONE)
    #
    # train[['score', 'fine_grained', 'raw']].to_csv('data/train_labels.csv', index=None)
    # dev[['score', 'fine_grained', 'raw']].to_csv('data/dev_labels.csv', index=None)
    # test[['score', 'fine_grained', 'raw']].to_csv('data/test_labels.csv', index=None)

    print("Start training w2v.")
    sentences_train = LineSentence('data/train.txt')
    sentences_dev = LineSentence('data/dev.txt')
    sentences_test = LineSentence('data/test.txt')

    word2vec_model = Word2Vec(sentences_train, size=WORD_VECTOR_DIM, window=5, min_count=1, workers=4, negative=5, sg=1)
    word2vec_model.init_sims(replace=True)
    keyed_vectors = word2vec_model.wv

    if not os.path.exists(WV_PATH):
        os.makedirs(WV_PATH)

    # save model
    with open(WV_PATH + '/wv', 'wb') as f:
        dill.dump(keyed_vectors, f)
    # keyed_vectors.save(WV_PATH)

    # append padding values to word vectors (add one word: <PAD>)
    word_vectors = keyed_vectors.vectors
    word_vectors = np.append(word_vectors, np.zeros((1, word_vectors.shape[1])), axis=0)

    x_train = [[keyed_vectors.vocab[token].index for token in d] for d in sentences_train]
    x_dev = [[keyed_vectors.vocab[token].index for token in d if token in keyed_vectors.vocab] for d in sentences_dev]
    x_test = [[keyed_vectors.vocab[token].index for token in d if token in keyed_vectors.vocab] for d in sentences_test]

    np.save('models/wv/word_vectors', word_vectors)
    np.save('models/wv/index2word', keyed_vectors.index2word)
    np.save('models/wv/x_train', x_train)
    np.save('models/wv/x_dev', x_dev)
    np.save('models/wv/x_test', x_test)

    print("Time used: {}".format(time.time() - start))


if __name__ == "__main__":
    main()

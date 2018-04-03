import numpy as np
import pandas as pd
import gensim


def get_vector_sequence(model, sentence, sequence_length, word_vector_dim):
    # generate a vector sequence from a sentence
    vector_sequence = []

    # cut off the part beyond sequence_length
    for i in range(sequence_length):
        if i > len(sentence):
            # pad the sequence with 0s
            vector_sequence.append(np.zeros(word_vector_dim))
        else:
            vector_sequence.append(model.wv(sentence[i]))
    return vector_sequence


##########################
# Train a word2vec model #
##########################
df_train = pd.read_csv('train.csv')

# gensim.models.Word2Vec requires lists of list of word which needs to be done by LineSentence
df_train[['sentence']].to_csv('train.txt', header=None, index=None, encoding='utf-8')

# create a 2D list called sentence where the first dimension represent sentences
# and the second dimension represents words
sentences = gensim.models.word2vec.LineSentence('train.txt')

# train the word2vec model
# the model gives out a 100d vector representing a word
w2v_model = gensim.models.Word2Vec(sentences, size=300, window=5, min_count=1, workers=4)
# w2v_model.save("w2v_model")



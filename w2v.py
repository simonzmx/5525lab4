import numpy as np
import gensim

# Hyper-parameters
word_vector_dim = 300
sentence_length = 128


def get_sequence_vectors(model, sen, max_length, wv_dim):
    # generate a vector sequence from a sentence
    # pad the sequence with 0s when i > sentence length
    # cut off the part beyond sequence_length
    vectors = np.zeros((max_length, wv_dim))
    for j in range(max_length):
        if j == len(list(sentence)):
            break
        vectors[j] = model.wv[sen[j]]
    return vectors


##########################
# Train a word2vec model #
##########################
# create a list of sentences where each sentence is a list of words
sentences = gensim.models.word2vec.LineSentence('train.txt')

# train the word2vec model
w2v_model = gensim.models.Word2Vec(sentences, size=word_vector_dim, window=5, min_count=1, workers=4)
w2v_model.save("w2v_model")

# get word vectors
sentences_vectors = np.zeros((len(list(sentences)), sentence_length, word_vector_dim))
for i, sentence in enumerate(sentences):
    sentences_vectors[i] = get_sequence_vectors(w2v_model, sentence, sentence_length, word_vector_dim)

print(sentences_vectors.shape)

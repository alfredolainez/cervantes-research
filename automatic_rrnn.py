import os
import numpy as np

from cervantes.box import WordVectorBox
from cervantes.language import TwoLevelsEmbedding
from keras.callbacks import EarlyStopping, ModelCheckpoint

def gogogo_sentences(train_texts, train_labels, test_texts, test_labels,
              embedding_file, num_sentences, words_per_sentence, wv_file,
              model_weights_file, model_spec_file, log_file,
              num_classes, keras_params):

    if not os.path.isfile(embedding_file):
        # We need a file with precomputed wordvectors
        print 'Building global word vectors from {}'.format(wv_file)

        gbox = WordVectorBox(WV_FILE)
        gbox.build(zero_token=True, normalize_variance=False, normalize_norm=True)

        # Build the language embedding with the given vector box and 300 words per text
        lembedding = TwoLevelsEmbedding(gbox, num_sentences, words_per_sentence,
                                        TwoLevelsEmbedding.WORD_PARAGRAPH_EMBEDDING)
        lembedding.compute(train_texts + test_texts)
        lembedding.set_labels(train_labels + test_labels)
        lembedding.save(embedding_file)
    else:
        lembedding = TwoLevelsEmbedding.load(embedding_file)

    #  MODEL DEPENDENT
    clf = RRNNClassifier(lembedding, num_classes=num_classes)
    # ! MODEL DEPENDENT
    clf.train(X=lembedding.data[:len(train_labels)], y=lembedding.labels[:len(train_labels)],
              model_weights_file=model_weights_file, model_spec_file=model_spec_file, **keras_params)
    clf.test(X=lembedding.data[len(train_labels):],
                        y=lembedding.labels[len(train_labels):])
    clf.log_results(log_file, X_test=lembedding.data[len(train_labels):],
                    y_test=lembedding.labels[len(train_labels):])

import datasets
from cervantes.nn.models import RRNNClassifier


WV_FILE = '../deep-text/embeddings/wv/glove.42B.300d.120000.txt'
RESULTS_DIR = "./experiments/rrnn_sentences_1/"

FIT_PARAMS = {
    "batch_size": 512,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=10, monitor='val_acc')]
}

print("==" * 40)
print("AG_NEWS")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_ag_news_data()
gogogo_sentences(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="AG_news_sentences.pkl", num_sentences=4, words_per_sentence=70,
          wv_file=WV_FILE,
          model_weights_file=RESULTS_DIR + "ag_news.weights",
          model_spec_file=RESULTS_DIR + "ag_news.spec",
          log_file=RESULTS_DIR + "ag_news.log",
          num_classes=4,
          keras_params=FIT_PARAMS)

#datasets.get_sogou_data()

print("==" * 40)
print("DBPedia")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_dbpedia_data()
FIT_PARAMS = {
    "batch_size": 512,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=10, monitor='val_acc')]
}
gogogo_sentences(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="Dbpedia_sentences.pkl", num_sentences=6, words_per_sentence=54,
          wv_file=WV_FILE,
          model_weights_file=RESULTS_DIR + "dbpedia.weights",
          model_spec_file=RESULTS_DIR + "dbpedia.spec",
          log_file=RESULTS_DIR + "dbpedia.log",
          num_classes=14,
          keras_params=FIT_PARAMS)

print("==" * 40)
print("Yelp Polarity")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_yelp_polarity_data()
FIT_PARAMS = {
    "batch_size": 256,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=10, monitor='val_acc')]
}
gogogo_sentences(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="Yelp_polarity_sentences.pkl", num_sentences=30, words_per_sentence=50,
          wv_file=WV_FILE,
          model_weights_file=RESULTS_DIR + "yelp_polarity.weights",
          model_spec_file=RESULTS_DIR + "yelp_polarity.spec",
          log_file=RESULTS_DIR + "yelp_polarity.log",
          num_classes=2,
          keras_params=FIT_PARAMS)

print("==" * 40)
print("Yelp Full Data")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_yelp_full_data()
FIT_PARAMS = {
    "batch_size": 256,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=10, monitor='val_acc')]
}
gogogo_sentences(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="Yelp_full_sentences.pkl", num_sentences=30, words_per_sentence=50,
          wv_file=WV_FILE,
          model_weights_file=RESULTS_DIR + "yelp_full.weights",
          model_spec_file=RESULTS_DIR + "yelp_full.spec",
          log_file=RESULTS_DIR + "yelp_full.log",
          num_classes=5,
          keras_params=FIT_PARAMS)

print("==" * 40)
print("Yahoo")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_yahoo_data()
FIT_PARAMS = {
    "batch_size": 256,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=10, monitor='val_acc')]
}
gogogo_sentences(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="Yahoo_sentences.pkl", num_sentences=25, words_per_sentence=60,
          wv_file=WV_FILE,
          model_weights_file=RESULTS_DIR + "yahoo.weights",
          model_spec_file=RESULTS_DIR + "yahoo.spec",
          log_file=RESULTS_DIR + "yahoo.log",
          num_classes=10,
          keras_params=FIT_PARAMS)

print("==" * 40)
print("Amazon polarity")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_amazon_polarity_data()
FIT_PARAMS = {
    "batch_size": 256,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=10, monitor='val_acc')]
}
gogogo_sentences(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="Amazon_polarity_sentences.pkl", num_sentences=14, words_per_sentence=50,
          wv_file=WV_FILE,
          model_weights_file=RESULTS_DIR + "amazon_polarity.weights",
          model_spec_file=RESULTS_DIR + "amazon_polarity.spec",
          log_file=RESULTS_DIR + "amazon_polarity.log",
          num_classes=2,
          keras_params=FIT_PARAMS)

print("==" * 40)
print("Amazon Full")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_amazon_full_data()
FIT_PARAMS = {
    "batch_size": 256,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=10, monitor='val_acc')]
}
gogogo_sentences(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="Amazon_full_sentences.pkl", num_sentences=14, words_per_sentence=50,
          wv_file=WV_FILE,
          model_weights_file=RESULTS_DIR + "amazon_full.weights",
          model_spec_file=RESULTS_DIR + "amazon_full.spec",
          log_file=RESULTS_DIR + "amazon_full.log",
          num_classes=5,
          keras_params=FIT_PARAMS)

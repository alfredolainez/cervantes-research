import csv
import os
import numpy as np

from cervantes.box import WordVectorBox
from cervantes.language import OneLevelEmbedding, TwoLevelsEmbedding

from datasets import get_yahoo_data, get_yelp_polarity_data, get_amazon_full_data, \
    get_sogou_data, get_ag_news_data, get_amazon_polarity_data, get_dbpedia_data, get_yelp_full_data

WV_FILE = '../deep-text/embeddings/wv/glove.42B.300d.120000.txt'

EMBEDDING_FILE = "AG_news_sentences.pkl"

print "Getting data in format texts / labels"
train_texts, train_labels, test_texts, test_labels = get_ag_news_data()


if not os.path.isfile(EMBEDDING_FILE):
    print "Building language embeddings. This requires parsing text so it might " \
      "be pretty slow "
    # We need a file with precomputed wordvectors
    print 'Building global word vectors from {}'.format(WV_FILE)

    gbox = WordVectorBox(WV_FILE)
    gbox.build(zero_token=True, normalize_variance=False, normalize_norm=True)

    # Build the language embedding with the given vector box and 300 words per text
    lembedding = TwoLevelsEmbedding(gbox, 4, 70, TwoLevelsEmbedding.WORD_PARAGRAPH_EMBEDDING)
    lembedding.compute(train_texts[:1000])
    lembedding.set_labels(train_labels + test_labels)
    lembedding.save(EMBEDDING_FILE)
else:
    lembedding = OneLevelEmbedding.load(EMBEDDING_FILE)
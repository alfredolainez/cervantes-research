import numpy as np
import matplotlib.pyplot as plt
import sys
from cervantes.language.tokenizer import EnglishTokenizer
import csv
from datasets import get_yahoo_data, get_yelp_polarity_data, get_amazon_full_data, \
    get_sogou_data, get_ag_news_data, get_amazon_polarity_data, get_dbpedia_data, get_yelp_full_data
import random

def sentence_distribution(texts):

    tokenizer = EnglishTokenizer()

    text_sentences = []
    sentences_lens = []
    for (i, text) in enumerate(texts):
        sys.stdout.write('Processing text %d out of %d \r' % (i + 1, len(texts)))
        sys.stdout.flush()
        sentences = tokenizer.tokenize_by_sentences(text)
        text_sentences.append(len(sentences))
        for sent in sentences:
            sentences_lens.append(len(sent))

    print "NUMBER OF SENTENCES"
    print "Percentile 50: " + str(np.percentile(text_sentences, 50))
    print "Percentile 70: " + str(np.percentile(text_sentences, 70))
    print "Percentile 80: " + str(np.percentile(text_sentences, 80))
    print "Percentile 90: " + str(np.percentile(text_sentences, 90))
    print "Percentile 95: " + str(np.percentile(text_sentences, 95))
    print "Percentile 99: " + str(np.percentile(text_sentences, 99))
    print "Percentile 99.5: " + str(np.percentile(text_sentences, 99.5))
    print "Percentile 99.9: " + str(np.percentile(text_sentences, 99.9))
    print
    print "SENTENCES"
    print "Percentile 50: " + str(np.percentile(sentences_lens, 50))
    print "Percentile 70: " + str(np.percentile(sentences_lens, 70))
    print "Percentile 80: " + str(np.percentile(sentences_lens, 80))
    print "Percentile 90: " + str(np.percentile(sentences_lens, 90))
    print "Percentile 95: " + str(np.percentile(sentences_lens, 95))
    print "Percentile 99: " + str(np.percentile(sentences_lens, 99))
    print "Percentile 99.5: " + str(np.percentile(sentences_lens, 99.5))
    print "Percentile 99.9: " + str(np.percentile(sentences_lens, 99.9))

    plt.hist(text_sentences, 20)
    plt.show()

    plt.hist(sentences_lens, 20)
    plt.show()


print "Getting data in format texts / labels"
#train_texts, train_labels, test_texts, test_labels = get_ag_news_data()
train_texts, train_labels, test_texts, test_labels = get_yahoo_data()
subsample = random.sample(train_texts, 30000)

sentence_distribution(subsample)
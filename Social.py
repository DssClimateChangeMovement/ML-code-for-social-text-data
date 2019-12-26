from collections import Counter
from multiprocessing import freeze_support
import xlrd
import nltk
import pandas as pd
import string
import ftfy
import numpy as np
import re
import xlsxwriter
import matplotlib.pyplot as plt
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import openpyxl
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from nltk.probability import ConditionalFreqDist
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy
from pprint import pprint

# plot
import pyLDAvis
import pyLDAvis.gensim  # don't skip this

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel



def main():

    def strip_emoji(text):
        RE_EMOJI = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
        return RE_EMOJI.sub(r'', text)


    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        stop_free = ftfy.fix_text(stop_free)
        # stop_free=ftfy.fix_text(stop_free)
        # txt_clean=""
        # for word in stop_free:
        #     for j in enumerate(startlist):
        #         txt_clean = "".join(word for word in stop_free if not word.startswith('@','pic.twitter.com',"#",))
        txt_clean = ''.join(word for word in stop_free if not word.startswith(starttuple))
        punc_free = ''.join(ch for ch in txt_clean if ch not in exclude)
        normalized = ''.join(word for word in punc_free if ps.stem(word))
        processed = re.sub(r"\d+", "", normalized)
        POStokens = [word for word in nltk.pos_tag(nltk.word_tokenize(processed))]
        return POStokens




    starttuple = ('@', 'pic.twitter.com', "#", "http", "bit.ly")
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    ps = PorterStemmer()

    data = pd.ExcelFile(r'C:\Users\User\Desktop\ret.xlsx')
    # if i do that the type of our objen will be OrderedDict
    xr = pd.read_excel(data, sheet_name=None)
    xr = pd.concat(xr, sort=False, ignore_index=True)
    # xr = pd.read_excel(data)

    # Drop all rows with different language than EN

    for i, y in enumerate(xr.Language):
        if y != "en":
            xr.drop(i, inplace=True)

    # xr.drop(['Tweet Type', 'Tweet_Type'], axis=1)
    # xr.drop(columns="Reach",inplace=True)
    # dd=xr.duplicated(subset=None,keep="first")


    xr["final"] = xr.Tweet.apply(lambda x: strip_emoji(str(x)))
    xr.final = xr.final.apply(lambda x: clean(str(x)))


    # Code that takes and stores in dictionaries Mentions and hashtag
    # tag_dict = {}
    # mention_dict = {}
    # for i in xr.Tweet:
    #
    #     tweet = str(i).lower()
    #     tweet_tokenized = tweet.split()
    #     for word in tweet_tokenized:
    #         # Hashtags - tokenize and build dict of tag counts
    #         if (word[0:1] == '#' and len(word) > 1):
    #             key = word.translate(string.punctuation)
    #             if key in tag_dict:
    #                 tag_dict[key] += 1
    #             else:
    #                 tag_dict[key] = 1
    #         # Mentions - tokenize and build dict of mention counts
    #         if (word[0:1] == '@' and len(word) > 1):
    #             key = word.translate(string.punctuation)
    #             if key in mention_dict:
    #                 mention_dict[key] += 1
    #             else:
    #                 mention_dict[key] = 1

    def cleasn(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        stop_free = ftfy.fix_text(stop_free)
        # stop_free=ftfy.fix_text(stop_free)
        # txt_clean=""
        # for word in stop_free:
        #     for j in enumerate(startlist):
        #         txt_clean = "".join(word for word in stop_free if not word.startswith('@','pic.twitter.com',"#",))
        txt_clean = ''.join(word for word in stop_free if not word.startswith(starttuple))
        punc_free = ''.join(ch for ch in txt_clean if ch not in exclude)
        normalized = ''.join(word for word in punc_free if ps.stem(word))
        processed = re.sub(r"\d+", "", normalized)
        POStokens = nltk.word_tokenize(processed)
        return POStokens


    s = pd.DataFrame()
    s = xr.Tweet.apply(lambda x: strip_emoji(str(x)))

    s = s.apply(lambda x: cleasn(str(x)))

    id2word = corpora.Dictionary(s)
    # Create Corpus
    texts = s
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # model building
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=20,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

    # pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    # a measure of how good the model is. lower the better.

    print('\nPerplexity: ', lda_model.log_perplexity(corpus))

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis,'vis.html')

    # def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    #     # Init output
    #     sent_topics_df = pd.DataFrame()
    #
    #     # Get main topic in each document
    #     for i, row in enumerate(ldamodel[corpus]):
    #         row = sorted(row, key=lambda x: (x[1]), reverse=True)
    #         # Get the Dominant topic, Perc Contribution and Keywords for each document
    #         for j, (topic_num, prop_topic) in enumerate(row):
    #             if j == 0:  # => dominant topic
    #                 wp = ldamodel.show_topic(topic_num)
    #                 topic_keywords = ", ".join([word for word, prop in wp])
    #                 sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
    #             else:
    #                 break
    #     sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    #
    #     # Add original text to the end of the output
    #     contents = pd.Series(texts)
    #     sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    #     return(sent_topics_df)
    #
    #
    # df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)
    #
    # # Format
    # df_dominant_topic = df_topic_sents_keywords.reset_index()
    # df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    #
    # # Show
    # df_dominant_topic.head(10)

if __name__ == "__main__":
    main()
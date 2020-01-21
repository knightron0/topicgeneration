import sys
import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import argparse
from sklearn.datasets import fetch_20newsgroups
from gensim.parsing import preprocessing

stemmer = SnowballStemmer("english")
train = fetch_20newsgroups(shuffle = True)
def lemmatize(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos = 'v'))

def preprocess(text):
    result = []
    for i in gensim.utils.simple_preprocess(text):
        if i not in gensim.parsing.preprocessing.STOPWORDS and len(i) > 3:
            result.append(lemmatize(i))
    return result

if __name__ == "__main__":
    pth = sys.argv[1]
    finwrds = []
    txt2 = ""
    with open(pth, "r") as file:
        txt2 = file.read()
    txt = txt2.split()
    for i in txt:
        rt = preprocess(i)
        if(rt != []):
            finwrds.append(preprocess(i))
    d = gensim.corpora.Dictionary(finwrds)
    bowc = [d.doc2bow(i) for i in finwrds]
    lda_model =  gensim.models.LdaMulticore(bowc, num_topics = 4, id2word = d, passes = 12,workers = 2)
    bowc2 = d.doc2bow(preprocess(txt2))
    for index, score in sorted(lda_model[bowc2], key=lambda tup: -1*tup[1]):
        print("Score: {} \n Topic: {}".format(score, lda_model.print_topic(index, 5)))

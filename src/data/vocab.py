from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
import numpy as np

internal_lexicon = dict()
with open('resources/functional_lexicon.txt', 'r', encoding='utf8') as f:
    for line in f:
        word, postag, sentitag = line.rstrip().split()
        try:
            internal_lexicon[word][postag] = sentitag
        except KeyError:
            internal_lexicon[word] = {postag: sentitag}

lemmatizer = WordNetLemmatizer()

def postag_penn_to_wn(postag):
    if postag.startswith('J'):
        return wn.ADJ # 'a'
    elif postag.startswith('N'):
        return wn.NOUN # 'n'
    elif postag.startswith('R'):
        return wn.ADV # 'r'
    elif postag.startswith('V'):
        return wn.VERB # 'v'
    return None

def get_sentitag(word, postag):
    if word in internal_lexicon:
        return internal_lexicon[word].get(postag, internal_lexicon[word].get('*', 'NULL'))

    wn_postag = postag_penn_to_wn(postag)
    if wn_postag is None: 
        return 'NULL'

    lemma = lemmatizer.lemmatize(word, pos=wn_postag)
    synsets = swn.senti_synsets(lemma, wn_postag)
    scores = np.zeros(3)
    for i, synset in enumerate(synsets):
        synset_scores = np.array([synset.neg_score(), synset.pos_score(), synset.obj_score()])
        scores = (i * scores + synset_scores) / (i + 1)
    sentitag_id = scores.argmax()
    if scores[sentitag_id] > 0.5:
        return ['N', 'P', 'NULL'][sentitag_id]
    else:
        return 'NULL'
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import string
import re
import os

nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))
english_punctuation = string.punctuation.replace("'", '')
lemmatizer = WordNetLemmatizer()
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# reform word
def lemmatize_word(word):
    lemmatizer = WordNetLemmatizer()
    pos_tagged = pos_tag([word])

    word, pos = pos_tagged[0]

    wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN

    return lemmatizer.lemmatize(word, pos=wordnet_pos)



def clean_text(text: str):
    return lemmatize_word(re.sub(r'\s+', " ", text.strip()).lower())


def noun_to_verb(noun):
    nltk_tagged = nltk.pos_tag([noun])
    wordnet_tag = get_wordnet_pos(nltk_tagged[0][1])

    if wordnet_tag == wordnet.NOUN:
        lemmatized_word = lemmatizer.lemmatize(noun, pos=wordnet_tag)
        return lemmatized_word
    return noun


def trans_same_words(texts: list[str]):
    dict1 = {}
    for text in texts:
        word = noun_to_verb(text)
        if word not in dict1:
            dict1[word] = []
        dict1[word].append(text)
    return [v[0] for _, v in dict1.items()]

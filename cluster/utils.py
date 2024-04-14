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
special_word_cast = {"allied": "ally", "synonym": "same"}
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
    if word in special_word_cast:
        return special_word_cast[word]
    lemmatizer = WordNetLemmatizer()
    pos_tagged = pos_tag([word])

    word, pos = pos_tagged[0]

    wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
    ret = lemmatizer.lemmatize(word, pos=wordnet_pos)
    return ret if ret not in special_word_cast else special_word_cast[ret]



def clean_text(text: str):
    return lemmatize_word(re.sub('\.', "", re.sub(r'\s+', " ", text.strip())).lower())

def longest_common_substring_length(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_length = max(max_length, dp[i][j])
            else:
                dp[i][j] = 0
    return max_length


def noun_to_verb(noun):
    nltk_tagged = nltk.pos_tag([noun])
    wordnet_tag = get_wordnet_pos(nltk_tagged[0][1])

    if wordnet_tag != wordnet.NOUN:
        return noun

    best_verb = None
    best_lcs = 0

    synsets = wordnet.synsets(noun, pos=wordnet.NOUN)

    for synset in synsets:
        for lemma in synset.lemmas():
            for related_form in lemma.derivationally_related_forms():
                if related_form.synset().pos() == 'v':
                    verb = related_form.name()
                    lcs = longest_common_substring_length(noun, verb)
                    if lcs > best_lcs:
                        best_lcs = lcs
                        best_verb = verb
    if not best_verb:
        return noun
    return best_verb

def trans_same_words(texts: list[str]):
    dict1 = {}
    for text in texts:
        word = noun_to_verb(text)
        if word not in dict1:
            dict1[word] = []
        dict1[word].append(text)
    return [v[0] for _, v in dict1.items()]

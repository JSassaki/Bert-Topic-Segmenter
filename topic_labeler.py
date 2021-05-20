import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import nltk
import json
import requests
from conllu import parse
from ufal.udpipe import Model, Pipeline, ProcessingError
sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
stopwords=stopwords.words('portuguese')
accepted_pos = ['VERB', 'PROPN', 'NOUN', 'ADJ']
model = Model.load('models/portuguese-bosque-ud-2.5-191206.udpipe')
pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT,
                    Pipeline.DEFAULT, 'conllu')
error = ProcessingError()

def filter_meaningful_words(text):
    ann = pipeline.process(text, error)
    sentences = parse(ann)
    word_list = []
    for sentence in sentences:
        for word in sentence:
            if word['form'] not in stopwords:

                if word['upos'] in accepted_pos:
                    word_list.append(word['lemma'])
    return word_list


file = open("tep2/base_tep2.txt", 'r', encoding='ISO-8859-1')
tep2 = file.read()
file.close()
tep2 = tep2.split('\n')
p = re.compile('\[(.*?)\] {(.*?)}')
category, synonyms = p.search(tep2[0]).groups()


def find_synonyms(word):
    synlist = []
    for i in range(len(tep2) - 1):
        category, synonyms = p.search(tep2[i]).groups()
        synonyms = synonyms.split(', ')
        if word in synonyms:
            synlist+=synonyms
    for syn in wn.synsets(word, lang='por'):
        for lemm in syn.lemmas('por'):
            synlist.append(lemm.name())
    return list(set(synlist))


def common_words(text):
    syn_dict = {}
    text = filter_meaningful_words(text)
    word_counts = Counter(text)
    top_words = [word for word, count in word_counts.most_common(10)]
    [syn_dict.update({word: find_synonyms(word)}) for word in top_words]
    for item in syn_dict:
        contador = 0
        for w in syn_dict[item]:
            contador += word_counts[w]
        if contador:
            word_counts[item] = contador
    return [word for word, count in word_counts.most_common(3)]

def create_label(archive):
    # file = open("stoplist_portugues.txt", 'r', encoding='utf8')
    # stopwords = file.read()
    # file.close()
    file = open("segmented/segmented_"+archive, 'r', encoding='utf8')
    text = file.read()
    file.close()

    text = text.replace("\n\n", "\n ")
    text = sent_tokenizer.tokenize(text.lower())
    print(text)
    topic_list = []
    topic = ""
    text[0] = text[0].replace("¶ ", '')

    for sent in text:

        if sent[0] != "¶":
            topic += (" "+sent)
        else:
            topic_list.append(topic)
            topic = ""
            topic += sent.replace("¶ ", ' ')
    topic_list.append(topic)
    print(topic_list)
    file = open("labeled/labeled_" + archive, "w", encoding="utf8")
    for topic in topic_list:
        # words_in_topic = topic.split(" ")
        # words_in_topic = [t for t in words_in_topic if t.lower() not in stopwords and t.isalnum()]
        print(topic)
        label = common_words(topic)
        for w in label:
            file.write(w.upper() + ' ')
        file.write('\n' + topic + '\n\n')
    print(archive + " rotulado com sucesso.")

import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import nltk
from conllu import parse
from ufal.udpipe import Model, Pipeline, ProcessingError
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
stopwords=stopwords.words('portuguese')
accepted_pos = ['VERB', 'PROPN', 'NOUN', 'ADJ']
model = Model.load('models/portuguese-bosque-ud-2.5-191206.udpipe')
pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT,
                    Pipeline.DEFAULT, 'conllu')
error = ProcessingError()
file = open("blacklist", 'r', encoding='utf8')
blacklist = file.read().split()
file.close()


def filter_meaningful_words(text):
    sentences= sent_tokenizer.tokenize(text)
    word_list = []
    for sentence in sentences:
        ann = pipeline.process(sentence, error)
        udpipe_result=parse(ann)
        for tokenlist in udpipe_result:
            for word in tokenlist:
                if word['form'] not in stopwords:
                    if word['upos'] in accepted_pos:
                        word_list.append(word['lemma'].lower())
    for word in word_list:
        if word in blacklist:
            word_list.remove(word)

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
    # print(list(set(synlist)))
    for syn in synlist:
        if synlist.count(syn)>1:
            synlist.remove(syn)
    return synlist


def common_words(text,df):
    # print(text)
    syn_dict = {}
    text = filter_meaningful_words(text)
    print(text)
    word_counts = Counter(text)
    top_words = [word for word, count in word_counts.most_common(10)]
    [syn_dict.update({word: find_synonyms(word)}) for word in top_words]
    # print(syn_dict)
    for item in syn_dict:
        # print("getting syns for", item)
        contador = 0
        for w in syn_dict[item]:
            contador += word_counts[w]
            if item != w:
                del word_counts[w]
                # print("deleting", w)
        if contador:
            word_counts[item] = contador
    # print(word_counts)
    dict = {}
    for w in top_words:
        dict.update({w:df.at[w, 'TF-IDF']})
    top_words_idf=[]
    for i in sorted(dict.items(), key=lambda item: item[1], reverse=True):
        top_words_idf.append(i[0])
    number_keywords=int(len(text)/15)
    if number_keywords < 3:
        number_keywords=3
    if number_keywords > 5:
        number_keywords=5

    return top_words_idf[:number_keywords]

def create_label(archive):
    startTime = time.time()
    file = open("segmented/novos/1.2_segmented_"+archive, 'r', encoding='utf8')
    text = file.read()
    file.close()

    text = text.replace("\n\n", "\n ")
    text = sent_tokenizer.tokenize(text)
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
    text_filtered=[]
    for topic in topic_list:
        text_filtered.append(' '.join(filter_meaningful_words(topic)))
    vectorizer = TfidfVectorizer(stop_words=stopwords,token_pattern=r"\S+")
    vectors = vectorizer.fit_transform(text_filtered)
    df = pd.DataFrame(vectors[0].T.todense(), index=vectorizer.get_feature_names(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    file = open("labeled/tfidf/labeled_" + archive, "w+", encoding="utf8")
    for topic in topic_list:
        label = common_words(topic,df)
        for w in label:
            file.write(w.upper() + ' ')
        file.write('\n' + topic + '\n\n')
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
    # print(archive + " rotulado com sucesso.")

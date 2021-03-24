import nltk


def segmentation_difference(archive):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    file = open("fulltexts/" + archive, 'r', encoding='utf8')
    text1 = file.read()
    file.close()
    text1 = sent_tokenizer.tokenize(text1)
    sentences1 = []
    for sent in text1:
        sent = sent.strip()
        sent = sent.replace('\n', ' ')
        sentences1.append(sent)
    file = open("segmented/segmented_" + archive, "r", encoding="utf8")
    text2 = file.read()
    text2 = sent_tokenizer.tokenize(text2)
    sentences2 = []
    for sent in text2:
        sent = sent.strip()
        sent = sent.replace('\n', ' ')
        sentences2.append(sent)
    topic_list1 = []
    for sent_n in range(0, len(sentences1)):
        if sentences1[sent_n][0] != "¶":
            topic_list1.append(0)
        else:
            topic_list1.append(1)
    topic_list2 = []
    for sent_n in range(0, len(sentences2)):
        if sentences2[sent_n][0] != "¶":
            topic_list2.append(0)
        else:
            topic_list2.append(1)

    true_positive = 0
    false_positive = 0
    false_negative = 0
    window_true_positive = 0
    window_false_negative = 0
    window_false_positive = 0
    for i in range(0, len(topic_list1)):
        if topic_list1[i] == 1:
            if topic_list2[i] == 1:
                true_positive += 1
            else:
                false_negative += 1
            if i > 0:
                if i < len(topic_list1) - 1:
                    if topic_list2[i] == 1 or topic_list2[i - 1] == 1 or topic_list2[i + 1] == 1:
                        window_true_positive += 1
                    else:  # falso negativo janela
                        window_false_negative += 1
                else:
                    if topic_list2[i] == 1 or topic_list2[i - 1] == 1:  # verdadeiro positivo janela
                        window_true_positive += 1
                    else:  # falso negativo janela
                        window_false_negative += 1
            else:
                if i < len(topic_list1) - 1:
                    if topic_list2[i] == 1 or topic_list2[i + 1] == 1:  # verdadeiro positivo janela
                        window_true_positive += 1
                    else:  # falso negativo janela
                        window_false_negative += 1
                else:
                    if topic_list2[i] == 1:  # verdadeiro positivo janela
                        window_true_positive += 1
                        true_positive += 1
                    else:  # falso negativo janela
                        window_false_negative += 1
                        false_negative += 1
        else:
            if topic_list2[i] == 1:
                false_positive += 1
            if i > 0:
                if i < len(topic_list1) - 1:
                    if topic_list2[i] == 1 or topic_list2[i - 1] == 1 or topic_list2[i + 1] == 1:
                        window_false_positive += 1
                else:
                    if topic_list2[i] == 1 or topic_list2[i - 1] == 1:
                        window_false_positive += 1
            else:
                if i < len(topic_list1) - 1:
                    if topic_list2[i] == 1 or topic_list2[i + 1] == 1:
                        window_false_positive += 1
                else:
                    if topic_list2[i] == 1:
                        window_false_positive += 1
                        false_positive += 1

    print("Text: " + archive)
    print("Precision: " + str(true_positive / (true_positive + false_positive)))
    print("Recall: " + str(true_positive / (true_positive + false_negative)))
    print("Precision w/ window: " + str(window_true_positive / (window_true_positive + false_positive)))
    print("Recall w/ window: " + str(window_true_positive / (window_true_positive + false_negative)))
    print("\n")

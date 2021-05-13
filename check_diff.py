import nltk
from nltk import segmentation
from nltk.metrics import windowdiff
import statistics

def segmentation_difference(archive):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    file = open("fulltexts/" + archive, 'r', encoding='utf8')
    ref_text = file.read()
    file.close()
    ref_text = sent_tokenizer.tokenize(ref_text)
    ref_sentences = []
    for sent in ref_text:
        sent = sent.strip()
        sent = sent.replace('\n', ' ')
        ref_sentences.append(sent)
    file = open("segmented/segmented_" + archive, "r", encoding="utf8")
    hyp_text = file.read()
    hyp_text = sent_tokenizer.tokenize(hyp_text)
    hyp_sentences = []
    topics=0
    for sent in hyp_text:
        sent = sent.strip()
        sent = sent.replace('\n', ' ')
        hyp_sentences.append(sent)
    ref_topic_list = []
    for sent_n in range(0, len(ref_sentences)):
        if ref_sentences[sent_n][0] != "¶":
            ref_topic_list.append(0)
        else:
            topics+=1
            ref_topic_list.append(1)
    hyp_topic_list = []
    mean=int(len(ref_topic_list)/topics+1)
    print("Mean: " + str(mean))
    for sent_n in range(0, len(hyp_sentences)):
        if hyp_sentences[sent_n][0] != "¶":
            hyp_topic_list.append(0)
        else:
            hyp_topic_list.append(1)
    seg_str1 = "".join(map(str, ref_topic_list))
    seg_str2 = "".join(map(str, hyp_topic_list))

    true_positive = 0
    false_positive = 0
    false_negative = 0
    cont_topics_ref = 0
    cont_topics_hyp = 0

    for i in range(0, len(ref_topic_list)):
        if ref_topic_list[i] == 1:
            cont_topics_ref+=1
            if hyp_topic_list[i] == 1:
                cont_topics_hyp+=1
                true_positive += 1
            else:
                false_negative += 1
        else:
            if hyp_topic_list[i] == 1:
                cont_topics_hyp += 1
                false_positive += 1

    print("Text: " + archive)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    print("True Positives: "+ str(true_positive))
    print("False Positives: " + str(false_positive))
    print("False Negatives: " + str(false_negative))
    print("Precision: " + '%.2f' % (precision * 100)+'%')
    print("Recall: " + '%.2f' % (recall * 100)+'%')
    print("F-score: " + '%.2f' % (200 * precision * recall / (precision + recall))+'%')
    print("WindowDiff: "+'%.2f' % segmentation.windowdiff(seg_str1,seg_str2,int(mean/2)))
    print("Pk: " + '%.2f' % segmentation.pk(seg_str1, seg_str2))
    print("Number of topics in original: " + str(cont_topics_ref))
    print("Number of topics in algorithm: " + str(cont_topics_hyp))
    print(ref_topic_list)
    print(hyp_topic_list)
    print(len(hyp_topic_list))
    print("\n")
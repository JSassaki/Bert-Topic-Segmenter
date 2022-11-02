import nltk
sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
from nltk import segmentation
# import nltk.tokenize.punkt as pkt
# from nltk.metrics import windowdiff
# import statistics
# class CustomLanguageVars(pkt.PunktLanguageVars):
#     _period_context_fmt = r"""
#             \S*                          # some word material
#             %(SentEndChars)s             # a potential sentence ending
#             \s*                       #  <-- THIS is what I changed
#             (?=(?P<after_tok>
#                 %(NonWord)s              # either other punctuation
#                 |
#                 (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
#             ))"""
#
#
# sent_tokenizer = pkt.PunktSentenceTokenizer(lang_vars=CustomLanguageVars())

def segmentation_difference(archive,c):
    # sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    file = open(archive, 'r', encoding='utf8')
    ref_text = file.read()
    file.close()
    ref_text = ref_text.replace("\n\n", "\n")
    ref_text = sent_tokenizer.tokenize(ref_text)
    # print(ref_text)
    del ref_text[-1]
    # ref_sentences = []
    # for sent in ref_text:
    #     # sent = sent.strip()
    #     sent = sent.replace('\n', ' ')
    #     ref_sentences.append(sent)
    file = open(archive+"_segmented", "r", encoding="utf8")
    hyp_text = file.read()
    file.close()
    hyp_text = hyp_text.replace("\n\n", "\n")
    hyp_text = sent_tokenizer.tokenize(hyp_text)
    hyp_sentences = []
    topics=0
    # print(archive)
    for sent in hyp_text:
        sent = sent.strip()
        sent = sent.replace('\n', ' ')
        hyp_sentences.append(sent)
    ref_topic_list = []
    for sent in ref_text:
        sent = sent.replace('\n', ' ')
        if not "==========" in sent:
            sent=sent.replace("==========","")
            ref_topic_list.append(0)
        else:
            topics+=1
            ref_topic_list.append(1)
    hyp_topic_list = []
    mean=int(len(ref_topic_list)/topics+1)
    # print("Mean: " + str(mean))
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
    smallest=99
    biggest=0
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
    cont=1
    media=0
    # print(archive)
    for i in hyp_topic_list[1:]:
        if i:
            if cont > biggest:
                biggest=cont
            if cont < smallest:
                smallest=cont
            media+=cont
            cont=1
        else:
            cont+=1
    if cont > biggest:
        biggest = cont
    if cont < smallest:
        smallest = cont
    # print(biggest, smallest)

    # print("Text: " + archive)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    # print("True Positives: "+ str(true_positive))
    # print("False Positives: " + str(false_positive))
    # print("False Negatives: " + str(false_negative))
    # print("Precision: " + '%.2f' % (precision * 100)+'%')
    # print("Recall: " + '%.2f' % (recall * 100)+'%')
    # print("F-score: " + '%.2f' % (200 * precision * recall / (precision + recall))+'%')
    # print("WindowDiff: "+'%.2f' % segmentation.windowdiff(seg_str1,seg_str2,int(mean/2)))
    # print("Pk: " + '%.2f' % segmentation.pk(seg_str1, seg_str2))
    # print("Number of topics in original: " + str(cont_topics_ref))
    # print("Number of topics in algorithm: " + str(cont_topics_hyp))
    # print(str(true_positive) +"\t"+ str(false_positive) +"\t"+ str(false_negative) +"\t"+ '%.2f' % (precision * 100)+'%' +"\t"+ '%.2f' % (recall * 100)+'%'+"\t"+ '%.2f' % (200 * precision * recall / (precision + recall))+'%'+"\t"+'%.2f' % segmentation.windowdiff(seg_str1,seg_str2,int(mean/2))+"\t"+ '%.2f' % segmentation.pk(seg_str1, seg_str2))
    # print(ref_topic_list)
    # print(hyp_topic_list)
    # print(len(hyp_topic_list))
    # print("\n")
    return [precision, recall, 2 * precision * recall / (precision + recall),
            segmentation.windowdiff(seg_str1, seg_str2, int(mean / 2)), segmentation.pk(seg_str1, seg_str2), biggest, smallest,media/cont_topics_hyp]

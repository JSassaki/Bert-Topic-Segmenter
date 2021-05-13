# # from gensim.models import KeyedVectors
# import numpy as np
# # from gensim import similarities
# # from gensim import matutils
# # import matplotlib.pyplot as plt
# # import uts
# # import pickle
# # import gensim
# import nltk
# # from nltk.metrics.segmentation import pk
# import time
# # import csv
# # import TextCleaner
# # from nltk.tokenize import word_tokenize
# from sent2vec.vectorizer import Vectorizer
# from scipy import spatial
# # from collections import Counter
#
# sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
#
# filename = 'listacompleta'
# vectorizer = Vectorizer()
#
#
# def sim_matrix(sentences):
#     n = len(sentences)
#     sentence_matrix = np.zeros((n, n))
#     vectorizer.bert(sentences)
#     print('vetorizado')
#     vectors_bert = vectorizer.vectors
#     for i in range(n):
#         for j in range(i, n):
#             sentence_matrix[i][j] = 1 / (1 + spatial.distance.cosine(vectors_bert[i], vectors_bert[j]))
#             sentence_matrix[j][i] = sentence_matrix[i][j]
#     # plt.imshow(sentence_matrix)
#     # plt.colorbar()
#     # plt.title(archive)
#     # plt.savefig(archive + '.png', dpi=1200)
#     # plt.show()
#
#     return sentence_matrix
#
#
# def rank_matrix(sim_matrix):
#     n = len(sim_matrix)
#     window = (min(n, 11))
#     rank = np.zeros((n, n))
#     for i in range(n):
#         for j in range(i, n):
#             r1 = max(0, i - int(window / 2))
#             r2 = min(n - 1, i + int(window / 2))
#             c1 = max(0, j - int(window / 2))
#             c2 = min(n - 1, j + int(window / 2))
#             sublist = sim_matrix[r1:(r2 + 1), c1:(c2 + 1)].flatten()
#             lowlist = [x for x in sublist if x < sim_matrix[i][j]]
#             rank[i][j] = len(lowlist) / ((r2 - r1 + 1) * (c2 - c1 + 1))
#             rank[j][i] = rank[i][j]
#     # plt.imshow(rank)
#     # plt.colorbar()
#     # plt.title(archive + '_filtered')
#     # plt.savefig(archive + '_filtered.png', dpi=1200)
#     # plt.show()
#
#     # Reynars maximization algorithm
#     sm = np.zeros((n, n))
#     prefix_sm = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             prefix_sm[i][j] = rank[i][j]
#             if i - 1 >= 0: prefix_sm[i][j] += prefix_sm[i - 1][j]
#             if j - 1 >= 0: prefix_sm[i][j] += prefix_sm[i][j - 1]
#             if i - 1 >= 0 and j - 1 >= 0: prefix_sm[i][j] -= prefix_sm[i - 1][j - 1]
#     for i in range(n):
#         for j in range(i, n):
#             if i == 0:
#                 sm[i][j] = prefix_sm[j][j]
#             else:
#                 sm[i][j] = prefix_sm[j][j] - prefix_sm[i - 1][j] \
#                            - prefix_sm[j][i - 1] + prefix_sm[i - 1][i - 1]
#             sm[j][i] = sm[i][j]
#
#     D = 1.0 * sm[0][n - 1] / (n * n)
#     darr, region_arr, idx = [D], [Region(0, n - 1, sm)], []
#     sum_region, sum_area = float(sm[0][n - 1]), float(n * n)
#     for i in range(n - 1):
#         mx, pos = -1e9, -1
#         for j, region in enumerate(region_arr):
#             if region.l == region.r:
#                 continue
#             region.split(sm)
#             den = sum_area - region.area + region.lch.area + region.rch.area
#             cur = (sum_region - region.tot + region.lch.tot + region.rch.tot) / den
#             if cur > mx:
#                 mx, pos = cur, j
#         assert (pos >= 0)
#         tmp = region_arr[pos]
#         region_arr[pos] = tmp.rch
#         region_arr.insert(pos, tmp.lch)
#         sum_region += tmp.lch.tot + tmp.rch.tot - tmp.tot
#         sum_area += tmp.lch.area + tmp.rch.area - tmp.area
#         darr.append(sum_region / sum_area)
#         idx.append(tmp.best_pos)
#
#     dgrad = [(darr[i + 1] - darr[i]) for i in range(len(darr) - 1)]
#     smooth_dgrad = [dgrad[i] for i in range(len(dgrad))]
#     if len(dgrad) > 1:
#         smooth_dgrad[0] = (dgrad[0] * 2 + dgrad[1]) / 3.0
#         smooth_dgrad[-1] = (dgrad[-1] * 2 + dgrad[-2]) / 3.0
#     for i in range(1, len(dgrad) - 1):
#         smooth_dgrad[i] = (dgrad[i - 1] + 2 * dgrad[i] + dgrad[i + 1]) / 4.0
#     dgrad = smooth_dgrad
#
#     avg, stdev = np.average(dgrad), np.std(dgrad)
#     cutoff = avg + 2 * stdev
#     assert (len(idx) == len(dgrad))
#     above_cutoff_idx = [i for i in range(len(dgrad)) if dgrad[i] >= cutoff]
#     if len(above_cutoff_idx) == 0:
#         boundary = []
#     else:
#         boundary = idx[:max(above_cutoff_idx) + 1]
#     ret = [0 for _ in range(n)]
#     for i in boundary:
#         ret[i] = 1
#         # boundary should not be too close
#         for j in range(i - 1, i + 2):
#             if j >= 0 and j < n and j != i and ret[j] == 1:
#                 ret[i] = 0
#                 break
#     return [1] + ret[:-1]
#     # return rank
#
#
# class Region:
#     """
#     Used to denote a rectangular region of similarity matrix,
#     never instantiate this class outside the package.
#     """
#
#     def __init__(self, l, r, sm_matrix):
#         assert (r >= l)
#         self.tot = sm_matrix[l][r]
#         self.l = l
#         self.r = r
#         self.area = (r - l + 1) ** 2
#         self.lch, self.rch, self.best_pos = None, None, -1
#
#     def split(self, sm_matrix):
#         if self.best_pos >= 0:
#             return
#         if self.l == self.r:
#             self.best_pos = self.l
#             return
#         assert (self.r > self.l)
#         mx, pos = -1e9, -1
#         for i in range(self.l, self.r):
#             carea = (i - self.l + 1) ** 2 + (self.r - i) ** 2
#             cur = (sm_matrix[self.l][i] + sm_matrix[i + 1][self.r]) / carea
#             if cur > mx:
#                 mx, pos = cur, i
#         assert (self.l <= pos < self.r)
#         self.lch = Region(self.l, pos, sm_matrix)
#         self.rch = Region(pos + 1, self.r, sm_matrix)
#         self.best_pos = pos
#
# def segment_topics(archive):
#
#     cont = 0
#     file = open("stoplist_portugues.txt", 'r', encoding='utf8')
#     stopwords = file.read()
#     file.close()
#     start_time = time.time()
#     cont += 1
#     # TextCleaner.clean_main(archive)
#     file = open("fulltexts/" + archive, 'r', encoding='utf8')
#     text = file.read()
#     file.close()
#     text = sent_tokenizer.tokenize(text)
#     sentences = []
#     sentences1 = []
#     for sent in text:
#         sent = sent.strip()
#         sent = sent.replace('\n', ' ')
#         sentences1.append(sent)
#         sent = sent.replace("¶ ", '')
#         sentences.append(sent)
#     n = len(sentences)
#     sent_sim_matrix = sim_matrix(sentences)
#     clusters = rank_matrix(sent_sim_matrix)
#     print(sent_sim_matrix)
#     print(archive)
#     print(clusters)
#     # topic_list1 = []
#     # for sent_n in range(0, len(sentences1)):
#     #     if sentences1[sent_n][0] != "¶":
#     #         topic_list1.append(0)
#     #     else:
#     #         topic_list1.append(1)
#     # file = open('clusters', 'a')
#     # file.write(archive)
#     # file.write(str(clusters))
#     # file.write('\n')
#     # file.close()
#     topic_n = 0
#     topic_list = [sentences[0] + '\n']
#
#     for i in range(1, n):
#
#         if not clusters[i]:
#             topic_list[topic_n] += (sentences[i] + '\n')
#         else:
#             topic_list.append(sentences[i] + '\n')
#             topic_n += 1
#     file = open("segmented/segmented_" + archive, "w", encoding="utf8")
#
#     for topic in topic_list:
#         # word = []
#         # words_in_topic = topic.split(" ")
#         # words_in_topic = (t for t in words_in_topic if t.lower() not in stopwords and t.isalnum())
#         # for w, v in Counter(words_in_topic).most_common(3):
#         #     file.write(w.upper() + ' ')
#
#         file.write('¶ '+topic + "\n")
#     file.close()
#
#     end_time = time.time()
#     time_taken = end_time - start_time

import numpy as np
import nltk
from vectorizer import Vectorizer
sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
import matplotlib.pyplot as plt
vectorizer = Vectorizer()
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')


def similarity_matrix(sentences):
    ## with sentence_transformers
    embeddings = model.encode(sentences, convert_to_tensor=True)
    cosine_scores = abs(util.pytorch_cos_sim(embeddings, embeddings).cpu())
    return cosine_scores


def rank_matrix(sim_matrix):
    n = len(sim_matrix)
    window = (min(n, 11))
    rank = np.zeros((n, n))
    # gargalo
    for i in range(n):
        for j in range(i, n):
            r1 = max(0, i - int(window + 1))
            r2 = min(n - 1, i + int(window - 1))
            c1 = max(0, j - int(window + 1))
            c2 = min(n - 1, j + int(window - 1))
            sublist = sim_matrix[r1:(r2 + 1), c1:(c2 + 1)].flatten()
            lowlist = [x for x in sublist if x < sim_matrix[i][j]]
            rank[i][j] = len(lowlist) / ((r2 - r1 + 1) * (c2 - c1 + 1))
            rank[j][i] = rank[i][j]
    # plt.imshow(rank)
    # plt.colorbar()
    # plt.show()
    # Reynars maximization algorithm
    # Kibado: https://github.com/intfloat/uts/blob/master/uts/c99.py
    sm = np.zeros((n, n))
    prefix_sm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            prefix_sm[i][j] = rank[i][j]
            if i - 1 >= 0:
                prefix_sm[i][j] += prefix_sm[i - 1][j]
            if j - 1 >= 0:
                prefix_sm[i][j] += prefix_sm[i][j - 1]
            if i - 1 >= 0 and j - 1 >= 0:
                prefix_sm[i][j] -= prefix_sm[i - 1][j - 1]
    for i in range(n):
        for j in range(i, n):
            if i == 0:
                sm[i][j] = prefix_sm[j][j]
            else:
                sm[i][j] = prefix_sm[j][j] - prefix_sm[i - 1][j] \
                           - prefix_sm[j][i - 1] + prefix_sm[i - 1][i - 1]
            sm[j][i] = sm[i][j]

    d = 1.0 * sm[0][n - 1] / (n * n)
    darr, region_arr, idx = [d], [Region(0, n - 1, sm)], []
    sum_region, sum_area = float(sm[0][n - 1]), float(n * n)
    for i in range(n - 1):
        mx, pos = -1e9, -1
        for j, region in enumerate(region_arr):
            if region.left == region.right:
                continue
            region.split(sm)
            den = sum_area - region.area + region.lch.area + region.rch.area
            cur = (sum_region - region.tot + region.lch.tot + region.rch.tot) / den
            if cur > mx:
                mx, pos = cur, j
        assert (pos >= 0)
        tmp = region_arr[pos]
        region_arr[pos] = tmp.rch
        region_arr.insert(pos, tmp.lch)
        sum_region += tmp.lch.tot + tmp.rch.tot - tmp.tot
        sum_area += tmp.lch.area + tmp.rch.area - tmp.area
        darr.append(sum_region / sum_area)
        idx.append(tmp.best_pos)

    dgrad = [(darr[i + 1] - darr[i]) for i in range(len(darr) - 1)]
    smooth_dgrad = [dgrad[i] for i in range(len(dgrad))]
    if len(dgrad) > 1:
        smooth_dgrad[0] = (dgrad[0] * 2 + dgrad[1]) / 3.0
        smooth_dgrad[-1] = (dgrad[-1] * 2 + dgrad[-2]) / 3.0
    for i in range(1, len(dgrad) - 1):
        smooth_dgrad[i] = (dgrad[i - 1] + 2 * dgrad[i] + dgrad[i + 1]) / 4.0
    dgrad = smooth_dgrad
    avg, stdev = np.average(dgrad), np.std(dgrad)
    cutoff = avg + 1.2 * stdev
    assert (len(idx) == len(dgrad))
    above_cutoff_idx = [i for i in range(len(dgrad)) if dgrad[i] >= cutoff]
    if len(above_cutoff_idx) == 0:
        boundary = []
    else:
        boundary = idx[:max(above_cutoff_idx) + 1]
    ret = [0 for _ in range(n)]
    for i in boundary:
        ret[i] = 1
        # boundary should not be too close
        for j in range(i - 1, i + 2):
            if 0 <= j < n and j != i and ret[j] == 1:
                ret[i] = 0
                break
    return [1] + ret[:-1], rank
    # return rank


class Region:
    """
    Used to denote a rectangular region of similarity matrix,
    never instantiate this class outside the package.
    """

    def __init__(self, l, r, sm_matrix):
        assert (r >= l)
        self.tot = sm_matrix[l][r]
        self.left = l
        self.right = r
        self.area = (r - l + 1) ** 2
        self.lch, self.rch, self.best_pos = None, None, -1

    def split(self, sm_matrix):
        if self.best_pos >= 0:
            return
        if self.left == self.right:
            self.best_pos = self.left
            return
        assert (self.right > self.left)
        mx, pos = -1e9, -1
        for i in range(self.left, self.right):
            carea = (i - self.left + 1) ** 2 + (self.right - i) ** 2
            cur = (sm_matrix[self.left][i] + sm_matrix[i + 1][self.right]) / carea
            if cur > mx:
                mx, pos = cur, i
        assert (self.left <= pos < self.right)
        self.lch = Region(self.left, pos, sm_matrix)
        self.rch = Region(pos + 1, self.right, sm_matrix)
        self.best_pos = pos





def plot(clusters, rank_sim_matrix, sentences, reference, archive):
    hypothetical = []
    for i in range(len(clusters)):
        if clusters[i] == 1:
            hypothetical.append(i)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plt.rc('font', size=20)
    f.set_figheight(10)
    f.set_figwidth(30)
    ax1.set_title("similarity matrix")
    ax1.imshow(rank_sim_matrix, cmap='gray')

    ax2.set_title("hypothetical breakpoints")
    ax2.imshow(rank_sim_matrix, cmap='gray')
    anterior = 0
    for i in hypothetical:
        if i > 0:
            rectangle = plt.Rectangle((anterior - 0.5, anterior - 0.5), i - anterior, i - anterior, fc=(0, 0, 0, 0),
                                      ec="red", lw=2)
            ax2.add_patch(rectangle)
            anterior = i
    rectangle = plt.Rectangle((anterior - 0.5, anterior - 0.5), len(sentences), len(sentences), fc=(0, 0, 0, 0),
                              ec="red", lw=2)
    ax2.add_patch(rectangle)
    ax3.set_title("original breakpoints")
    im = ax3.imshow(rank_sim_matrix, cmap='gray')

    anterior = 0
    for i in hypothetical:
        if i > 0:
            rectangle = plt.Rectangle((anterior - 0.5, anterior - 0.5), i - anterior, i - anterior, fc=(0, 0, 0, 0),
                                      ec="red", lw=2)
            ax3.add_patch(rectangle)
            anterior = i
    rectangle = plt.Rectangle((anterior - 0.5, anterior - 0.5), len(sentences), len(sentences), fc=(0, 0, 0, 0),
                              ec="red", lw=2)
    ax3.add_patch(rectangle)
    anterior = 0
    for i in reference:
        if i > 0:
            rectangle = plt.Rectangle((anterior - 0.5, anterior - 0.5), i - anterior, i - anterior, fc=(0, 0, 0, 0),
                                      ec="magenta", lw=2)
            ax3.add_patch(rectangle)
            anterior = i
    rectangle = plt.Rectangle((anterior - 0.5, anterior - 0.5), len(sentences), len(sentences), fc=(0, 0, 0, 0),
                              lw=2,
                              ec="magenta", )
    ax3.add_patch(rectangle)
    cb_ax = f.add_axes([0.93, 0.1, 0.02, 0.8])
    cbar = f.colorbar(im, cax=cb_ax)
    plt.title(archive)
    plt.savefig("renders/" + archive + "_new.png")
    plt.show()


def segment_topics(archive):
    cont = 1
    file = open("fulltexts/" + archive, 'r', encoding='utf8')
    text = file.read()
    file.close()
    text = text.replace("\n\n", "\n")
    text = sent_tokenizer.tokenize(text)
    sentences = []
    reference = []
    contador = 0
    for sent in text:
        sent = sent.replace('\n', ' ')
        if "¶ " in sent:
            reference.append(contador)
            sent = sent.replace("¶ ", '')
        sentences.append(sent)
        contador += 1
    n = len(sentences)
    print(n)
    sent_sim_matrix = similarity_matrix(sentences)
    clusters, rank_sim_matrix = rank_matrix(sent_sim_matrix)
    topic_n = 0
    topic_list = [sentences[0] + '\n']

    for i in range(1, n):
        if not clusters[i]:
            topic_list[topic_n] += (sentences[i] + '\n')
        else:
            topic_list.append(sentences[i] + '\n')
            topic_n += 1
    file = open("segmented/segmented_" + archive, "w", encoding="utf8")
    for topic in topic_list:
        file.write('¶ ' + topic + "\n")
    file.close()
    plot(clusters, rank_sim_matrix, sentences, reference, archive)
    # print("Reynar: ", clusters)
    # print("Spectral Clustering: ", SpectralClustering(14,affinity='precomputed').fit_predict(sent_sim_matrix))
    # eigen_values, eigen_vectors = np.linalg.eigh(sent_sim_matrix)
    # print("KMeans: ", KMeans(n_clusters=14, init='k-means++').fit_predict(eigen_vectors[:, 2:4]))
    # print("DBSCAN: ",DBSCAN(min_samples=1).fit_predict(sent_sim_matrix))
    # clusters=AffinityPropagation(affinity='precomputed').fit_predict(sent_sim_matrix)
    # print(sent_sim_matrix)
    # print("Affinity Propagation:", clusters)
    # topic_n = 0
    # topic_list = [sentences[0] + '\n']

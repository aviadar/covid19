import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import pandas as pd
import tensorflow as tf
import tqdm


class SentenceUtil:
    def __init__(self, input_txt_list, k_clusters, save_input=False):
        # @param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        self.similarity = None
        self.kmeans = None
        self.model = hub.load(module_url)
        print("module %s loaded" % module_url)
        self._sentence_similarity(input_txt_list)
        self._cluster_sentences(k_clusters)
        self.input = None
        if save_input:
            self.input = input_txt_list

    def _embed(self, input_txt_list):
        self.embedding = self.model([input_txt_list.iloc[0]])
        for i, txt in enumerate(input_txt_list.iloc[1:]):
            print(i)
            try:
                self.embedding = tf.concat([self.embedding, self.model([txt])], 0)
            except Exception as e:
                print(e)

    def _sentence_similarity(self, input_txt_list):
        self._embed(input_txt_list)
        self.similarity = cosine_similarity(self.embedding)

    def _cluster_sentences(self, k):
        self.kmeans = KMeans(n_clusters=k, random_state=0).fit(self.embedding).labels_

    def get_k_most_similar(self, compared_index, k):
        topk_ind = self.similarity[compared_index, :].argsort()[-(k + 1):][::-1][1:]
        return topk_ind, self.similarity[compared_index, :][topk_ind]

    def plot_similarity(self, labels=None):
        mask = np.triu(np.ones_like(self.similarity, dtype=bool))
        sns.set(font_scale=1.2)
        g = sns.heatmap(
            self.similarity,
            mask=mask,
            vmin=0,
            vmax=1,
            cmap="YlOrRd")
        if labels:
            g.set_xticklabels(labels, rotation=90)
            g.set_yticklabels(labels, rotation=0)
        g.set_title("Semantic Textual Similarity")
        plt.show()

    def plot_clusters(self):
        embedding = MDS(n_components=2)
        mds = pd.DataFrame(embedding.fit_transform(self.embedding),
                           columns=['component1', 'component2'])
        mds['cluster'] = self.kmeans

        sns.scatterplot(data=mds, x="component1", y="component2", hue="cluster")
        if self.input:
            for i, txt in enumerate(self.input):
                plt.text(mds.component1[i], mds.component2[i], txt)
        plt.show()


def test_sentence_similarity():
    messages = [
        # Smartphones
        "I like my phone",
        "My phone is not good.",
        "Your cellphone looks great.",

        # Weather
        "Will it snow tomorrow?",
        "Recently a lot of hurricanes have hit the US",
        "Global warming is real",

        # Food and health
        "An apple a day, keeps the doctors away",
        "Eating strawberries is healthy",
        "Is paleo better than keto?",

        # Asking about age
        "How old are you?",
        "what is your age?",
    ]

    sentence_similarity = SentenceUtil(messages, k_clusters=4, save_input=True)
    sentence_similarity.get_k_most_similar(compared_index=2, k=3)
    sentence_similarity.plot_similarity(labels=messages)
    sentence_similarity.plot_clusters()


def test_sentence_similarity_2():
    df = pd.read_csv(r'C:\Users\aviadar\PycharmProjects\advanced_ml\covid19\covid_df.csv')

    sentence_similarity = SentenceUtil(df.abstract[:10], k_clusters=4)
    # sentence_similarity.get_k_most_similar(compared_index=2, k=3)
    # sentence_similarity.plot_similarity(labels=messages)
    # sentence_similarity.plot_clusters()

# test_sentence_similarity_2()

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


class SentenceSimilarity:
    def __init__(self, input_txt_list):
        # @param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        self.similarity = None
        self.model = hub.load(module_url)
        print("module %s loaded" % module_url)
        self._sentence_similarity(input_txt_list)

    def _embed(self, input_txt_list):
        self.embedding = self.model(input_txt_list)

    def _sentence_similarity(self, input_txt_list):
        self._embed(input_txt_list)
        self.similarity = cosine_similarity(self.embedding)

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

    sentence_similarity = SentenceSimilarity(messages)
    sentence_similarity.get_k_most_similar(compared_index=2, k=3)
    sentence_similarity.plot_similarity(labels=messages)


test_sentence_similarity()

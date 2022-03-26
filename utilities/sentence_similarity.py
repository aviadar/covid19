import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


class SentenceSimilarity:
    def __init__(self):
        # @param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        self.similarity = None
        self.model = hub.load(module_url)
        print("module %s loaded" % module_url)

    def _embed(self, input_txt_list):
        self.embedding = self.model(input_txt_list)

    def sentence_similarity(self, input_txt_list):
        self._embed(input_txt_list)
        self.similarity = cosine_similarity(self.embedding)

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
            g.set_yticklabels(labels, rotation=90)
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

    sentence_similarity = SentenceSimilarity()
    sentence_similarity.sentence_similarity(messages)
    sentence_similarity.plot_similarity(labels=messages)


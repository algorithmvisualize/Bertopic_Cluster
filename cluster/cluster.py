from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from .utils import *
import os
import torch
import random
import numpy as np


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


random_state = 10123
set_seed(random_state)
model_path = os.path.join(os.path.dirname(__file__), "pretrained_model/all-MiniLM-L6-v2")
limit_score = 0.15
umap_model_n_components = 32
vectorizer_model = CountVectorizer(stop_words="english")
tokenizer = vectorizer_model.build_tokenizer()


class Cluster:

    def __init__(self, texts: list[str], metric: str = 'euclidean', temperature: float = 0.2,
                 model_path: str = model_path, ):
        self.texts = [clean_text(t) for t in texts]
        self.umap_model = UMAP(random_state=random_state, n_neighbors=15,
                               n_components=umap_model_n_components if len(self.texts) - 2 > umap_model_n_components else len(self.texts) - 2, min_dist=0.0, metric=metric)
        self.hdbscan_model = HDBSCAN(min_cluster_size=2, alpha=1.0, metric=metric,
                                     cluster_selection_method="leaf"
                                     )

        self.texts_array = [tokenizer(t) for t in self.texts]

        self.ctfidf_model = ClassTfidfTransformer()
        self.embedding_model = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature  # future work
        self.topic_model = CustomBERTopic(
            embedding_model=self.embedding_model,  # Step 1 - Extract embeddings
            umap_model=self.umap_model,  # Step 2 - Reduce dimensionality
            hdbscan_model=self.hdbscan_model,  # Step 3 - Cluster reduced embeddings
            vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
            ctfidf_model=self.ctfidf_model,  # Step 5 - Extract topic words
            nr_topics="auto",
            top_n_words=5
        )

    def run(self):
        topics, probabilities = self.topic_model.fit_transform(self.texts)
        total_len = len(self.topic_model.topic_sizes_)
        topic_info = {i: self.topic_model.get_topic(i) for i in range(total_len - 1)}
        import os
        original_token = None
        if "TOKENIZERS_PARALLELISM" in os.environ:
            original_token = os.environ["TOKENIZERS_PARALLELISM"]
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        coherence_scores = compute_coherence_scores(self.topic_model, self.texts, vectorizer_model)
        topic_coherence = list(zip(range(len(coherence_scores)), coherence_scores))
        if not original_token:
            os.environ.pop("TOKENIZERS_PARALLELISM")

        sorted_topics_by_coherence = sorted(topic_coherence, key=lambda x: x[1], reverse=True)
        sorted_topics_by_coherence = [(process_same_words(topic_info[i]), j) for (i, j) in sorted_topics_by_coherence]
        return sorted_topics_by_coherence


def process_same_words(data: list[tuple[str, float]]):
    all_data = {}
    for d in data:
        temp = lemmatize_word(d[0])
        if temp in all_data:
            all_data[temp] += d[1]
        else:
            all_data[temp] = d[1]
    all_data = [(k, v) for k, v in all_data.items()]
    ret = [(k, v) for (k, v) in all_data if v > limit_score]
    if not ret:
        return all_data[:2]
    else:
        return ret

    # return all_data
class CustomBERTopic(BERTopic):
    def get_topic(self, topic=None, full: bool = False):
        """Rewrite get_topic method to remove the representative of an empty string"""
        topic_words = super().get_topic(topic)
        filtered_topic_words = [(word, score) for word, score in topic_words if word]
        scores = [score for _, score in topic_words]
        sum_scores = sum(scores)
        # filtered_topic_words = [(word, score / sum_scores) for word, score in filtered_topic_words if score / sum_scores >= limit_score]
        filtered_topic_words = [(word, score / sum_scores) for word, score in filtered_topic_words]

        return filtered_topic_words


def compute_coherence_scores(topic_model, texts, vectorizer_model, coherence_metric="c_v"):
    tokenizer = vectorizer_model.build_tokenizer()
    cleaned_texts = [tokenizer(text) for text in texts]
    dictionary = Dictionary(cleaned_texts)
    corpus = [dictionary.doc2bow(text) for text in cleaned_texts]
    topic_words = [
        [word for word, _ in topic_model.get_topic(topic)]
        for topic in range(len(topic_model.get_topic_freq()) - 1)
    ]
    coherence_model = CoherenceModel(topics=topic_words, texts=cleaned_texts, corpus=corpus, dictionary=dictionary,
                                     coherence=coherence_metric)
    return coherence_model.get_coherence_per_topic()

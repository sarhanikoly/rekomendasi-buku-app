import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RekomendasiBuku:
    def __init__(self, path_csv):
        self.df = pd.read_csv(path_csv)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['judul'])

    def rekomendasikan(self, judul_input, top_n=5):
        cosine_sim = cosine_similarity(self.vectorizer.transform([judul_input]), self.tfidf_matrix)
        scores = list(enumerate(cosine_sim[0]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        rekomendasi_idx = [i[0] for i in scores[1:top_n+1]]
        return self.df.iloc[rekomendasi_idx]
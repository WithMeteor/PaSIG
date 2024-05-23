import math
import copy


class TfIdf:
    def __init__(self):
        self.num_docs = 0
        self.vocab = {}
        self.vocab_temp = {}
        self.corpus = []

    def build_corpus(self, corpus):
        self.corpus = corpus
        self._merge_corpus(corpus)

    def get_tfidf(self):
        tfidf_list = []
        for sentence in self.corpus:
            tfidf_list.append(self._cal_sent_tfidf(sentence))
        return tfidf_list

    def add_sentence(self, sentence):
        self.vocab_temp = copy.copy(self.vocab)
        words = sentence.strip().split()
        words = set(words)
        for word in words:
            self.vocab_temp[word] = self.vocab_temp.get(word, 0.0) + 1.0

    def get_sentence_tfidf(self, sentence):
        tfidf = {}
        terms = sentence.strip().split()
        terms_set = set(terms)
        num_terms = len(terms)
        for term in terms_set:
            # 计算 TF 值
            tf = float(terms.count(term)) / num_terms
            # 计算 IDF 值，在实际实现时，可以提前将所有词的 IDF 提前计算好，然后直接使用。
            idf = math.log(self.num_docs / (self.vocab_temp.get(term, 0.0) + 1.0))
            # 计算 TF-IDF 值
            # tfidf[term] = tf * idf
            tfidf_value = round(tf * idf, 3)
            if tfidf_value == 0:
                tfidf_value = 0.001
            tfidf[term] = tfidf_value
        return tfidf

    def _merge_corpus(self, corpus):
        """
        统计语料库，输出词表，并统计包含每个词的文档数。
        """
        self.num_docs = len(corpus)
        for sentence in corpus:
            words = sentence.strip().split()
            words = set(words)
            for word in words:
                self.vocab[word] = self.vocab.get(word, 0.0) + 1.0

    def _cal_term_idf(self, term):
        """
        计算 IDF 值
        """
        return math.log(self.num_docs / (self.vocab.get(term, 0.0) + 1.0))

    def _cal_sent_tfidf(self, sentence):
        tfidf = {}
        terms = sentence.strip().split()
        terms_set = set(terms)
        num_terms = len(terms)
        for term in terms_set:
            # 计算 TF 值
            tf = float(terms.count(term)) / num_terms
            # 计算 IDF 值，在实际实现时，可以提前将所有词的 IDF 提前计算好，然后直接使用。
            idf = self._cal_term_idf(term)
            # 计算 TF-IDF 值
            # tfidf[term] = tf * idf
            tfidf_value = round(tf * idf, 3)
            if tfidf_value == 0:
                tfidf_value = 0.001
            tfidf[term] = tfidf_value
        return tfidf


if __name__ == '__main__':
    Corpus = [
        "What is the weather like today",
        "what is for dinner tonight",
        "this is question worth pondering",
        "it is a beautiful day today"
    ]

    MyTfIdf = TfIdf()
    MyTfIdf.build_corpus(Corpus)
    TfidfValues = MyTfIdf.get_tfidf()
    for tfidfValue in TfidfValues:
        print(tfidfValue)

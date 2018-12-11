#!/usr/bin/env python
# coding=utf-8
import numpy as np
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.decomposition  import TruncatedSVD
from sklearn.pipeline import Pipeline

class LSA:
    def __init__(self, documents):
        self.documents = documents

    def svd(self, dim=100):
        ## use idf then each element in document-words matrix is the tf-idf of words, smooth_idf is tf*log(N/df)
        vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
        ## n_components 为输出的维度，比特征维度要小，默认100
        svdmodel = TruncatedSVD(n_components=dim, algorithm='randomized', n_iter=10)
        svd_transformer = Pipeline([('tfidf', vectorizer),('svd', svdmodel)])
        svd_matrix = svd_transformer.fit_transform(self.documents)
        return svd_matrix



from gensim.corpora import MmCorpus
from gensim.models.ldamodel import LdaModel

class LDA:
    def __init__(self, documents, id2word=''):
        self.documents = documents
        # load id->word mapping (the dictionary)
        if id2word=='':
            self.id2word = load_from_text('wiki_en_wordids.txt')
        elif isinstance(id2word,dict):
            self.id2word = id2word
        else:
            import traceback
            traceback.print_exc('id2word error... type error')
            self.id2word = {}

    def ldamodel(self):
        # extract 100 LDA topics, updating once every 10,000
        lda = lda = LdaModel(corpus=self.documents, id2word=self.id2word, num_topics=100, update_every=1, chunksize=10000, passes=1)
        doc_bow = doc2bow(document.split())
        doc_lda = lda[doc_bow]
    # load corpus iterator



    # use LDA model: transform new doc to bag-of-words, then apply lda

    # doc_lda is vector of length num_topics representing weighted presence of each topic in the doc


text1 = '''
In linear algebra, the singular-value decomposition (SVD) is a factorization of a real or complex matrix. It is the generalization of the eigendecomposition of a positive semidefinite normal matrix (for example, a symmetric matrix with positive eigenvalues) to any {\displaystyle m\times n} m\times n matrix via an extension of the polar decomposition. It has many useful applications in signal processing and statistics.
'''
text2 = '''
Formally, the singular-value decomposition of an {\displaystyle m\times n} m\times n real or complex matrix {\displaystyle \mathbf {M} } \mathbf {M}  is a factorization of the form {\displaystyle \mathbf {U\Sigma V^{*}} } {\displaystyle \mathbf {U\Sigma V^{*}} }, where {\displaystyle \mathbf {U} } \mathbf {U}  is an {\displaystyle m\times m} m\times m real or complex unitary matrix, {\displaystyle \mathbf {\Sigma } } \mathbf{\Sigma} is an {\displaystyle m\times n} m\times n rectangular diagonal matrix with non-negative real numbers on the diagonal, and {\displaystyle \mathbf {V} } \mathbf {V}  is an {\displaystyle n\times n} n\times n real or complex unitary matrix. The diagonal entries {\displaystyle \sigma _{i}} \sigma _{i} of {\displaystyle \mathbf {\Sigma } } \mathbf{\Sigma} are known as the singular values of {\displaystyle \mathbf {M} } \mathbf {M} . The columns of {\displaystyle \mathbf {U} } \mathbf {U}  and the columns of {\displaystyle \mathbf {V} } \mathbf {V}  are called the left-singular vectors and right-singular vectors of {\displaystyle \mathbf {M} } \mathbf {M} , respectively.
'''
text3 = '''
A matrix satisfying the first condition of the definition is known as a generalized inverse. If the matrix also satisfies the second definition, it is called a generalized reflexive inverse. Generalized inverses always exist but are not in general unique. Uniqueness is a consequence of the last two conditions.
'''
text4 = '''
pLSA，即概率潜在语义分析，采取概率方法替代 SVD 以解决问题。其核心思想是找到一个潜在主题的概率模型，该模型可以生成我们在文档-术语矩阵中观察到的数据。特别是，我们需要一个模型 P(D,W)，使得对于任何文档 d 和单词 w，P(d,w) 能对应于文档-术语矩阵中的那个条目。
'''

text5 = '''
UK: Prince Charles spearheads British royal revolution. LONDON 1996-08-20; GERMANY: Historic Dresden church rising from WW2 ashes. DRESDEN, Germany 1996-08-21; GERMANY: Historic Dresden church rising from WW2 ashes. DRESDEN, Germany 1996-08-21
'''

def testLSA():
    lsa = LSA([text1, text2, text3, text4])
    print(lsa.svd(5))
# testLSA()

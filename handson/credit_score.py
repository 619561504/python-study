from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

import nltk
import urlextract
import re
from html import unescape
from collections import Counter


class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names) -> None:
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, **fit_params):
        return X[self.attribute_names].values


class MyLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        le = LabelEncoder()
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=1)
        rows, columns = X.shape

        f = None
        for i in range(columns):
            encoder_r = le.fit_transform(X[:, i])
            if f is None:
                f = encoder_r
            else:
                f = np.concatenate([f, encoder_r])
        return f


class MyLabelBinarizer(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names) -> None:
        self.attribute_names = attribute_names
        self.feature_names = []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        label_binarizer = LabelBinarizer()

        if len(X.shape) == 1:
            row_c, = X.shape
            column_c = 1
        else:
            row_c, column_c = X.shape

        feature_fix = None

        for column in range(column_c):
            ret = label_binarizer.fit_transform(X[:, column])
            if feature_fix is None:
                feature_fix = ret
            else:
                feature_fix = np.concatenate((feature_fix, ret), axis=1)
            attribute_name = self.attribute_names[column]
            if label_binarizer.y_type_ == "binary":
                self.feature_names.append(attribute_name)
            else:
                for clz in label_binarizer.classes_:
                    self.feature_names.append("%s_%s" % (attribute_name, clz))
        return feature_fix


def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)


def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except:
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)


urlextract = urlextract.URLExtract()
stemmer = nltk.PorterStemmer()


class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_header=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_number=True, stemming=True):
        self.strip_header = strip_header
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_number
        self.stemming = stemming

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transform = []
        for email in X:
            text = email_to_text(email)
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and urlextract is not None:
                urls = list(set(urlextract.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', "NUMBER", text)
            if self.remove_punctuation:
                text = re.sub(r"\W+", ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transform.append(word_counts)
        return X_transform


from scipy.sparse import csr_matrix


class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.most_common = most_common
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self

    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        print(csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1)).toarray())
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))


if __name__ == "__main__":
    # columns = ["f1", "f2", "f3"]
    # df = pd.DataFrame(data=[["none", "yes", "none"], ["guarantor", "no", "yes, registered under the customers name"],
    #                         ["co-Applicant", "yes", "qq"]], columns=columns)
    #
    # binarizer = MyLabelBinarizer(columns)
    # print(binarizer.fit_transform(df.values))
    # print(binarizer.feature_names)
    # "multipart({})".format(", ".join())
    cc = []
    cc.append(Counter("abdcasdfafda"))
    cc.append(Counter("adfasfdasfa"))
    vocab_transformer = WordCounterToVectorTransformer(10)
    vectors = vocab_transformer.fit_transform(cc)
    print(vectors.toarray())
    row = [0, 0, 1, 2, 2, 2]
    col = [0, 2, 2, 0, 1, 2]
    data = [1, 2, 3, 4, 5, 6]
    print(csr_matrix((data, (row, col)), shape=(3,3)).toarray())


from sklearn.base import BaseEstimator, TransformerMixin


class ValueReplace(BaseEstimator, TransformerMixin):

    def __init__(self, attributes, replacements={"Yes": 1, "No": 0}):
        self.replacements = replacements
        self.attributes = attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for attr in self.attributes:
            for k in self.replacements.keys():
                X[attr].replace(k, self.replacements[k], inplace=True)
        return X

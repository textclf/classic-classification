from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class NaiveBayesVectorizer(CountVectorizer):
    '''
    Label sensitive featurizer using the n-gram counts, 
    weighted by a log-ratio in à la Naive Bayes classification
    '''

    def __init__(self, analyzer='word', min_df=1, alpha=1.0, ngram_range=(1, 1), max_features=None):
        super(NaiveBayesVectorizer, self).__init__(analyzer=analyzer, min_df=min_df, ngram_range=ngram_range)
        self.max_features = max_features
        self.alpha = 1.0
    
    def _get_counts(self, X):
        '''
        Return a counter for an iterable of documents
        '''

        analyzer = super(NaiveBayesVectorizer, self).build_analyzer()
        if isinstance(X, basestring):
            X = [X]
            
        counts = Counter()
        for x in X:
            counts.update(analyzer(x))
            
        return counts
    
    def fit(self, X, y):
        ''' 
        Fits a NaiveBayesVectorizer
        
        Args:
        -----
            X: a list of documents
            y: a list or array of {1, 0} representing 
                positive and negative examples
        
        Returns:
        --------
            self
        
        Raises:
        -------
            None
        '''
        

        # seperate examples
        pos = []
        neg = []

        for doc, label in zip(X, y):
            if label == 0.0:
                neg.append(doc)
            elif label == 1.0:
                pos.append(doc)
            else:
                raise ValueError('Label: {} not recognized!'.format(label))
        
        pos_counts, neg_counts = self._get_counts(pos), self._get_counts(neg)
        all_counts = pos_counts + neg_counts
        
        # filter vocabulary
        _all_counts = {token: count for token, count in all_counts.most_common(self.max_features) if count >= self.min_df}
        idx_dict = {tok: i for i, (tok, _) in enumerate(all_counts.most_common(self.max_features))}

        
        num_tokens = len(idx_dict)

        p = np.ones(num_tokens) * self.alpha
        q = np.ones(num_tokens) * self.alpha

        for t in idx_dict:
            p[idx_dict[t]] += pos_counts[t]
            q[idx_dict[t]] += neg_counts[t]
        
        p /= np.abs(p).sum()
        q /= np.abs(q).sum()
        
        log_ratio = np.log(np.divide(p, q))
    
        self.all_counts = _all_counts
        self.idx_dict = idx_dict
        self.log_ratio = log_ratio

        return self
        
    def _transform_doc(self, doc):
        counts = self._get_counts(doc)        
        vec = np.zeros(len(self.idx_dict))
        for t in counts:
            if t in self.idx_dict:
                vec[self.idx_dict[t]] += counts[t]
        return vec * self.log_ratio
    
    def transform(self, X):
        if isinstance(X, basestring):
            X = [X]
        X_out = np.array([self._transform_doc(x) for x in X])
        return X_out

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

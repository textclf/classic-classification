from classic.classifiers import TextClassifier, NaiveBayesClassifier, SGDTextClassifier, \
    LogisticClassifier, SVMClassifier, PerceptronClassifier, RandomForestTextClassifier
from nlpdatahandlers import ImdbDataHandler

import sys

IMDB_DATA_DEFAULT = '../deep-text/datasets/aclImdb/aclImdb'

if __name__ == '__main__':

    if len(sys.argv) > 1 and sys.argv[1] != "":
        source = sys.argv[1]
    else:
        source = IMDB_DATA_DEFAULT

    print "Loading data from original source"
    imdb = ImdbDataHandler(source=source)
    (train_reviews, train_labels) = imdb.get_data(type=ImdbDataHandler.DATA_TRAIN, shuffle=True)
    (test_reviews, test_labels) = imdb.get_data(type=ImdbDataHandler.DATA_TEST, shuffle=True)

    # Simple bag of words with SGD
    sgd = SGDTextClassifier(train_reviews, train_labels,
                            test_texts=test_reviews, test_labels=test_labels,
                            compute_features=True)
    sgd.grid_search_cv(verbose=5, n_jobs=4)
    test_error = sgd.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    # Now with bigrams too
    sgd = SGDTextClassifier(train_reviews, train_labels, ngram_range=(1,2),
                            test_texts=test_reviews, test_labels=test_labels,
                            compute_features=True)
    sgd.grid_search_cv(verbose=5, n_jobs=4)
    test_error = sgd.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    # Simple bag of words with NB
    nb = NaiveBayesClassifier(train_reviews, train_labels,
                              test_texts=test_reviews, test_labels=test_labels)
    nb.set_bag_of_ngrams() # Also can compute bag of words manually
    nb.grid_search_cv(verbose=5, n_jobs=4)
    test_error = nb.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    # Now with bigrams too
    nb = NaiveBayesClassifier(train_reviews, train_labels, ngram_range=(1,2),
                              test_texts=test_reviews, test_labels=test_labels)
    nb.set_bag_of_ngrams() # Also can compute bag of words manually
    nb.grid_search_cv(verbose=5, n_jobs=4)
    test_error = nb.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    # Simple bag of words with Random Forest
    rf = RandomForestTextClassifier(train_reviews, train_labels,
                                    test_texts=test_reviews, test_labels=test_labels)
    rf.set_bag_of_ngrams() # We can compute bag of words manually
    rf.grid_search_cv(n_jobs=4, verbose=5)
    test_error = rf.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    # Now with bigrams too
    rf2 = RandomForestTextClassifier(train_reviews, train_labels, ngram_range=(1,2),
                             test_texts=test_reviews, test_labels=test_labels,
                             compute_features=True)
    rf2.grid_search_cv(n_jobs=4, verbose=5)
    test_error = rf2.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    # Simple bag of words with Support Vector Machines
    svm = SVMClassifier(train_reviews, train_labels,
                       test_texts=test_reviews, test_labels=test_labels,
                       compute_features=True)
    svm.grid_search_cv(n_jobs=4, verbose=5)
    test_error = svm.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    svm = SVMClassifier(train_reviews, train_labels, ngram_range=(1,2),
                       test_texts=test_reviews, test_labels=test_labels,
                       compute_features=True)
    svm.grid_search_cv(n_jobs=4, verbose=5)
    test_error = svm.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    # Simple bag of words with a logistic classifier
    lr = LogisticClassifier(train_reviews, train_labels,
                            test_texts=test_reviews, test_labels=test_labels,
                            compute_features=True)
    lr.grid_search_cv(verbose=5, n_jobs=4)
    test_error = lr.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    lr = LogisticClassifier(train_reviews, train_labels, ngram_range=(1,2),
                            test_texts=test_reviews, test_labels=test_labels,
                            compute_features=True)
    lr.grid_search_cv(verbose=5, n_jobs=4)
    test_error = lr.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    # SGD up to 3-grams
    sgd = SGDTextClassifier(train_reviews, train_labels, ngram_range=(1,2,3),
                            test_texts=test_reviews, test_labels=test_labels,
                            compute_features=True)
    sgd.grid_search_cv(verbose=5, n_jobs=4)
    test_error = sgd.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20
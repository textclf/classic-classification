from classic.classifiers import TextClassifier, NaiveBayesClassifier, SGDTextClassifier, \
    LogisticClassifier, SVMClassifier, PerceptronClassifier, RandomForestTextClassifier
from nlpdatahandlers import ImdbDataHandler

import sys

IMDB_DATA_DEFAULT = '../deep-text/datasets/aclImdb/aclImdb'

if __name__ == '__main__':

    print "Loading data from original source"
    imdb = ImdbDataHandler(source=IMDB_DATA_DEFAULT)
    (train_reviews, train_labels) = imdb.get_data(type=ImdbDataHandler.DATA_TRAIN, shuffle=True)
    (test_reviews, test_labels) = imdb.get_data(type=ImdbDataHandler.DATA_TEST, shuffle=True)

    print "Naive Bayes"
    nb = NaiveBayesClassifier()
    nb.set_training_data(train_reviews, train_labels)
    nb.set_test_data(test_reviews, test_labels)
    nb.set_bag_of_ngrams()

    nb.train()
    train_error = nb.get_training_error()
    test_error = nb.get_test_error()
    print "Training error: " + str(train_error)
    print "Test error: " + str(test_error)

    print "SGD Classifier"
    sgd = SGDTextClassifier(train_reviews, train_labels,
                            test_texts=test_reviews, test_labels=test_labels)
    #train_error = sgd.get_training_error()
    #test_error = sgd.get_test_error()
    #print "Training error: " + str(train_error)
    #print "Test error: " + str(test_error)
    sgd.set_bag_of_ngrams()
    sgd.grid_search_cv(verbose=0, n_jobs=4)


    print "Logistic classifier"
    sgd = LogisticClassifier()
    sgd.set_training_data(train_reviews, train_labels)
    sgd.set_test_data(test_reviews, test_labels)
    sgd.set_bag_of_ngrams()

    sgd.train()
    train_error = sgd.get_training_error()
    test_error = sgd.get_test_error()
    print "Training error: " + str(train_error)
    print "Test error: " + str(test_error)

    print "SVM classifier"
    sgd = SVMClassifier()
    sgd.set_training_data(train_reviews, train_labels)
    sgd.set_test_data(test_reviews, test_labels)
    sgd.set_bag_of_ngrams()

    sgd.train()
    train_error = sgd.get_training_error()
    test_error = sgd.get_test_error()
    print "Training error: " + str(train_error)
    print "Test error: " + str(test_error)

    print "Perceptron classifier"
    sgd = PerceptronClassifier()
    sgd.set_training_data(train_reviews, train_labels)
    sgd.set_test_data(test_reviews, test_labels)
    sgd.set_bag_of_ngrams()

    sgd.train()
    train_error = sgd.get_training_error()
    test_error = sgd.get_test_error()
    print "Training error: " + str(train_error)
    print "Test error: " + str(test_error)

    print "Random forest classifier"
    sgd = RandomForestTextClassifier()
    sgd.set_training_data(train_reviews, train_labels)
    sgd.set_test_data(test_reviews, test_labels)
    sgd.set_bag_of_ngrams()

    sgd.train()
    train_error = sgd.get_training_error()
    test_error = sgd.get_test_error()
    print "Training error: " + str(train_error)
    print "Test error: " + str(test_error)
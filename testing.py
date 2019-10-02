import preprocessing
import training
import math
import sys

'''
Preprocessing the Training Data:
Step 1: Add the <s> and </s> tags and append to a list
Step 2: Generate a python dict with above list
Step 3: Replace all tokens with (1) count with <unk> tag

For code, see preprocessing.py
'''
# Filepath to the training data
train_fp = 'data/brown-train.txt'

# Preprocessing the training data
# Build a dictionary from the preprocessed data
train_list = preprocessing.preprocess_train(train_fp)
train_dict = preprocessing.build_dict(train_list)

'''
Training the following models:
1. Unigram Maximum Likelihood Model
2. Bigram Maximum Likelihood Model
3. Bigram Model with Add-One Smoothing
4. Bigram Model with Discounting and Katz backoff

For code, see training.py
'''


# Unigram Model
# unigram_probs = training.unigram_train(train_list, train_dict)
# test_list must be preprocessed
def unigram_predict(test_list, unigram_probs):
    log_prob = 0

    for token in test_list:
        token_prob = math.log(unigram_probs[token], 2)
        print('The unigram log probability for', token,
              'is', round(token_prob, 3))
        log_prob += token_prob

    return log_prob/len(test_list)


# For larger data sets
def unigram_predict_big(test_list, unigram_probs):
    log_prob = 0

    for token in test_list:
        token_prob = math.log(unigram_probs[token], 2)
        log_prob += token_prob

    return log_prob/len(test_list)


# Bigram Model
# bigram_counts = bigram_dict(train_list)
# bigram_probs = training.bigram_train(bigram_counts, train_list, train_dict)
# len_list = len(train_list)
def bigram_predict(bigram_list, bigram_probs, len_list):
    log_prob = 0
    zero_prob = False

    for bigram in bigram_list:
        try:
            token_prob = math.log(bigram_probs[bigram], 2)
            print('The bigram log probability for', bigram, 'is', round(token_prob, 3))
            log_prob += token_prob
        except KeyError:
            print('This is an unobserved bigram. The probability for', bigram, 'is 0.')
            zero_prob = True
            log_prob = 0

    if zero_prob:
        return 0
    else:
        return log_prob/len_list


def bigram_predict_no_print(bigram_list, bigram_probs, len_list):
    log_prob = 0
    zero_prob = False

    for bigram in bigram_list:
        try:
            token_prob = math.log(bigram_probs[bigram], 2)
            log_prob += token_prob
        except KeyError:
            zero_prob = True
            log_prob = -sys.maxsize

    if zero_prob:
        return -sys.maxsize
    else:
        return log_prob/len_list


# For larger data sets
def bigram_predict_big(filepath, bigram_probs, train_dict, len_list):
    file = open(filepath, 'r')
    log_prob = 0

    for line in file:
        line_list = preprocessing.sentence_preprocess(line, lst=[])
        unk_list = preprocessing.unkify_test(line_list, train_dict)
        line_bigrams = training.bigrams(unk_list)
        line_prob = bigram_predict_no_print(line_bigrams, bigram_probs, len_list)
        log_prob += line_prob

    file.close()

    return log_prob

# Bigram Model with Add-One Smoothing
# len_dict = len(train_dict)
# add_one_probs = training.add_one_train(train_bdict, train_list, train_dict, len_dict)
def add_one_predict(bigram_list, add_one_probs, dct, len_list):
    log_prob = 0

    for bigram in bigram_list:
        if bigram in add_one_probs:
            token_prob = math.log(add_one_probs[bigram], 2)
            print('The add-one log probability for', bigram,
                  'is', round(token_prob, 3))
            log_prob += token_prob
        else:
            token_prob = math.log(1/(dct[bigram[0]] + len(dct)), 2)
            print('The add-one log probability for', bigram,
                  'is', round(token_prob, 3))
            log_prob += token_prob

    return log_prob/len_list


def add_one_predict_no_print(bigram_list, add_one_probs, dct, len_list):
    log_prob = 0

    for bigram in bigram_list:
        if bigram in add_one_probs:
            token_prob = math.log(add_one_probs[bigram], 2)
            log_prob += token_prob
        else:
            token_prob = math.log(1/(dct[bigram[0]] + len(dct)), 2)
            log_prob += token_prob

    return log_prob/len_list


# For larger data sets
def add_one_predict_big(filepath, add_one_probs, train_dict, len_list):
    file = open(filepath, 'r')
    log_prob = 0

    for line in file:
        line_list = preprocessing.sentence_preprocess(line, lst=[])
        unk_list = preprocessing.unkify_test(line_list, train_dict)
        line_bigrams = training.bigrams(unk_list)
        line_prob = add_one_predict_no_print(line_bigrams, add_one_probs,
                                             train_dict, len_list)
        log_prob += line_prob

    file.close()

    return log_prob


# Bigram Model with Discounting and Katz Backoff
# len_train = len(train_list)
# len_list = len(test_list)
def katz_predict(bigram_list, katz_probs, unigrams,
                 bigrams_A, len_train, len_list):
    log_prob = 0

    for bigram in bigram_list:
        if bigram in katz_probs:
            token_prob = math.log(katz_probs[bigram], 2)
            print('The Katz log probability for', bigram,
                  'is', round(token_prob, 3))
            log_prob += token_prob
        else:
            katz_alpha = training.katz_alpha(bigram[0], bigrams_A, unigrams)
            prob_mass_B = len_train - sum(bigrams_A[bigram[0]].values()) - 1
            token_prob = math.log(katz_alpha *
                                  unigrams[bigram[1]] / prob_mass_B)
            print('The Katz log probability for', bigram,
                  'is', round(token_prob, 3))
            log_prob += token_prob

    return log_prob/len_list


# For larger data sets
def katz_predict_big(bigram_list, katz_probs, unigrams,
                     bigrams_A, len_train, len_list):
    log_prob = 0

    for bigram in bigram_list:
        if bigram in katz_probs:
            token_prob = math.log(katz_probs[bigram], 2)
            log_prob += token_prob
        else:
            katz_alpha = training.katz_alpha(bigram[0], bigrams_A, unigrams)
            prob_mass_B = len_train - sum(bigrams_A[bigram[0]].values()) - 1
            token_prob = math.log(katz_alpha *
                                  unigrams[bigram[1]] / prob_mass_B)
            log_prob += token_prob

    return log_prob/len_list


'''
Perplexity
'''


def perplexity(log_prob):
    try:
        return 2 ** (-log_prob)
    except OverflowError:
        return 'Too Large'


'''
Preprocessing the Test Data:
Step 1: Add the <s> and </s> tags and append to a list
Step 2: Generate a python dict with above list
Step 3: Replace all unseen tokens with <unk> token

For code, see preprocessing.py
'''
# Filepaths to the test data
test_fp = 'data/brown-test.txt'
learner_fp = 'data/learner-test.txt'

# Preprocessing the training data
test_list = preprocessing.preprocess_test(test_fp, train_dict)
learner_list = preprocessing.preprocess_test(learner_fp, train_dict)


'''
Analyzing the Test Data
'''


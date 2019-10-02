import preprocessing
import training
import math
import sys

'''
Unigram Maximum Likelihood Model
'''


# prints the unigram probability of each sentence
def unigram_predict(test_list, unigram_probs):
    log_prob = 0

    for token in test_list:
        token_prob = math.log(unigram_probs[token], 2)
        print('The unigram log probability for', token,
              'is', round(token_prob, 3))
        log_prob += token_prob

    return log_prob/len(test_list)


# For larger data sets
# removes the print statements from above
def unigram_predict_no_print(test_list, unigram_probs):
    log_prob = 0

    for token in test_list:
        token_prob = math.log(unigram_probs[token], 2)
        log_prob += token_prob

    return log_prob/len(test_list)


'''
Bigram Maximum Likelihood Model


'''


# len_list is the size of the test corpus
# also prints the probabilities of each sentence
def bigram_predict(bigram_list, bigram_probs, len_list):
    log_prob = 0
    zero_prob = False

    for bigram in bigram_list:
        try:
            token_prob = math.log(bigram_probs[bigram], 2)
            print('The bigram log probability for', bigram,
                  'is', round(token_prob, 3))
            log_prob += token_prob
        except KeyError:
            print('This is an unobserved bigram. The probability for',
                  bigram, 'is 0.')
            zero_prob = True
            log_prob = 0

    if zero_prob:
        return 0
    else:
        return log_prob/len_list


# Removes the print statements from bigram_predict
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
# Because of the problem of taking logs of zero probabilities
# I created code to take the probabilities of each sentence
# Just in case I needed to see which were zero and which were not
def bigram_predict_file(filepath, bigram_probs, train_dict, len_list):
    file = open(filepath, 'r')
    log_prob = 0

    for line in file:
        line_list = preprocessing.sentence_preprocess(line, lst=[])
        unk_list = preprocessing.unkify_test(line_list, train_dict)
        line_bigrams = training.bigrams(unk_list)
        line_prob = bigram_predict_no_print(line_bigrams, bigram_probs,
                                            len_list)
        log_prob += line_prob

    file.close()

    return log_prob


'''
Bigram Model with Add-One Smoothing
'''


# dct is the unigram counts of the training data
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


# For larger data sets
# removes the print statements from above
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


'''
Bigram Model with Discounting and Katz Backoff
'''


# unigrams are the unigram counts of set A
# len_train is the size of the training data
# len_list is the size of the test data
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
# removes the print statements from aboev
def katz_predict_no_print(bigram_list, katz_probs, unigrams,
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
Perplexity:
straight-forward calculation
'''


def perplexity(log_prob):
    try:
        return 2 ** (-log_prob)
    except OverflowError:
        return 'Too Large or Infinite'

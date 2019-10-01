import preprocessing


train_fp = 'data/brown-small.txt'
w_list = preprocessing.add_s_tag(train_fp)
w_dict = preprocessing.build_dict(w_list)
w_list_unk = preprocessing.replace_unk(w_list, w_dict)
w_dict_unk = preprocessing.build_dict(w_list_unk)
w_count = len(w_list_unk)

'''
Unigram Maximum Likelihood Model
'''


def unigram_prob(token, dct, len_list):
    return float(dct[token])/len_list


def unigram_train(lst, dct):
    len_list = len(lst)
    unigram_dict = {}
    for token in lst:
        if token not in unigram_dict:
            unigram_dict[token] = unigram_prob(token, dct, len_list)
        else:
            continue
    return unigram_dict


'''
Bigram Maximum Likelihood Model
'''


def bigram_count(lst):
    bigram_dict = {}
    for i in range(1, len(lst)):
        if lst[i] == '<s>':
            continue
        else:
            if (lst[i-1], lst[i]) not in bigram_dict:
                bigram_dict[(lst[i-1], lst[i])] = 1
            elif (lst[i-1], lst[i]) in bigram_dict:
                bigram_dict[(lst[i-1], lst[i])] += 1
    return bigram_dict


# bigram_counts = bigram_count(w_list_unk)
def bigram_prob(bigram_counts, prior, target, dct):
    return float(bigram_counts[(prior, target)])/dct[prior]


def bigram_train(bigram_counts, lst, dct):
    bigram_probs = {}
    for i in range(1, len(lst)):
        if lst[i] == '<s>':
            continue
        else:
            if (lst[i-1], lst[i]) not in bigram_probs:
                bigram_probs[(lst[i-1], lst[i])] = (
                    bigram_prob(bigram_counts, lst[i-1], lst[i], dct))
            # elif (lst[i-1], lst[i]) in bigram_probs:
            #     continue
    return bigram_probs


'''
Bigram Model with Add-One Smoothing
'''


# len_dict = len(bigram_count(w_list_unk))
def add_one_prob(bigram_counts, prior, target, dct, len_dict):
    return float((bigram_counts[(prior, target)]+1))/(dct[prior]+len_dict)


def add_one_train(bigram_counts, lst, dct, len_dict):
    add_one_probs = {}
    for i in range(1, len(lst)):
        if lst[i] == '<s>':
            continue
        else:
            if (lst[i-1], lst[i]) not in add_one_probs:
                add_one_probs[(lst[i-1], lst[i])] = (add_one_prob(
                    bigram_counts, lst[i-1], lst[i], dct, len_dict))
            # elif (lst[i-1], lst[i]) in add_one_probs:
            #     continue
    return add_one_probs


'''
Bigram Model with Discounting and Katz backoff
'''


# # katz_A = preprocessing.build_dict(preprocessing.add_s_tag(w_list))
# def dc_katz_A(katz_A):
#     for key in katz_A:
#         katz_A[key] -= 0.5
#     return katz_A
def set_A(train_list):
    unigrams_A = {train_list[0]: 1}
    bigrams_A = {}

    for i in range(1, len(train_list)):
        if train_list[i] not in unigrams_A:
            unigrams_A[train_list[i]] = 1
        elif train_list[i] in unigrams_A:
            unigrams_A[train_list[i]] += 1
        if train_list[i] not in bigrams_A:
            bigrams_A[train_list[i]] = {train_list[i-1]: 1}
        elif (train_list[i] in bigrams_A and
              train_list[i] != '<s>'):
            if train_list[i-1] in bigrams_A[train_list[i]]:
                bigrams_A[train_list[i]][train_list[i-1]] += 1
            else:
                bigrams_A[train_list[i]] = {train_list[i-1]: 1}

    return unigrams_A, bigrams_A


# katz_A = dc_katz_A(katz_A)
# test_list = preprocessing.add_s_tag(testfile)
def set_B(bigrams_A, test_list):
    unigrams_B = {}
    bigrams_B = {}

    if test_list[0] not in bigrams_A:
        unigrams_B = {test_list[0]: 1}

    for i in range(1, len(test_list)):
        if test_list[i] in bigrams_A:
            continue
        elif test_list[i] not in bigrams_A:
            if test_list[i] not in bigrams_B:
                unigrams_B[test_list[i]] = 1
                bigrams_B[test_list[i]] = {test_list[i-1]: 1}
            elif test_list[i] in bigrams_B:
                unigrams_B[test_list[i]] += 1
                if (test_list[i-1] not in bigrams_B[test_list[i]] and
                        test_list[i] != '<s>'):
                    bigrams_B[test_list[i]] = {test_list[i-1]: 1}
                elif test_list[i-1] in bigrams_B[test_list[i]]:
                    bigrams_B[test_list[i]][test_list[i-1]] += 1

    return unigrams_B, bigrams_B


def katz_alpha(prior, bigrams_A, unigram_counts, discount=0.5):
    dc_sum = 0
    for key in bigrams_A:
        if key[0] == prior:
            dc_sum += bigrams_A[key] - discount
    return 1 - dc_sum/unigram_counts[prior]


def katz_backoff(prior, target, bigrams_A, unigrams_A, bigrams_B, unigrams_B):
    if target in bigrams_A:
        return (bigrams_A[target][prior] - 0.5)/unigrams_A[prior]
    else:
        alpha = katz_alpha(prior, bigrams_A, unigrams_B)
        return alpha * (float(unigrams_B[target])/sum(unigrams_B.values()))

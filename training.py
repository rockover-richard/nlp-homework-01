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
    return unigram_dict


'''
Bigram Maximum Likelihood Model
'''


def bigrams(lst):
    bigram_list = []
    for i in range(len(lst)-1):
        if lst[i] != '</s>':
            bigram_list.append((lst[i], lst[i+1]))
    return bigram_list


def bigram_dict(lst):
    bigram_dict = {}
    for i in range(1, len(lst)):
        if lst[i] == '<s>':
            continue
        elif (lst[i-1], lst[i]) not in bigram_dict:
            bigram_dict[(lst[i-1], lst[i])] = 1
        elif (lst[i-1], lst[i]) in bigram_dict:
            bigram_dict[(lst[i-1], lst[i])] += 1
    return bigram_dict


def bigram_prob(bigram_counts, prior, target, dct):
    return float(bigram_counts[(prior, target)])/dct[prior]


def bigram_train(bigram_counts, lst, dct):
    bigram_probs = {}
    for i in range(1, len(lst)):
        if lst[i] == '<s>':
            continue
        elif (lst[i-1], lst[i]) not in bigram_probs:
            bigram_probs[(lst[i-1], lst[i])] = (
                bigram_prob(bigram_counts, lst[i-1], lst[i], dct))
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
        elif (lst[i-1], lst[i]) not in add_one_probs:
            add_one_probs[(lst[i-1], lst[i])] = (add_one_prob(
                bigram_counts, lst[i-1], lst[i], dct, len_dict))
    return add_one_probs


'''
Bigram Model with Discounting and Katz Backoff
'''


# Set A is built as a nested dictionary:
# bigrams_A[prior][target] = counts of (prior, target)
def build_set_A(train_list):
    unigrams = {train_list[-1]: 1}
    bigrams_A = {}

    for i in range(len(train_list)-1):
        if train_list[i] not in unigrams:
            unigrams[train_list[i]] = 1
        elif train_list[i] in unigrams:
            unigrams[train_list[i]] += 1
        if train_list[i] not in bigrams_A:
            bigrams_A[train_list[i]] = {train_list[i+1]: 1}
        elif (train_list[i] in bigrams_A and
              train_list[i] != '</s>'):
            if train_list[i+1] in bigrams_A[train_list[i]]:
                bigrams_A[train_list[i]][train_list[i+1]] += 1
            else:
                bigrams_A[train_list[i]][train_list[i+1]] = 1

    return unigrams, bigrams_A


def katz_alpha(prior, bigrams_A, unigrams, discount=0.5):
    dc_sum = 0

    for key in bigrams_A:
        if key == prior:
            for token in bigrams_A[key]:
                dc_sum += bigrams_A[key][token] - discount

    return 1 - dc_sum/unigrams[prior]


def katz_backoff_prob(prior, target, unigrams, bigrams_A, len_list):
    if prior in bigrams_A:
        return (bigrams_A[prior][target] - 0.5)/unigrams[prior]
    else:
        prob_mass_B = len_list - sum(bigrams_A[prior].values()) - 1
        return (katz_alpha(prior, bigrams_A, unigrams) *
                (float(unigrams[target]) / prob_mass_B))


def katz_backoff(lst, unigrams, bigrams_A, len_list):
    katz_probs = {}
    for i, token in enumerate(lst):
        if token == '<s>':
            continue
        else:
            katz_probs[(lst[i-1], token)] = katz_backoff_prob(lst[i-1],
                                                              token,
                                                              unigrams,
                                                              bigrams_A,
                                                              len_list)
    return katz_probs

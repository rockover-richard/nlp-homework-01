import preprocessing


train_fp = 'data/brown-small.txt'
w_list = preprocessing.add_s_tag(train_fp)
w_dict = preprocessing.build_dict(w_list)
w_list_unk = preprocessing.replace_unk(w_list, w_dict)
w_dict_unk = preprocessing.build_dict(w_list_unk)
w_count = len(w_list_unk)


def unigram_prob(token):
    return float(w_dict[token])/len(w_list)


bigrams = preprocessing.bigram_count(w_list_unk)


def bigram_prob(target, prior):
    return float(bigrams[(target, prior)])/w_dict_unk[prior]


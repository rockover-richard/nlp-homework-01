'''
Preprocessing functions

'''


def sentence_preprocess(s, lst=[]):
    lst.append('<s>')
    for token in s.split():
        lst.append(token.lower())
    lst.append('</s>')
    return lst


# Adds the start of sentence and end of sentence tags of txt file
# This function creates a list
def add_s_tag(filepath):
    file = open(filepath, 'r')
    lst = []

    for line in file:
        sentence_preprocess(line, lst)

    file.close()
    return lst


# Builds a python dictionary with counts
def build_dict(lst):
    dct = {}
    for token in lst:
        if token not in dct:
            dct[token] = 1
        elif token in dct:
            dct[token] += 1
    return dct


# Replaces words that occur once in the corpus with <unk> token
def unkify_train(train_list, train_dict):
    for i, token in enumerate(train_list):
        if train_dict[token] == 1:
            train_list[i] = '<unk>'
    return train_list


# Replaces unseen words in test set with <unk> token
def unkify_test(test_list, train_dict):
    for i, token in enumerate(test_list):
        if token not in train_dict:
            test_list[i] = '<unk>'
    return test_list


# Complete preprocessing for training data
def preprocess_train(filepath):
    lst = add_s_tag(filepath)
    dct = build_dict(lst)
    unk_list = unkify_train(lst, dct)

    return unk_list


# Complete preprocessing for test data
def preprocess_test(filepath, train_dict):
    lst = add_s_tag(filepath)
    unk_list = unkify_test(lst, train_dict)

    return unk_list

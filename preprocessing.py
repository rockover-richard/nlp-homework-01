'''
Preprocessing functions for Assignment 1

'''


# Adds the start of sentence and end of sentence tags
def add_s_tag(filepath):
    file = open(filepath, 'r')
    w_list = []
    for line in file:
        w_list.append('<s>')
        for word in line.split():
            w_list.append(word.lower())
        w_list.append('</s>')
    file.close()
    return w_list


# Builds a python dictionary with counts
def build_dict(w_list):
    w_dict = {}
    for word in w_list:
        if word not in w_dict:
            w_dict[word] = 1
        elif word in w_dict:
            w_dict[word] += 1
    return w_dict


# Replaces words that occur once in the corpus with <unk>
def replace_unk(w_list, w_dict):
    for i, word in enumerate(w_list):
        if w_dict[word] == 1:
            w_list[i] = '<unk>'
        else:
            continue
    return w_list


def bigram_count(w_list):
    bigrams = {}
    for i in range(1, len(w_list)):
        if w_list[i] == '<s>':
            continue
        else:
            if (w_list[i-1], w_list[i]) not in bigrams:
                bigrams[(w_list[i-1], w_list[i])] = 1
            elif (w_list[i-1], w_list[i]) in bigrams:
                bigrams[(w_list[i-1], w_list[i])] += 1
    return bigrams


# s_list = []
# for ind, line in enumerate(file):
#     s_list.append(line.split())
#     w_list.append('<s>')
#     for i, word in enumerate(s_list[ind]):
#         s_list[ind][i] = word.lower()
#         w_list.append(word.lower())
#     s_list[ind].insert(0, '<s>')
#     s_list[ind].append('</s>')
#     w_list.append('</s>')
#     if ind == 2:
#         break

# print(s_list)

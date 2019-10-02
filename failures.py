# Old preprocessing
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

# Old build_set_B
# if train_list[i] in bigrams_A:
#     continue
# elif train_list[i] not in bigrams_A:
#     if test_list[i] not in bigrams_B:
#         unigrams_B[test_list[i]] = 1
#         bigrams_B[test_list[i]] = {test_list[i-1]: 1}
#     elif test_list[i] in bigrams_B:
#         unigrams_B[test_list[i]] += 1
#         if (test_list[i-1] not in bigrams_B[test_list[i]] and
#                 test_list[i] != '<s>'):
#             bigrams_B[test_list[i]] = {test_list[i-1]: 1}
#         elif test_list[i-1] in bigrams_B[test_list[i]]:
#             bigrams_B[test_list[i]][test_list[i-1]] += 1


# Set B is built as a nested dictionary:
# bigrams_B[prior][target] = counts of (prior, target)
# This is O(n ** 2)
def build_set_B(bigrams_A, train_list):
    bigrams_B = {}

    for i in range(len(train_list)-1):
        if train_list[i] not in bigrams_B:
            bigrams_B[train_list[i]] = {}
        for j in range(1, len(train_list)):
            if train_list[j] in bigrams_A[train_list[i]]:
                continue
            elif train_list[j] not in bigrams_B[train_list[i]]:
                bigrams_B[train_list[i]][train_list[j]] = 1
            elif train_list[j] in bigrams_B[train_list[i]]:
                bigrams_B[train_list[i]][train_list[j]] += 1
        print(i, end=" ")

    return bigrams_B


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


# Filepaths to the test data
test_fp = 'data/brown-test.txt'
learner_fp = 'data/learner-test.txt'

# Preprocessing the training data
test_list = preprocessing.preprocess_test(test_fp, train_dict)
learner_list = preprocessing.preprocess_test(learner_fp, train_dict)


'''
Analyzing the Test Data
'''

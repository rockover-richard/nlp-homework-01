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

filename = 'data/brown-small.txt'

file = open(filename, 'r')
s_list = []
w_list = []
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

for line in file:
    w_list.append('<s>')
    for word in line.split():
        w_list.append(word.lower())
    w_list.append('</s>')

print(w_list)
print(s_list)
file.close()

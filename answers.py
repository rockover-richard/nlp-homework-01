'''
Homework 1
Byung Kim
for CS74040: NLP Fall 2019 by Prof. Alla Rozovskaya
Due: 10/03/2019

Answers to the Questions
'''

# Import all necessary modules
import preprocessing
import training
import testing

# Filepaths to the data
train_fp = 'data/brown-train.txt'
test_fp = 'data/brown-test.txt'
learner_fp = 'data/learner-test.txt'


print('Preprocessing Training Data...')

train_list = preprocessing.preprocess_train(train_fp)
train_dict = preprocessing.build_dict(train_list)

print('Finished!')

input('Press ENTER to continue to Question 1!\n')

'''
Question 1
How many word types (unique words) are there in the training corpus?
Please include the padding symbols and the unknown token.
'''

print('Question 1:')
print('There are', len(train_dict), 'word types in the training corpus.')

input('\nPress ENTER to continue to Question 2!\n')

'''
Question 2
How many word tokens are there in the training corpus?
'''

print('Question 2:')
print('There are', len(train_list), 'word tokens in the training corpus.')


input('\nPress ENTER to continue to Question 3!\n')

'''
Question 3
What percentage of word tokens and word types in each of the test corpora
did not occur in training (before you mapped the unknown words to <unk>
in training and test data)?
'''

print('Question 3:')
print('First, I preprocess the training and test data,')
print('but do not replace any word with <unk>.\n')

print('Adding <s>, </s> tags and creating dictionaries...')
train_temp = preprocessing.add_s_tag(train_fp)
test_temp = preprocessing.add_s_tag(test_fp)
learner_temp = preprocessing.add_s_tag(learner_fp)

train_tdict = preprocessing.build_dict(train_temp)
test_tdict = preprocessing.build_dict(test_temp)
learner_tdict = preprocessing.build_dict(learner_temp)
print('Preprocessing finished!\n')


# For word tokens in test data but not in training data
def unseen_tokens_perc(test_list, train_dict):
    unseen_sum = 0
    for token in test_list:
        if token not in train_dict:
            unseen_sum += 1
    return float(unseen_sum)/len(test_list)


# For word types in test data but not in training data
def unseen_type_perc(test_dict, train_dict):
    unseen_sum = 0
    for key in test_dict:
        if key not in train_dict:
            unseen_sum += 1
    return float(unseen_sum)/len(test_dict)


print('Using functions to calculate the percentages:')
print('For the brown-test data:')
print('The percentage of tokens not in the training data for brown-test is: \
      {:.2%}'.format(unseen_tokens_perc(test_temp, train_tdict)))
print('The percentage of word types not in the training data for brown-test is: \
      {:.2%}'.format(unseen_type_perc(test_tdict, train_tdict)))
print('----------------------------------------------------------------------')
print('For the learner-test data:')
print('The percentage of tokens not in the training data for learner-test is: \
      {:.2%}'.format(unseen_tokens_perc(learner_temp, train_tdict)))
print('The percentage of word types not in the training data for learner-test is: \
      {:.2%}'.format(unseen_type_perc(learner_tdict, train_tdict)))

input('\nPress ENTER to continue to Question 4!\n')

'''
Question 4
What percentage of bigrams (bigram types and bigram tokens) in each of the test
corpora that did not occur in training
(treat <unk> as a token that has been observed).
'''

print('Question 4:')

print('First, I preprocess all the data...')

# Training data was already preprocessed in Question 1
test_list = preprocessing.preprocess_test(test_fp, train_dict)
learner_list = preprocessing.preprocess_test(learner_fp, train_dict)
print('Preprocessing finished!\n')

print('Then, I build a list of bigrams for all the data,')
print('as well as bigram dictionary with counts...')

# The bigram lists for all three data sets
train_bigrams = training.bigrams(train_list)
test_bigrams = training.bigrams(test_list)
learner_bigrams = training.bigrams(learner_list)

# And the bigram dictionary with counts for all three data sets
train_bdict = training.bigram_dict(train_list)
test_bdict = training.bigram_dict(test_list)
learner_bdict = training.bigram_dict(learner_list)
print('Bigram processing finished!\n')

print('Using the code from Question 3...')
print('For the brown-test data:')
print('The percentage of bigram tokens not in the training data for brown-test is: \
      {:.2%}'.format(unseen_tokens_perc(test_bigrams, train_bdict)))
print('The percentage of bigram types not in the training data for brown-test is: \
      {:.2%}'.format(unseen_type_perc(test_bdict, train_bdict)))
print('----------------------------------------------------------------------')
print('For the learner-test data:')
print('The percentage of bigram tokens not in the training data for learner-test \
       is: {:.2%}'.format(unseen_tokens_perc(learner_bigrams, train_bdict)))
print('The percentage of bigram types not in the training data for learner-test \
       is: {:.2%}'.format(unseen_type_perc(learner_bdict, train_bdict)))

input('\nPress ENTER to continue to Question 5 and 6!\n')

'''
Questions 5 and 6
Compute the log probabilities of the following sentences under the three models
(ignore capitalization and pad each sentence as described above).
Please list all of the parameters required to compute the probabilities and
show the complete calculation. Which of the parameters have zero values
under each model?

    He was laughed off the screen .
    There was no compulsion behind them .
    I look forward to hearing your reply .

Compute the perplexities of each of the sentences above
under each of the models.
'''

print('Questions 5 and 6 are answered together per model,')
print('and please see the write up for the parameters.\n')

print('First, I preprocess the three sentences...')
s1 = 'He was laughed off the screen .'
s2 = 'There was no compulsion behind them .'
s3 = 'I look forward to hearing your reply .'

# Preprocessing using the sentence_preprocess function
s1_list = preprocessing.sentence_preprocess(s1, lst=[])
s2_list = preprocessing.sentence_preprocess(s2, lst=[])
s3_list = preprocessing.sentence_preprocess(s3, lst=[])

# Using unkify_test to add <unk> where needed
s1_unk = preprocessing.unkify_test(s1_list, train_dict)
s2_unk = preprocessing.unkify_test(s2_list, train_dict)
s3_unk = preprocessing.unkify_test(s3_list, train_dict)

print(s1_unk)
print(s2_unk)
print(s3_unk)
print('Preprocessing finished!\n')

input('Press Enter to continue to Unigram ML\n')

print('UNIGRAM MAXIMUM LIKELIHOOD MODEL\n')
print('Training the model with brown-train.txt...')
unigram_probs = training.unigram_train(train_list, train_dict)
print('Training finished!\n')

print('Sentence 1:')
s1_predict = testing.unigram_predict(s1_unk, unigram_probs)
print('The log probability of sentence 1 in the Unigram ML Model is: \
      {:.3f}'.format(s1_predict))
print('The perplexity of sentence 1 in the Unigram ML Model is: {:.1f}'.format(
      testing.perplexity(s1_predict)))
print('---------------------------------------------------------------------')
print('Sentence 2:')
s2_predict = testing.unigram_predict(s2_unk, unigram_probs)
print('The log probability of sentence 2 in the Unigram ML Model is: \
      {:.3f}'.format(s2_predict))
print('The perplexity of sentence 2 in the Unigram ML Model is: {:.1f}'.format(
      testing.perplexity(s2_predict)))
print('---------------------------------------------------------------------')
print('Sentence 3:')
s3_predict = testing.unigram_predict(s3_unk, unigram_probs)
print('The log probability of sentence 3 in the Unigram ML Model is: \
      {:.3f}'.format(s3_predict))
print('The perplexity of sentence 3 in the Unigram ML Model is: {:.1f}'.format(
      testing.perplexity(s3_predict)))
print('---------------------------------------------------------------------')

input('\nPress Enter to continue to Bigram ML\n')

print('BIGRAM MAXIMUM LIKELIHOOD MODEL\n')
print('Training the model with brown-train.txt...')
bigram_probs = training.bigram_train(train_bdict, train_list, train_dict)
print('Training finished!\n')

print('Create list of bigrams for the three sentences...')
s1_bigrams = training.bigrams(s1_unk)
s2_bigrams = training.bigrams(s2_unk)
s3_bigrams = training.bigrams(s3_unk)
print(s1_bigrams)
print(s2_bigrams)
print(s3_bigrams)
print('Bigram lists constructed!\n')

print('Sentence 1:')
s1_bpred = testing.bigram_predict(s1_bigrams, bigram_probs, len(s1_unk))
print('The probability of sentence 1 in the Bigram ML Model is: \
      {:.3f}'.format(s1_bpred))
print('The perplexity of sentence 1 in the Bigram ML Model is infinite.')
print('---------------------------------------------------------------------')
print('Sentence 2:')
s2_bpred = testing.bigram_predict(s2_bigrams, bigram_probs, len(s2_unk))
print('The log probability of sentence 2 in the Bigram ML Model is: \
      {:.3f}'.format(s2_bpred))
print('The perplexity of sentence 2 in the Bigram ML Model is: {:.1f}'.format(
      testing.perplexity(s2_bpred)))
print('---------------------------------------------------------------------')
print('Sentence 3:')
s3_bpred = testing.bigram_predict(s3_bigrams, bigram_probs, len(s3_unk))
print('The probability of sentence 3 in the Bigram ML Model is: \
      {:.3f}'.format(s3_bpred))
print('The perplexity of sentence 3 in the Bigram ML Model is infinite.')
print('---------------------------------------------------------------------')

input('\nPress Enter to continue to Bigram Model with Add-One Smoothing\n')

print('BIGRAM with ADD-ONE SMOOTHING\n')
print('Training the model with brown-train.txt...')
len_dict = len(train_dict)
add_one_probs = training.add_one_train(train_bdict, train_list,
                                       train_dict, len_dict)
print('Training finished!\n')

print('Sentence 1:')
s1_aopred = testing.add_one_predict(s1_bigrams, add_one_probs,
                                    train_dict, len(s1_unk))
print('The log probability of sentence 1 in the Bigram Add-One Model is: \
      {:.3f}'.format(s1_aopred))
print('The perplexity of sentence 1 in the Bigram Add-One Model is: \
      {:.1f}'.format(testing.perplexity(s1_aopred)))
print('---------------------------------------------------------------------')
print('Sentence 2:')
s2_aopred = testing.add_one_predict(s2_bigrams, add_one_probs,
                                    train_dict, len(s2_unk))
print('The log probability of sentence 2 in the Bigram Add-One Model is: \
      {:.3f}'.format(s2_aopred))
print('The perplexity of sentence 2 in the Bigram Add-One Model is: \
      {:.1f}'.format(testing.perplexity(s2_aopred)))
print('---------------------------------------------------------------------')
print('Sentence 3:')
s3_aopred = testing.add_one_predict(s3_bigrams, add_one_probs,
                                    train_dict, len(s3_unk))
print('The log probability of sentence 3 in the Bigram Add-One Model is: \
      {:.3f}'.format(s3_aopred))
print('The perplexity of sentence 3 in the Bigram Add-One Model is: \
      {:.1f}'.format(testing.perplexity(s3_aopred)))
print('---------------------------------------------------------------------')

input('\nPress Enter to continue to Bigram Model with Discounting and Katz Backoff\n')

print('BIGRAM with DISCOUNTING AND KATZ BACKOFF\n')
print('Training the model with brown-train.txt...')
katz_unigrams, bigrams_A = training.build_set_A(train_list)
katz_probs = training.katz_backoff(train_list, katz_unigrams,
                                   bigrams_A, len(train_list))
print('Training finished!\n')

print('Sentence 1:')
s1_kpred = testing.katz_predict(s1_bigrams, katz_probs, katz_unigrams,
                                bigrams_A, len(train_list), len(s1_unk))
print('The log probability of sentence 1 in the Bigram Katz Backoff Model is: \
      {:.3f}'.format(s1_kpred))
print('The perplexity of sentence 1 in the Bigram Katz Backoff Model is: \
      {:.1f}'.format(testing.perplexity(s1_kpred)))
print('---------------------------------------------------------------------')
print('Sentence 2:')
s2_kpred = testing.katz_predict(s2_bigrams, katz_probs, katz_unigrams,
                                bigrams_A, len(train_list), len(s1_unk))
print('The log probability of sentence 2 in the Bigram Katz Backoff Model is: \
      {:.3f}'.format(s2_kpred))
print('The perplexity of sentence 2 in the Bigram Katz Backoff Model is: \
      {:.1f}'.format(testing.perplexity(s2_kpred)))
print('---------------------------------------------------------------------')
print('Sentence 3:')
s3_kpred = testing.katz_predict(s3_bigrams, katz_probs, katz_unigrams,
                                bigrams_A, len(train_list), len(s1_unk))
print('The log probability of sentence 3 in the Bigram Katz Backoff Model is: \
      {:.3f}'.format(s3_kpred))
print('The perplexity of sentence 3 in the Bigram Katz Backoff Model is: \
      {:.1f}'.format(testing.perplexity(s3_kpred)))
print('---------------------------------------------------------------------')

print('\nIn Summary:')
input('Press Enter for Sentence 1 Probabilities and Perplexities\n')
print('Sentence 1 Log Probabilities by Model:')
print('Unigram ML Model: {:.3f}'.format(s1_predict))
print('Bigram ML Model is: {:.3f}'.format(s1_bpred))
print('Bigram Add-One Model is: {:.3f}'.format(s1_aopred))
print('Bigram Katz Backoff Model is: {:.3f}'.format(s1_kpred))
print('\nSentence 1 Perplexities by Model:')
print('Unigram ML Model is: {:.1f}'.format(testing.perplexity(s1_predict)))
print('Bigram ML Model is: infinite.')
print('Bigram Add-One Model is: {:.1f}'.format(testing.perplexity(s1_aopred)))
print('Bigram Katz Backoff Model is: {:.1f}'.format(testing.perplexity(s1_kpred)))
print('---------------------------------------------------------------------')
input('Press Enter for Sentence 2 Probabilities and Perplexities\n')
print('Sentence 2 Log Probabilities by Model:')
print('Unigram ML Model is: {:.3f}'.format(s2_predict))
print('Bigram ML Model is: {:.3f}'.format(s2_bpred))
print('Bigram Add-One Model is: {:.3f}'.format(s2_aopred))
print('Bigram Katz Backoff Model is: {:.3f}'.format(s2_kpred))
print('\nSentence 2 Perplexities by Model:')
print('Unigram ML Model is: {:.1f}'.format(testing.perplexity(s2_predict)))
print('Bigram Add-One Model is: {:.1f}'.format(testing.perplexity(s2_aopred)))
print('Bigram ML Model is: {:.1f}'.format(testing.perplexity(s2_bpred)))
print('Bigram Katz Backoff Model is: {:.1f}'.format(testing.perplexity(s2_kpred)))
print('---------------------------------------------------------------------')
input('Press Enter for Sentence 3 Probabilities and Perplexities\n')
print('Sentence 3 Log Probabilities by Model:')
print('Unigram ML Model is: {:.3f}'.format(s3_predict))
print('Bigram ML Model is: {:.3f}'.format(s3_bpred))
print('Bigram Add-One Model is: {:.3f}'.format(s3_aopred))
print('Bigram Katz Backoff Model is: {:.3f}'.format(s3_kpred))
print('\nSentence 3 Perplexities by Model:')
print('Unigram ML Model is: {:.1f}'.format(testing.perplexity(s3_predict)))
print('Bigram ML Model is infinite.')
print('Bigram Add-One Model is: {:.1f}'.format(testing.perplexity(s3_aopred)))
print('Bigram Katz Backoff Model is: {:.1f}'.format(testing.perplexity(s3_kpred)))
print('---------------------------------------------------------------------')

input('\nPress ENTER to continue to Question 7!\n')

'''
Questions 7
Compute the perplexities of the entire test corpora, separately for
the brown-test.txt and learner-test.txt under each of the models.
Discuss the differences in the results you obtained.
'''

print('Please see answers.py to see code for the following steps:')
print('Processing and Training Steps Redone for Completeness...')
# 1. Training data preprocessing
train_fp = 'data/brown-train.txt'
train_list = preprocessing.preprocess_train(train_fp)
train_dict = preprocessing.build_dict(train_list)
# Bigrams list and dictionary with counts for training data
train_bigrams = training.bigrams(train_list)
train_bdict = training.bigram_dict(train_list)

# 2. Test data preprocessing
test_fp = 'data/brown-test.txt'
learner_fp = 'data/learner-test.txt'
test_list = preprocessing.preprocess_test(test_fp, train_dict)
learner_list = preprocessing.preprocess_test(learner_fp, train_dict)
# Bigrams lists and dictionaries with counts
test_bigrams = training.bigrams(test_list)
learner_bigrams = training.bigrams(learner_list)
test_bdict = training.bigram_dict(test_list)
learner_bdict = training.bigram_dict(learner_list)
print('Preprocessing finished!\n')

# 3. Train Unigram ML Model
# to create dictionary of tokens and probabilities
unigram_probs = training.unigram_train(train_list, train_dict)

# 4. Train Bigram ML Model
# to create dictionary of bigrams and probabilities
bigram_probs = training.bigram_train(train_bdict, train_list, train_dict)

# 5. Train Bigram with Add-One Smoothing
add_one_probs = training.add_one_train(train_bdict, train_list,
                                       train_dict, len(train_dict))

# 6. Train Bigram with Discounting and Katz Backoff
katz_unigrams, bigrams_A = training.build_set_A(train_list)
katz_probs = training.katz_backoff(train_list, katz_unigrams,
                                   bigrams_A, len(train_list))
print('All models trained!\n')

input('Press Enter for brown-test.txt\n')

print('BROWN-TEST.TXT')
# Unigram ML Model on brown-test.txt
test_unigram = testing.unigram_predict_no_print(test_list, unigram_probs)
test_uni_perplexity = testing.perplexity(test_unigram)
print('The perplexity of the Unigram ML Model on brown-test.txt: \
      {:.1f}'.format(test_uni_perplexity))

# Bigram ML Model on brown-test.txt
test_bigram = testing.bigram_predict_file(test_fp, bigram_probs,
                                          train_dict, len(test_list))
test_bi_perplexity = testing.perplexity(test_bigram)
print('The perplexity of the Bigram ML Model on brown-test.txt:',
      test_bi_perplexity)

# Bigram Model with Add-One Smoothing on brown-test.txt
test_add_one = testing.add_one_predict_no_print(test_bigrams, add_one_probs,
                                                train_dict, len(test_list))
test_ao_perplexity = testing.perplexity(test_add_one)
print('The perplexity of the Bigram Add-One Smoothing Model on brown-test.txt: \
      {:.1f}'.format(test_ao_perplexity))

# Bigram Model with Discounting and Katz Backoff
print('The Katz Backoff takes a few more seconds...')
test_katz = testing.katz_predict_no_print(test_bigrams, katz_probs,
                                          katz_unigrams, bigrams_A,
                                          len(train_list), len(test_list))
test_katz_perplexity = testing.perplexity(test_katz)
print('The perplexity of the Bigram Katz Backoff Model on brown-test.txt: \
      {:.1f}'.format(test_katz_perplexity))

input('\nPress Enter for learner-test.txt\n')

print('LEARNER-TEST.TXT')
# Unigram ML Model on learner-test.txt
learner_unigram = testing.unigram_predict_no_print(learner_list, unigram_probs)
learner_uni_perplexity = testing.perplexity(learner_unigram)
print('The perplexity of the Unigram ML Model on learner-test.txt: \
      {:.1f}'.format(learner_uni_perplexity))

# Bigram ML Model on learner-test.txt
learner_bigram = testing.bigram_predict_file(learner_fp, bigram_probs,
                                             train_dict, len(learner_list))
learner_bi_perplexity = testing.perplexity(learner_bigram)
print('The perplexity of the Bigram ML Model on learner-test.txt: ',
      learner_bi_perplexity)

# Bigram Model with Add-One Smoothing on brown-test.txt
learner_add_one = testing.add_one_predict_no_print(learner_bigrams,
                                                   add_one_probs,
                                                   train_dict,
                                                   len(learner_list))
learner_ao_perplexity = testing.perplexity(learner_add_one)
print('The perplexity of the Bigram Add-One Smoothing Model on learner-test.txt: \
      {:.1f}'.format(learner_ao_perplexity))

# Bigram Model with Discounting and Katz Backoff
print('The Katz Backoff takes a few more seconds...')
learner_katz = testing.katz_predict_no_print(learner_bigrams,
                                             katz_probs,
                                             katz_unigrams,
                                             bigrams_A,
                                             len(train_list),
                                             len(learner_list))
learner_katz_perplexity = testing.perplexity(learner_katz)
print('The perplexity of the Bigram Katz Backoff Model on learner-test.txt: \
      {:.1f}'.format(learner_katz_perplexity))

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
import math
import sys

# Filepaths to the data
train_fp = 'data/brown-train.txt'
test_fp = 'data/brown-test.txt'
learner_fp = 'data/learner-test.txt'


'''
Preprocessing the Training Data:
Step 1: Add the <s> and </s> tags and append to a list
Step 2: Generate a python dict with above list
Step 3: Replace all tokens with (1) count with <unk> tag

Preprocessing the Test Data:
Step 1: Add the <s> and </s> tags and append to a list
Step 2: Generate a python dict with above list
Step 3: Replace all unseen tokens with <unk> token

For code, see preprocessing.py
'''
print('Preprocessing Data...')



input('Press ENTER to continue...')

'''
Question 1
How many word types (unique words) are there in the training corpus?
Please include the padding symbols and the unknown token.
'''

print('Question 1: ')






'''
Training the following models:
1. Unigram Maximum Likelihood Model
2. Bigram Maximum Likelihood Model
3. Bigram Model with Add-One Smoothing
4. Bigram Model with Discounting and Katz backoff

For code, see training.py
'''

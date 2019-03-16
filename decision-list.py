'''
@Description: This is Programming Assignment "4" for AIT 690: NLP class.In this assignment we create a Python program called decision-list.py that implements a decision list classifier to perform word sense disambiguation.training data contains examples of the word line used in the sense of a phone line and a product line, where the correct sense is marked in the text. 


The program runs without any error and  learn its decision list from line-train.xml and then apply that to line-test.xml.

@ Problem Defination:Design and implement a Python program called decision-list.py .which trains from a decision list and then apply that decision list to each
of the sentences found in line-test.xml in order to assign a sense to the word line.training data contains examples of the word line used in the sense of a phone line and a product line, where the correct sense is marked in the text. 


@Examples of program input and output :
Input:  decision-list.py line-train.xml line-test.xml my-decision-list.txt 
Output:  my-line-answers.txt  will be created 

line-train.xml        - Tagged training corpus 
line-test.xml 		  - Untagged test corpus 
my-decision-list.txt  - Output Log
my-line-answers.txt  - Output with Id and sense tag
my-line-answers.txt will be used to evaluate accuracy and confusion matrix

@Algorithim:
		1.Get inputs for system console
	2.load training data to pandas dataframe.
	3.Clean training data.
	4.Generate feature vectors for phone and product.
	5.Get count of phone and product sense count.
	6.word counts across feature.
	7.propablilities across feature.
	8.load test data to pandas dataframe.
	9.clean test data.
	10.POS tag test data.
	11.score test data cross phone feature vector.
	12.score test data cross product feature vector.
	13.score test data cross product feature vector.
	14.prediction to sysout
		
		


@Date: 31 Oct, 2018.

Confusion matrix: = [[55 17]
                     [1 53]]
Accuracy: = 0.8571428571428571'''

import sys
import nltk
import string
import numpy as np
import time
import pandas as pd
import xml.etree.ElementTree as ET
from copy import copy
from itertools import count
from collections import defaultdict

import logging
logger = logging.getLogger('WSD')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def process_children(element):
    '''
    Return concatinated text string from xml children elemnets. with out ambigious word
    '''
    text = ''
    for child in element.getchildren():
        try:
            text = text + ' ' + child.text
        except TypeError:
            pass
    logger.debug("Text contatination without ambigious word: {}".format(text))
    return text

def process_children2(element):
    '''
    Return concatinated text string from xml children elemnets. with ambigious word
    '''
    text = ''
    for child in element.getchildren():
        if child.getchildren():
            text = text + ' ' + process_children2(child)
        try:
            text = text + ' ' + child.text
        except TypeError:
            pass
    logger.debug("Text contatination with ambigious word: {}".format(text))
    return text

def get_context_list_train(xmlobject):
    '''
    Converts training data XML element tree to a python list of dictionaries to use as data frame.
    '''
    data = []
    dict_item = {}
    iterator = xmlobject.iter()
    logger.debug("creating traing data from context")
    for element in iterator:
        if element.tag == 'context':
            dict_item['context'] = process_children(element) # With out ambigious word.
            dict_item['context2'] = process_children2(element) # With ambigious word.
            data.append(copy(dict_item))
            dict_item = {}
            continue
        elif element.tag == 'answer':
            dict_item['senseid'] = element.attrib.get('senseid')
            dict_item['element'] = ET.tostring(element).decode("utf-8").strip('\n')
            continue
    return data

def get_dataframe_train_data(train_file):
    '''
    Returns dataframe for training data.
    '''
    logger.debug("creating traing dataframe")
    train_data = ET.parse(train_file)
    train_data = get_context_list_train(train_data)
    return pd.DataFrame(train_data, columns=["element", 'senseid', 'context', 'context2'])

def get_context_list_test(xmlobject):
    '''
    Converts test data XML element tree to a python list of dictionaries to use as data frame.
    '''
    data = []
    data_item = {}
    iterator = xmlobject.iter()
    logger.debug("creating testing data from xml")
    for element in iterator:
        if element.tag == 'instance':
            data_item['id'] = element.attrib.get('id')
            data_item['context'] = process_children(element.getchildren()[0])
            data_item['context2'] = process_children2(element.getchildren()[0])
            data.append(copy(data_item))
    return data

def get_dataframe_test_data(test_file):
    '''
    Returns dataframe for test data.
    '''
    test_data = ET.parse(test_file)
    test_data = get_context_list_test(test_data)
    return pd.DataFrame(test_data, columns=['id', 'context', 'context2'])

def clean_data(df, column_name_I, column_name_o):
    '''
    Removes punctuations, stopwords and create word tokens per sense instance.
    '''
    logger.debug("Cleaning: ")
    stopwords = nltk.corpus.stopwords.words('english')
    remove_punct = lambda text: ''.join((char for char in text if char not in string.punctuation))
    remove_stopwords = lambda sentence: [word for word in sentence.lower().split() if word not in stopwords]
    unpunct_array = df[column_name_I].apply(remove_punct)
    logger.debug("Cleaning: removing punctiations \n {}".format(unpunct_array))
    df[column_name_o] = unpunct_array.apply(remove_stopwords)
    logger.debug("Cleaning: removing stopwords \n {}".format(df[column_name_o]))
    return df

def get_tagged_features_freq(df, column_name_I, column_name_o):
    '''
    Generate feature vectors for phone and product senses.
    '''
    logger.debug("Creating: POS feature vectors")
    df[column_name_o] = df[column_name_I].apply(nltk.pos_tag)
    all_feature_phone = defaultdict(lambda: 0)
    all_feature_product = defaultdict(lambda: 0)
    for pos_set in df[df['senseid'] == 'phone'][column_name_o]:
        for word_tag in pos_set:
            all_feature_phone[word_tag] = all_feature_phone[word_tag] + 1
    for pos_set in df[df['senseid'] == 'product'][column_name_o]:
        for word_tag in pos_set:
            all_feature_product[word_tag] = all_feature_product[word_tag] + 1
    return all_feature_phone, all_feature_product

def score(pos_set, feature_set):
    '''
    Score parts of speech tagged token set with feature_set (i.e. Can be sense feature set for phone sense or product sense)
    '''
    logger.debug("scoreing: using POS tagged tokens")
    score_ = 0
    for word_tag in pos_set:
        score_ = score_ + feature_set[word_tag]
    return score_

def get_index(lst, word):
    '''
    Word index fetching function.
    '''
    logger.debug("Fetcing: index of ambgious word form context")
    try:
       return lst.index(word)
    except ValueError:
       return np.NaN

def get_offset_word(lst, word, offset):
    '''
    Word offset based fetching function
    '''
    logger.debug("Fetcing: offsset word from ambgious word form context")
    try:
       return lst[int(lst.index(word) + offset)]
    except (ValueError, IndexError):
       return np.NaN


def gen_count_column(df, word_col, tokens_col):
    '''
    Offset word count function to generate propabilities.
    '''
    logger.debug("Counting: generating word conts for offset words.")
    df[word_col + '_count'] = np.array([item.count(train_df[word_col][idx])
                                        for idx, item in enumerate(train_df[tokens_col])])

if __name__ == '__main__':
	start_time=time.time()
	# STEP-1: Get inputs for system console
	train_file = sys.argv[1]
	test_file = sys.argv[2]
	output_file = sys.argv[3]
	ambigious_word = 'line'

	hdlr = logging.FileHandler(output_file)
	hdlr.setFormatter(formatter)
	logger.addHandler(hdlr)
	logger.setLevel(logging.DEBUG)
	logger.info("Hello World!")


	# STEP-2: load training data to pandas dataframe.
	train_df = get_dataframe_train_data(train_file)

	# STEP-3: Clean training data.
	train_df = clean_data(train_df, 'context', 'word_tokens')
	train_df = clean_data(train_df, 'context2', 'word_tokens2')  # with ambigious word
	train_df['word_count2'] = train_df['word_tokens2'].apply(lambda x:len(x))

	# STEP-4: Generate feature vectors for phone and product.
	freq_features_phone, freq_features_product = get_tagged_features_freq(train_df, 'word_tokens', 'pos_tag')
	freq_features_phone_2, freq_features_product_2 = get_tagged_features_freq(train_df, 'word_tokens2', 'pos_tag2')
	train_df['word_tokens2_count'] = train_df['word_tokens2'].apply(lambda x: len(x))

	# STEP-5: Get count of phoe and product sense count.
	phone_freq = train_df[train_df['senseid'] == 'phone'].shape[0]
	product_freq = train_df[train_df['senseid'] == 'product'].shape[0]
	p_phone = np.float(phone_freq/(phone_freq + product_freq)) # propability of phone sense
	p_product = 1.0 - p_phone # propability of product sense

	train_df['word_1'] = train_df['word_tokens2'].apply(get_offset_word, args=('line', 1))
	train_df['word_n1'] = train_df['word_tokens2'].apply(get_offset_word, args=('line', -1))
	train_df['word_2'] = train_df['word_tokens2'].apply(get_offset_word, args=('line', 2))
	train_df['word_n2'] = train_df['word_tokens2'].apply(get_offset_word, args=('line', -2))

	# word counts across feature.
	gen_count_column(train_df, 'word_1', 'word_tokens2')
	gen_count_column(train_df, 'word_n1', 'word_tokens2')
	gen_count_column(train_df, 'word_2', 'word_tokens2')
	gen_count_column(train_df, 'word_n2', 'word_tokens2')   

	# propablilities across feature.
	train_df['p_word_1_context2'] = train_df['word_1_count']/train_df['word_tokens2_count'] 
	train_df['p_word_n1_context2'] = train_df['word_n1_count']/train_df['word_tokens2_count']
	train_df['p_word_2_context2'] = train_df['word_2_count']/train_df['word_tokens2_count']
	train_df['p_word_n2_context2'] = train_df['word_n2_count']/train_df['word_tokens2_count']

	train_df['p_word_1_context2'][train_df['p_word_1_context2'] == 0] = 1
	train_df['p_word_n1_context2'][train_df['p_word_n1_context2'] == 0] = 1
	train_df['p_word_2_context2'][train_df['p_word_2_context2'] == 0] = 1
	train_df['p_word_n2_context2'][train_df['p_word_n2_context2'] == 0] = 1

	# compute log likelyhood
	train_df['p_word_1_context2'] = -np.log(train_df['p_word_1_context2'])
	train_df['p_word_n1_context2'] = -np.log(train_df['p_word_n1_context2'])
	train_df['p_word_2_context2'] = -np.log(train_df['p_word_2_context2'])
	train_df['p_word_n2_context2'] = -np.log(train_df['p_word_n2_context2'])

	# STEP-6: load test data to pandas dataframe.
	test_df = get_dataframe_test_data(test_file)

	# STEP-7: clean test data.
	test_df = clean_data(test_df, 'context', 'word_tokens')
	test_df = clean_data(test_df, 'context2', 'word_tokens2')

	# STEP-8: POS tag test data.
	test_df['pos_tag'] = test_df['word_tokens'].apply(nltk.pos_tag)
	test_df['pos_tag2'] = test_df['word_tokens2'].apply(nltk.pos_tag)
	test_df['word_1'] = test_df['word_tokens2'].apply(get_offset_word, args=('line', 1))
	test_df['word_n1'] = test_df['word_tokens2'].apply(get_offset_word, args=('line', -1))
	test_df['word_2'] = test_df['word_tokens2'].apply(get_offset_word, args=('line', 2))
	test_df['word_n2'] = test_df['word_tokens2'].apply(get_offset_word, args=('line', -2))

	phone_score_1_list = []
	phone_score_n1_list = []
	phone_score_2_list = []
	phone_score_n2_list = []
	product_score_1_list = []
	product_score_n1_list = []
	product_score_2_list = []
	product_score_n2_list = []
	for w1, wn1, w2, wn2 in zip(test_df['word_1'], test_df['word_n1'], test_df['word_2'], test_df['word_n2']):
	   cond_1 = train_df['word_1'] == w1
	   cond_n1 = train_df['word_n1'] == wn1
	   cond_2 = train_df['word_2'] == w2
	   cond_n2 = train_df['word_n2'] == wn2
	   p_1 = train_df['p_word_1_context2'][cond_1]
	   p_n1 = train_df['p_word_n1_context2'][cond_n1]
	   p_2 = train_df['p_word_2_context2'][cond_2]
	   p_n2 = train_df['p_word_n2_context2'][cond_n2]
	   s_1 = train_df['senseid'][cond_1]
	   s_n1 = train_df['senseid'][cond_n1]
	   s_2 = train_df['senseid'][cond_2]
	   s_n2 = train_df['senseid'][cond_n2]
	   phone_score_1 = 0
	   phone_score_1 = phone_score_1 + sum((p for s,p in zip(s_1, p_1) if s == 'phone'))
	   phone_score_n1 = 0
	   phone_score_n1 = phone_score_n1 + sum((p for s,p in zip(s_n1, p_n1) if s == 'phone'))
	   phone_score_2 = 0
	   phone_score_2 = phone_score_2 + sum((p for s,p in zip(s_2, p_2) if s == 'phone'))
	   phone_score_n2 = 0
	   phone_score_n2 = phone_score_n2 + sum((p for s,p in zip(s_n2, p_n2) if s == 'phone'))
	   product_score_1 = 0
	   product_score_1 = product_score_1 + sum((p for s,p in zip(s_1, p_1) if s == 'product'))
	   product_score_n1 = 0
	   product_score_n1 = product_score_n1 + sum((p for s,p in zip(s_n1, p_n1) if s == 'product'))
	   product_score_2 = 0
	   product_score_2 = product_score_2 + sum((p for s,p in zip(s_2, p_2) if s == 'product'))
	   product_score_n2 = 0
	   product_score_n2 = product_score_n2 + sum((p for s,p in zip(s_n2, p_n2) if s == 'product'))
	   phone_score_1_list.append(phone_score_1)
	   phone_score_n1_list.append(phone_score_n1)
	   phone_score_2_list.append(phone_score_2)
	   phone_score_n2_list.append(phone_score_n2)
	   product_score_1_list.append(product_score_1)
	   product_score_n1_list.append(product_score_n1)
	   product_score_2_list.append(product_score_2)
	   product_score_n2_list.append(product_score_n2)

	test_df['phone_score_1'] = np.array(phone_score_1_list)
	test_df['phone_score_n1'] = np.array(phone_score_n1_list)
	test_df['phone_score_2'] = np.array(phone_score_2_list)
	test_df['phone_score_n2'] = np.array(phone_score_n2_list)
	test_df['product_score_1'] = np.array(product_score_1_list)
	test_df['product_score_n1'] = np.array(product_score_n1_list)
	test_df['product_score_2'] = np.array(product_score_2_list)
	test_df['product_score_n2'] = np.array(product_score_n2_list)

	# STEP-9: score test data cross phone feature vector.
	test_df['phone_score_pos'] = test_df['pos_tag'].apply(score, args=(freq_features_phone,))

	# STEP-10: score test data cross product feature vector.
	test_df['product_score_pos'] = test_df['pos_tag'].apply(score, args=(freq_features_product,))

	test_df['phone_score'] = test_df['phone_score_1'] + \
		test_df['phone_score_n1'] + test_df['phone_score_2'] + \
		test_df['phone_score_n2'] + test_df['phone_score_pos']

	test_df['product_score'] = test_df['product_score_1'] + \
		test_df['product_score_n1'] + test_df['product_score_2'] + \
		test_df['product_score_n2'] + test_df['product_score_pos']    

	# STEP-11: score test data cross product feature vector.
	test_df['sense'] = np.where(test_df['phone_score'] > test_df['product_score'], 'phone',
					   np.where(test_df['phone_score'] < test_df['product_score'], 'product', 'phone'))

	# STEP-12: prediction to sysout
	for id_, sense in zip(test_df['id'], test_df['sense']):
		print(f'<answer instance="{id_}" senseid="{sense}"/>')
	logger.info("--- %s seconds ---" %(time.time() - start_time))

	
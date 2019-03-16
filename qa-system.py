"""

@Description:
The project was to implement a Question Answering (QA) system in Python called qa-system.py. The system should handle questions from any domain, and should provide answers in complete sentences that are specific to the question asked. System should be able to answer Who, What, When and Where questions (but not Why or How questions).It follows an approach similar to that of the AskMSR system, and simply reformulate the question as a series of "answer patterns" that we then search for in Wikipedia, where you are hoping to find an exact match to one of your patterns.It uses existing toolkits to interact with Wikipedia.

@ Problem Defination:
A question answering system is usually used for the task of returning a particular piece of information to the user in response to a question. These systems typically use Information Retrieval (IR), information extraction techniques and for employing Natural Language Processing methodologies. However, doing so can be very extensive from the initial phases. Thus, after the document set is reduced, the NLP techniques can be utilized.
@Examples of program input and output :
Input question type:
	When was Google founded?
	When was George Washington born?

	Where is GMU?
	Where is India?
	Where are Amazon Forests?

	What is Laptop?
	What is Earth?
	What is meant by Apple ?

	Who was Mahatama Gandhi ?
	Who was George Washington ?
Output:
	out will be one line answer. 
@Algorithim:
1.it takes the question given by the user 
2.identifies the type of Wh-question asked 
3.part-of-speech (POS) tags the words with verbs, nouns, pronounces, helping, verb, etc
4.search text tagged as Noun, Geographical location, entity, etc. on Wikipedia 
5.text undergoes segmentation and these tokenized sentences are filtered to match the queries
6.then ranked and the highest ranked sentence is formulated as a final answer and displayed as an output to the use

@Authors: Sri Ram Sagar Kappagantula
	  Harsimrat Kaur
	  Ritika De
@Date: Dec 8, 2018


"""



import sys
from sys import argv
import wikipedia
import wikipediaapi
import nltk
from nltk import word_tokenize
import re
from nltk.util import ngrams
from collections import Counter
#import nltk
from nltk.tokenize import TweetTokenizer, sent_tokenize
from nltk.corpus import wordnet 
import numpy as np
import string

import spacy
nlp = spacy.load('en_core_web_lg')


def SearchInWiki(term):									
	SearchResults=wikipedia.search(term)			# search identified Noun in wiki
	Results=str(SearchResults)
	ToSearch=SearchResults[0]						# if multiple pages like NY city, NY state, NY uni fore Ny, pick first result
	#print("this can mean  "+Results)
	#print(" I hope you mean to find "+ ToSearch)

	try:
		wiki_wiki = wikipediaapi.Wikipedia('en')
		wiki_html = wikipediaapi.Wikipedia(
				language='en',
				extract_format=wikipediaapi.ExtractFormat.WIKI			#Using Wki api
		)

		page_py = wiki_wiki.page(ToSearch)
		if page_py.exists():
			#print("Found it")
			W_summary=wikipedia.summary(ToSearch)
			FullText=wiki_html.page(ToSearch)							#get All text on that page
			MyText=FullText.text

		else:
			print("Could not Wiki page by this title")
	except wikipedia.exceptions.DisambiguationError:
		print("Could mean many things, PLease be specific")			#handle disambiguation error

	return MyText
""" Tokenize wiki page text into sentences"""
""" Do stemmming of text"""
def TokenSent(MyText):

	Wsummary=str(MyText)
	lemma = nltk.wordnet.WordNetLemmatizer()					#stemming of the text summary
	Text23=lemma.lemmatize(Wsummary)
	tokenizer_words = TweetTokenizer()
	tokens_sentences =nltk.sent_tokenize(MyText)
	#print(tokens_sentences)
	#my_sntences=nltk.sent_tokenize(Wsummary)

	return tokens_sentences
	"""Tokenize sent to Words to find sentences with are matching our query """
def TokenWords(My_sent):
	tokenizer_words = TweetTokenizer()
	tokens_words = [tokenizer_words.tokenize(t) for t in My_sent]

	return tokens_words

def clean(FinalSent):

	r=['.',',','(',')',';']
	for x in range(len(r)):
		
		try:
			FinalSent=FinalSent.replace(r[x],'')				#remove symbols from text
		except ValueError:
			break
	#print(FinalSent)
	return FinalSent
""" Query is question asked by the user; here we are trying to finf words tagged as noun and Verb"""
def get_ent(Query):								

	doc=nlp(Query)
	li_ent=[]
	li_verb=[]
	for ent in doc.ents:
		li_ent.append(ent)							#get entities like Noun, PLaces etc
	for token in doc:
		if token.pos_=="VERB":						#get Verbs
			li_verb.append(token)
		
	return(li_ent, li_verb)

#r(Who (is|was)(.*)(\?))


def break_ques(ques):
	line = str(ques);

	searchObj = re.search( r'When (is|are|was|were)(.*)(\?)', line, re.M|re.I)

	if searchObj:
	   #print( "searchObj.group() : ", searchObj.group())
	   x=searchObj.group(1)
	   term=searchObj.group(2)
	else:
	   print("Nothing found!!")
	return(x,term)



def get_syn():														#this is used to get synonyms of Verb
	syns = wordnet.synsets("born")
	for i in range(len(syns)):
		print(syns[i].lemmas()[0].name())

def list_with_verb(tokens_sentences,Verb1):
	tokenizer_words = TweetTokenizer()
	tokens_word_sent = [tokenizer_words.tokenize(t) for t in tokens_sentences]
	listd=[]
	for i in range(len(tokens_sentences)):
	#print(tokens_sentences[i])
		for j in range(len(tokens_word_sent[i])):
			#print(tokens_sentences[i][j])
			
			if tokens_word_sent[i][j]==Verb1:
				ftokens_sentences=tokens_word_sent[i][j:]
				#print(ftokens_sentences)
				listd.append(ftokens_sentences)
	return(listd)

def cos_sim(a,b):											#calulate Cosine distance between two verctors
	dot_product=np.dot(a,b)
	norm_a=np.linalg.norm(a)
	norm_b=np.linalg.norm(b)
	return dot_product/(norm_a*norm_b)

def getSimilarity(dict1, dict2):							#cal Similarity between two strings
	all_words=[]
	for key in dict1:
		all_words.append(key)
	for key in dict2:
		all_words.append(key)
	all_words_size=len(all_words)

	v1=np.zeros(all_words_size, dtype=np.int)
	v2=np.zeros(all_words_size, dtype=np.int)
	i=0
	for (key) in all_words:
		v1[i]=dict1.get(key,0)
		v2[i]=dict2.get(key,0)
		i=i+1
	return cos_sim(v1,v2)

def p_count(stemmed_tokens):									#helps in calculating frequency to get matrices
	count=nltk.defaultdict(int)
	for word in stemmed_tokens:
		count[word]+=1
	return count

def remove_symbols(anyString):			         #to claen list

	r=[',','(',')',';','_','–','-','[',']']
	for x in range(len(r)):
		while True:
			try:
				anyString.remove(r[x])
			except ValueError:
				break
	#print(answer_sent)
	return(anyString)


def for_when(tokens_sentences,tokens_word_sent):

	My_li=list_with_verb(tokens_sentences,Verb1)			#get matching sentences
	file.write(str(My_li))
	print(Verb1)
	file.write(str(My_li))
	if Verb1=="founded":
		Sent1=['founded','in']
	elif Verb1=="born":
		Sent1=['born','on']
		print("blaaaa")
	elif Verb1=="invented":
		Sent1=['invented','in']

	Sent2=My_li										
	"""To get best answer fot when type question """
	score=[]
	dict1=p_count(Sent1)
	for x in range(len(Sent2)):
		dict2=p_count(Sent2[x])
		r=getSimilarity(dict1,dict2)
		score.append(r)
	#print(score)
	answer_sent=Sent2[score.index(max(score))]
	#print(answer_sent)
	""" Formulation os answer string """
	FinalSent="".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in answer_sent]).strip()
	pattern = '([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[a-z]+)?(?:\s+[A-Z][a-z]+)+)'
	result = re.findall(pattern,FinalSent)
	
	FinalSent=str(FinalSent)
	try:
		from dateutil.parser import parse
		dt=[]
		dt, tokens = parse(FinalSent, fuzzy_with_tokens=True)
		dt=str(dt).split(' ')[0]
		#print(dt)
		#print(ToSearch+' '+'was'+' '+ Verb+' in '+dt+'.')
		xq= Verb1+' on '+dt+'.'
		
	except:
		xq=FinalSent

	return xq

def for_where(ToSearch,MyText):
	"""This function gives Where answers, """
	doc=nlp(ToSearch)
	#print("Here")
	for ent in doc.ents:
		if ent.label_== "GPE":
			ans_1=wikipedia.summary(ToSearch, sentences=1)
			ans=(ans_1).split('is')[1]
		else:
			doc1=nlp(MyText)
			listR=[]
			for ent in doc1.ents:
				if ent.label_== "GPE":							#get sent with locations in it
					listR.append(str(ent))
			from collections import Counter
			new_vals = Counter(listR)
			new_vals_p = Counter(listR).most_common(4)

			y=sorted(new_vals, key=new_vals.get, reverse=True)			#get best matched answer
			#print(y)
			file.write("best matching locations")
			file.write(str(new_vals_p))
			ans=y[0]
			#print(ans)
	return ans
#Since we are using wiki, 1st line will have best and concrete answer, 
#for who/what we are not ranking the matches
#we are getting 1st line of summary and this is always 100% accurate

def for_what(ToSearch,tokens_sentences,x):
	#print("a")
	li=[]
	first_sent=str(tokens_sentences[0])
	li.append(str(tokens_sentences[0]))

	clean_first_sent=clean(first_sent)
	if x=="is":
		
		a,b=clean_first_sent.split(' is ',1)
		
		ans=((''.join(str(e) for e in ToSearch))+' '+x+' '+ b+'.')
		ans = ans.replace("[","")
		ans = ans.replace("]","")
	
	elif x=='was':
		
		a,b=clean_first_sent.split(' was ',1)
		ans=(''.join(str(e) for e in ToSearch)+' '+x+' '+ b+'.')
		ans = ans.replace("[","")
		ans = ans.replace("]","")
		#print("thsi"+ans)
	

	return ans


"""Main starts here"""


if __name__== '__main__':
	#start_time=time.time()
	

	chat=str("Hi")
	#chat=input()
	file=open("log-file.txt","a+")
	file.write("Starting New Log.....                                   ")

	##############################################
	while not re.findall(r'([Bb]ye)$|([Ee]xit)$|([Qu]it)$',chat):     #Program can be exited by saying exit/bye
		
		
		try:
						
			print("Now Ask question to wiki..")
			chat=input()
			
			if re.findall(r'([Bb]ye)$|([Ee]xit)$|([Qu]it)$',chat):
				break
			
			#print(chat)
			Fword=chat.split(' ', 1)[0]     #get first word
			#print(Fword)
			
			ToSearch,Verb=get_ent(chat)     # get subject and verb
			try:
				Verb1=str(Verb[1])
				#ToSearch=str(ToSearch[0])
			except:
				pass
			print(str(ToSearch))
			print(Verb[0])
			TSearch=str(ToSearch)
			MyText=SearchInWiki(TSearch)		#get wiki page text
			tokens_sentences=TokenSent(MyText)   # tokenize sentences
			#print(tokens_sentences[0])
			tokens_word_sent=TokenWords(tokens_sentences)	#token words
			#print("Kill ")
			if Fword=="When":
				ans1=for_when(tokens_sentences,tokens_word_sent)
				#print(ans1)
				ans=(''.join(str(e) for e in ToSearch)+' '+'was'+ans1+'.')
			elif Fword=="Where":
							
				ans1=for_where(TSearch,MyText)
				ans=(TSearch+" is in "+str(ans1)+".")
				ans = ans.replace("[","")
				ans = ans.replace("]","")
				
			elif Fword=="What":
				
				
				ans=for_what(TSearch,tokens_sentences,str(Verb[0]))
			elif Fword=="Who":
				
				ans=for_what(TSearch,tokens_sentences,str(Verb[0]))
					
			print(ans)
			file.write(chat)
			file.write(ans)
			
		except:
			
			print("I dont know the answer, please follow following rules:\n Ques should start with capital word.\n Use was for dead people.\n Noun should start with capital letter \n Your ques should not have more than one verb like founded, born etc")
			file.write("wrong question asked       ")
	file.close()






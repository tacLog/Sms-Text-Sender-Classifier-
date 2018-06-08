import csv
import string
import sys
import codecs
import xml.etree.ElementTree as ET
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

reload(sys)
sys.setdefaultencoding('utf-8')

#main data class
class SmsText:
	protocol = 0
	number = 0
	date = 0
	in_out_type = 0
	body = ""
	sent_date = 0
	readable_Date = ""
	contact_name = ""
	read_label = ""
	tokens = []
	features = []

	def __init__(self, input_tuple):
		protocol, number, date, in_out_type, body, sent_date, readable_Date, contact_name = input_tuple
		self.protocol = int(protocol)
		self.number = eliminate_punctuation(number)
		self.date = int(date)
		self.in_out_type = int(in_out_type)
		self.body = body
		self.sent_date = int(sent_date)
		self.readable_Date  = readable_Date
		self.contact_name = contact_name

def eliminate_punctuation(word):
	punctuation = set(string.punctuation)
	new_word = ''
	word = word.replace(' ','')
	for char in word:
		if char in punctuation:
			pass
		else:
			new_word = ''.join((new_word, char))
	return new_word

def parseSMS(xmlfile):

	#tree of xml
	tree = ET.parse(xmlfile)

	#get head of tree
	head = tree.getroot()

	#print head

	texts = []

	for sms in head.findall('./sms'):
		protocol = sms.attrib['protocol']
		number = sms.attrib['address']
		date = sms.attrib['date']
		in_out_type = sms.attrib['type']
		body = sms.attrib['body']
		sent_date = sms.attrib['date_sent']
		readable_Date = sms.attrib['readable_date']
		contact_name = sms.attrib['contact_name']
		current = SmsText((protocol, number, date, in_out_type, body, sent_date, readable_Date, contact_name))
		texts.append(current)

	#print head.attrib['count']
	#print len(texts)
	#assert(len(texts) == head.attrib['count']),"warning, text count does not match extracted object count"
	return texts


	
def getTopPeopleNumbers(texts, number,in_out):
	people_count = {}
	for text in texts:
		if text.in_out_type == in_out:
			if text.number not in people_count:
				people_count[text.number] =1
			else:
				people_count[text.number] +=1


	close_poeple = []
	for key in people_count:
		if people_count[key] > 500:
			close_poeple.append((key, people_count[key]))

	close_poeple.sort(key=lambda tup: tup[1], reverse=True)
	#print close_poeple
	return close_poeple[:number]

def getByNumbers(texts, numbers, in_out):
	output = []
	for text in texts:
		if text.in_out_type == in_out:
			if text.number in numbers:
				output.append(text)
	return output

def processCorpus(rawCorpus):
	punctuation = set(string.punctuation)
	stemmer = PorterStemmer()
	get_unique_tokens = False
	if get_unique_tokens:
		# set used to count unique tokens
		token_types = set()
	#set needed to count other things
	token_frequency = {}
	for message in rawCorpus:
		line = message.body
		#lowercasing of lines
		line = line.lower()
		line = word_tokenize(line)
		# Calculates number of unique tokens after tokenization
		if get_unique_tokens:
			for token in line:
				token_types.add(token)

		line = [word for word in line if word not in stopwords.words('english')]
		line = [word for word in line if word not in punctuation]
		line = [stemmer.stem(word) for word in line]

		# Assemble frequency dictionary
		for token in line:
			if token_frequency.get(token):
				token_frequency[token] += 1
			else:
				token_frequency[token] = 1

		message.tokens = line 


	return rawCorpus, token_frequency



def xml_parser(filename = 'sms-20180601001754.xml'):

	texts = parseSMS(filename)

	in_out = 1 #in or out(1 or 2)

	#in = objects, number of people
	close_numbers_count = getTopPeopleNumbers(texts, 2, in_out)
	print close_numbers_count

	close_numbers = [i[0] for i in close_numbers_count]
	rawCorpus = getByNumbers(texts, close_numbers, in_out)
	print len(rawCorpus)

	lable_count = 1
	labels = {}
	for key in close_numbers_count:
		labels[key[0]] = lable_count
		lable_count = lable_count -1

	print "labels: "
	print labels

	#set read_label goal
	for message in rawCorpus:
		message.read_label = message.number
		message.label = labels[message.number]

	#process corpus
	corpus, token_frequency = processCorpus(rawCorpus)

	return corpus, token_frequency

	



if __name__ == "__main__":
	xml_parser()

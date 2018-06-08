import xml_parser as xml
import copy
import random
import math
import string

# Input:
#  Total_instance_list -- a 2-tuple comprising the label and the words (instance label, instance)
#  Token_frequency -- a dictionary containing word frequencies for elements of the above list
# Output:
#  A 2-tuple (instance list, token_frequency) pruned so all words had a frequency >= 5
def five_occurence_pruner(total_instance_list, token_frequency):
	pruned_instance_list = []
	for instance in total_instance_list:
		line = instance.tokens
		new_line = []
		for word in line:
			if token_frequency[word] >= 5:
				new_line.append(word)
		instance.tokens = new_line

	return total_instance_list


# Input:
#  Total_instance_list -- a 2-tuple comprising the label and the words (instance label, instance)
# Output:
#  A dictionary with all unique tokens from all instances as keys and all values intialized to 0
def get_vocabulary(total_instance_list):
	vocabulary_token_types = set()
	for message in total_instance_list:
		for token in message.tokens:
			vocabulary_token_types.add(token)


	occurences = {}
	for key in vocabulary_token_types:
		occurences[key] = 0

	return occurences

# Input:
#  Instances -- a list of instances where each instance is a tuple (label, [features])
#  Instance_dict -- a dictionary where each key is an item in the vocabulary space and each value is set to 0
# Output:
#  A list of 3-tuples similar to 2-tuple instances that was inputted,
#     with an additional dictionary mapping each feature to its number of occurances:
#     (label, [features], {feature:occurance, ...})
def final_instance_assembler(instances, instance_dict):
	for message in instances:
		label = message.label
		features = message.tokens
		out_dict = copy.deepcopy(instance_dict)
		for feature in features:
			if out_dict.has_key(feature):
				out_dict[feature] += 1

		output = []
		for key in out_dict:
			output.append(out_dict[key])
		message.features = output

	return instances

def context_gather(corpus):
	for message in corpus:
		exclamtion = 0
		period = 0
		comma = 0
		caps = 0
		length = 0
		for word in message.body.split():
			for char in word:
				if char == '!':
					exclamtion +=1
				if char == '.':
					period +=1
				if char == ',':
					comma +=1
				if char in string.ascii_uppercase:
					caps +=1
			length +=1
		message.features.extend((exclamtion,period,comma,caps,length))
	return corpus




def dataMaker(split = .9):
	corpus, token_frequency = xml.xml_parser()

	corpus = five_occurence_pruner(corpus, token_frequency)

	print "pre prunning zeros"
	print len(corpus)
	#prunning
	new_messages = []
	for message in corpus:
		if len(message.tokens) !=0:
			new_messages.append(message)
	corpus = new_messages

	print "post prunning"
	print len(corpus)

	vocabulary = get_vocabulary(corpus)
	corpus = final_instance_assembler(corpus, vocabulary)

	corpus = context_gather(corpus)

	print "Packaging corpus"
	corpus = Corpus(corpus,split)

	return corpus

class Corpus:
	total_instances = []
	test_set = []
	train_set = []

	def __init__(self, total_instances, split):
		random.shuffle(total_instances)
		length = len(total_instances)
		div = int(math.floor(length*split))
		train = total_instances[:div]
		test = total_instances[div:]
		self.total_instances = total_instances
		self.test_set = test
		self.train_set = train



if __name__ == "__main__":
	dataMaker()
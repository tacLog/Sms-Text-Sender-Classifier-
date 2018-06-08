import corpusProcessor
import numpy as np
import math
import copy

class LogisticRegression:
	weights =None
	iterations =None
	rate = 0.01

	def __init__(self, n, iterations):
		self.weights = np.zeros((n,), dtype=float)
		self.iterations = iterations

	def weightsL2Norm(self):
		sum = 0
		for i in self.weights:
			sum = sum + i
		if sum >0:
			return math.sqrt(sum)
		return -1
	
	def sigmoid(self, z):
		sig = math.exp(-z)
		sig = 1 + sig
		return 1.0/sig

	def probPred1(self, x):
		sum =0
		for i in range(len(x)):
			sum = sum + self.weights[i]*x[i]
		return self.sigmoid(sum)

	def predict(self, x):
		prob = self.probPred1(x)
		if prob >= .5:
			return 1
		return 0

	def printPerformance(self, test_instances):
		acc = 0.0
		tp = 0
		tn = 0
		fp = 0
		fn = 0

		for i in range(len(test_instances)):
			predicted_label = self.predict(test_instances[i].features)
			lable = test_instances[i].label
			if lable == predicted_label:
				if predicted_label ==1:
					tp =tp+1
				else:
					tn = tn+1
			else:
				if predicted_label == 1:
					fp =fp+1
				else:
					fn =fn+1

		acc = (tp + tn) / float(len(test_instances))

		print "Accaracy=" + str(acc)
		print "Confusion Matrix"
		print str(tp) + "  " +str(fn)
		print str(fp) + "  " +str(tn)

	def train(self, corpus):
		for n in range(self.iterations):
			lik = 0.0
			for i in range(len(corpus)):
				label = corpus[i].label
				features = corpus[i].features
				probPred = self.probPred1(features)
				for w in range(len(self.weights)):
					self.weights[w] = self.weights[w] + self.rate*features[w]*(label - probPred)
				localSum = 0
				for z in range(len(features)):
					localSum = localSum + features[z] *self.weights[z]
				lik = lik + label*localSum - math.log(1+math.exp(localSum))
			print "iteration: " + str(n) +" lik: " + str(lik)


class LogisticRegressionReg:
	weights =None
	iterations =None
	lam = 0.0001
	rate = 0.01

	def __init__(self, n, iterations):
		self.weights = np.zeros((n,), dtype=float)
		self.iterations = iterations

	def weightsL2Norm(self):
		sum = 0
		for i in self.weights:
			sum = sum + i
		if sum >0:
			return math.sqrt(sum)
		return -1
	
	def sigmoid(self, z):
		sig = math.exp(-z)
		sig = 1 + sig
		return 1.0/sig

	def probPred1(self, x):
		sum =0
		for i in range(len(x)):
			sum = sum + self.weights[i]*x[i]
		return self.sigmoid(sum)

	def predict(self, x):
		prob = self.probPred1(x)
		if prob >= .5:
			return 1
		return 0

	def printPerformance(self, test_instances):
		acc = 0.0
		tp = 0
		tn = 0
		fp = 0
		fn = 0

		for i in range(len(test_instances)):
			predicted_label = self.predict(test_instances[i].features)
			lable = test_instances[i].label
			if lable == predicted_label:
				if predicted_label ==1:
					tp =tp+1
				else:
					tn = tn+1
			else:
				if predicted_label == 1:
					fp =fp+1
				else:
					fn =fn+1

		acc = (tp + tn) / float(len(test_instances))

		print "Accaracy=" + str(acc)
		print "Confusion Matrix"
		print str(tp) + "  " +str(fn)
		print str(fp) + "  " +str(tn)

	def train(self, corpus):
		for n in range(self.iterations):
			lik = 0.0
			for i in range(len(corpus)):
				label = corpus[i].label
				features = corpus[i].features
				probPred = self.probPred1(features)
				for w in range(len(self.weights)):
					self.weights[w] = self.weights[w] + self.rate*features[w]*(label - probPred)- self.rate*self.lam*self.weights[w]
				localSum = 0
				for z in range(len(features)):
					localSum = localSum + features[z] *self.weights[z]
				lik = lik + label*localSum - math.log(1+math.exp(localSum))
			print "iteration: " + str(n) +" lik: " + str(lik)

class Perceptron:
	weights =None
	iterations =None
	rate = 0.01

	def __init__(self, n, iterations):
		self.weights = np.zeros((n,), dtype=float)
		self.iterations = iterations

	def weightsL2Norm(self):
		sum = 0
		for i in self.weights:
			sum = sum + i
		if sum >0:
			return math.sqrt(sum)
		return -1


	def probPred1(self, x):
		sum =0
		for i in range(len(x)):
			sum = sum + self.weights[i]*x[i]
		return sum

	def predict(self, x):
		prob = self.probPred1(x)
		if prob >= .5:
			return 1
		return -1

	def printPerformance(self, test_instances):
		acc = 0.0
		tp = 0
		tn = 0
		fp = 0
		fn = 0

		for i in range(len(test_instances)):
			predicted_label = self.predict(test_instances[i].features)
			label = test_instances[i].label
			if label == 0:
				label =-1
			if label == predicted_label:
				if predicted_label ==1:
					tp =tp+1
				else:
					tn = tn+1
			else:
				if predicted_label == 1:
					fp =fp+1
				else:
					fn =fn+1

		acc = (tp + tn) / float(len(test_instances))

		print "Accaracy=" + str(acc)
		print "Confusion Matrix"
		print str(tp) + "  " +str(fn)
		print str(fp) + "  " +str(tn)

	def train(self, corpus):
		for n in range(self.iterations):
			lik = 0.0
			for i in range(len(corpus)):
				label = corpus[i].label

				if label == 0:
					label =-1

				features = corpus[i].features
				activation = self.probPred1(features)
				if activation*label <=0:
					for w in range(len(self.weights)):
						self.weights[w] += features[w]*label
				#
				#localSum = 0
				#for z in range(len(features)):
				#	localSum = localSum + features[z] *self.weights[z]
				#lik = lik + label*localSum - math.log(1+math.exp(localSum))
			print "iteration: " + str(n) #+" lik: " + str(lik)


def main():
	corpus = corpusProcessor.dataMaker(.9)
	print "extracting corpus"
	test = corpus.test_set
	train = corpus.train_set
	trainBais = copy.deepcopy(train)
	for instance in trainBais:
		instance.features.append(1)
	testBais = copy.deepcopy(test)
	for instance in testBais:
		instance.features.append(1)


	model1 = LogisticRegression(len(test[1].features), 10)
	model2 = LogisticRegression(len(test[1].features)+1, 10)
	model3 = LogisticRegressionReg(len(test[1].features), 10)
	model4 = Perceptron(len(test[1].features)+1, 10)


	# for i in range (10,500,10):
	# 	print "=========================================================="
	# 	print "starting model normal"
	# 	print "iterations:  " +str(i)
		

	# 	print "training"
	# 	model1.train(train)

	# 	print "Norm of the learned weights = " + str(model1.weightsL2Norm())
	# 	print "Length of the weight vector = " + str(len(model1.weights))

	# 	print "-----------------Printing train set performance-----------------"
	# 	model1.printPerformance(train)

	# 	print "-----------------Printing test set performance-----------------"
	# 	model1.printPerformance(test)


	# for i in range(10, 500, 10):

	# 	print "=========================================================="
	# 	print "starting model with bais"
	# 	print "iterations:  " +str(i)
	# 	print "training"
	# 	model2.train(trainBais)

	# 	print "Norm of the learned weights = " + str(model2.weightsL2Norm())
	# 	print "Length of the weight vector = " + str(len(model2.weights))

	# 	print "-----------------Printing train set performance-----------------"
	# 	model2.printPerformance(trainBais)

	# 	print "-----------------Printing test set performance-----------------"
	# 	model2.printPerformance(testBais)

	# for i in range(10, 500, 10):

	# 	print "=========================================================="
	# 	print "starting model with L2 Regularization"
	# 	print "iterations:  " +str(i)

	# 	print "training"
	# 	model3.train(train)

	# 	print "Norm of the learned weights = " + str(model2.weightsL2Norm())
	# 	print "Length of the weight vector = " + str(len(model2.weights))

	# 	print "-----------------Printing train set performance-----------------"
	# 	model3.printPerformance(train)

	# 	print "-----------------Printing test set performance-----------------"
	# 	model3.printPerformance(test)


	for i in range(10, 500, 10):

		print "=========================================================="
		print "starting model with bais"
		print "iterations:  " +str(i)
		print "training"
		model4.train(trainBais)

		print "Norm of the learned weights = " + str(model4.weightsL2Norm())
		print "Length of the weight vector = " + str(len(model4.weights))

		print "-----------------Printing train set performance-----------------"
		model4.printPerformance(trainBais)

		print "-----------------Printing test set performance-----------------"
		model4.printPerformance(testBais)

if __name__ == "__main__":
	main()
''' knn practice '''

from sklearn import datasets
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
import heapq
import random
from collections import Counter
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix
import numpy as np
from numpy import *
from sklearn.model_selection import train_test_split

''' function '''

def split(num, rate, datasets):
	tr_x = []
	tr_y = []
	te_x = []
	te_y = []
	c =[]
	c1 =[]
	n = num * rate

	for i in range(0,50,1):
		c.append([datasets["data"][i],datasets["target"][i]])
	random.shuffle(c)

	for i in range(0,25,1):
		tr_x.append(c[i][0])
		tr_y.append(c[i][1])
	for i in range(25,50,1):
		te_x.append(c[i][0])
		te_y.append(c[i][1])

	c[:] = []
	for i in range(50,100,1):
		c.append([datasets["data"][i],datasets["target"][i]])
	random.shuffle(c)

	for i in range(0,25,1):
		tr_x.append(c[i][0])
		tr_y.append(c[i][1])
	for i in range(25,50,1):
		te_x.append(c[i][0])
		te_y.append(c[i][1])


	c[:] = []
	for i in range(100,150,1):
		c.append([datasets["data"][i],datasets["target"][i]])
	random.shuffle(c)

	for i in range(0,25,1):
		tr_x.append(c[i][0])
		tr_y.append(c[i][1])
	for i in range(25,50,1):
		te_x.append(c[i][0])
		te_y.append(c[i][1])

	tr_x = np.array(tr_x)
	tr_y = np.array(tr_y)
	te_x = np.array(te_x)
	te_y = np.array(te_y)

	return tr_x, tr_y, te_x, te_y


def dist(x, y):
	a =  np.sqrt(np.dot(np.subtract(x, y),(np.subtract(x, y).T)))
	return a


''' main '''

def main():

	#variable
	tr_y2 = []
	te_y2 = []
	error_rate_tr = []
	error_rate_te = []
	dist_arr = []
	min_dist_lebel = []
	min_dist = []
	lebels = []
	tr_error_count = 0
	te_error_count = 0
	te_error = 0
	tr_error = 0
	tr_acc = []
	te_acc = []

	#import data
	data = datasets.load_iris();


	#split
	tr_x, tr_y, te_x, te_y = split(150, 0.5, data)

	#shuffle
	c = list(zip(tr_x, tr_y))
	random.shuffle(c)
	tr_x, tr_y = zip(*c)
	c = list(zip(te_x, te_y))
	random.shuffle(c)
	te_x, te_y = zip(*c)

	print "---------------------- training ---------------------- \n"

	#training
	print "training datasets\n"
	for i in range(75):
		print tr_x[i],tr_y[i]
	print "\ntraining error rate:\n"
	for k in range(1,21,1):
		for i in range(0,75,1):
			for j in range(0,75,1):
				dist_arr.append(dist(tr_x[i], tr_x[j])) 
			min_dist = heapq.nsmallest(k, dist_arr)
		
			# find lebel
			for  l in range(k):
				for m in range(0,75,1):
					if min_dist[l] == dist_arr[m] :
						min_dist_lebel.append(tr_y[m])

			#give lebel
			lebel = Counter(min_dist_lebel).most_common(1)[0][0]
			tr_y2.append(lebel)

			#tr error rate
			if tr_y[i] != tr_y2[i] :
				tr_error_count = tr_error_count + 1

			#initialize dist
			min_dist_lebel[:] = []			
			dist_arr[:] = []
			min_dist[:] = []

		#print error rate
		tr_error_rate = tr_error_count / 75.0
		error_rate_tr.append(tr_error_rate)
		print "when k = %d tr_error_rate = %f  \n" %(k, tr_error_rate)
		
		tr_y2[:] = []
		tr_error_count = 0
		tr_error_rate = 0

	print "---------------------- testing ---------------------- \n"	
	print "testing datasets\n"
	for i in range(75):
		print te_x[i],te_y[i]
	print "\ntesting error rate:\n"
	#testing
	for k in range(1,21,1):
		for i in range(0,75,1):
			for j in range(0,75,1):
				dist_arr.append(dist(te_x[i], tr_x[j])) 
			min_dist = heapq.nsmallest(k, dist_arr)
		
			# find lebel
			for  l in range(k):
				for m in range(0,75,1):
					if min_dist[l] == dist_arr[m] :
						min_dist_lebel.append(tr_y[m])

			#give lebel
			lebel = Counter(min_dist_lebel).most_common(1)[0][0]
			te_y2.append(lebel)

			#te error rate
			if te_y[i] != te_y2[i] :
				te_error_count = te_error_count + 1

			#initialize dist
			min_dist_lebel[:] = []			
			dist_arr[:] = []
			min_dist[:] = []

		#print error rate
		te_error_rate = te_error_count / 75.0
		error_rate_te.append(te_error_rate)
		print "when k = %d te_error_rate = %f  \n" %(k, te_error_rate)
		
		te_y2[:] = []
		te_error_count = 0
		te_error_rate = 0

	
	k = np.arange(1, 21)
	''' Accuracy '''
	for i in range(0,20,1):
		tr_acc.append(1 - error_rate_tr[i])
		te_acc.append(1 - error_rate_te[i])

	
	''' Visualize the accuracy of the model'''

	plt.subplot(211)
	plt.plot(k, error_rate_tr)
	plt.xlabel("") 
	plt.ylabel("Error rate") 
	plt.title("The error rate of training data") 
	
	plt.subplot(212)
	plt.plot(k, error_rate_te)
	plt.xlabel("k") 
	plt.ylabel("Error rate") 
	plt.title("The error rate of testing data") 
	#plt.savefig("filename.png",dpi=300,format="png") 
	plt.show()

	plt.subplot(211)
	plt.plot(k, tr_acc)
	plt.xlabel("") 
	plt.ylabel("Accuracy") 
	plt.title("The accuracy of training data") 
	
	plt.subplot(212)
	plt.plot(k, te_acc)
	plt.xlabel("k") 
	plt.ylabel("Accuracy") 
	plt.title("The accuracy of testing data") 
	#plt.savefig("filename.png",dpi=300,format="png") 
	plt.show()


if __name__ == '__main__':
    main()


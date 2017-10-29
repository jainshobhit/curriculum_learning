import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import csv
import itertools

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class DistributionFit():
	def __init__(self, transform = None, numClasses=10, criterion = nn.CrossEntropyLoss(),
		optimizer = None, classes = [], net = None, train=True, test = True, numEpochs = 20):

		self.transform=transform
		self.numClasses=numClasses
		#self.numSubsets=numSubsets
		self.criterion = criterion
		self.optimizer = optimizer
		self.net = net
		self.train = train
		self.test = test
		self.numEpochs = numEpochs
		self.classes = classes

		self.trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                    	    download=False, transform=self.transform)
		self.testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                    	   download=False, transform=self.transform)


		#self.classes = random.sample(range(0,100),self.numClasses) 	#[34, 23, 56 ...]
		self.classDict={}
		self.reverseDict={}
		#self.reverseClassDict={}
		for label in range(self.numClasses):
			self.classDict[self.classes[label]]=label 	#{34:0, 23:1, 56:2 ...}
			self.reverseDict[label]=self.classes[label] 	#{0: 34, 1:23, 2:56 ...}
		#print self.classDict
	
		self.genTrainSet()
		self.genTestSet()

		self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=4,
        	                                  shuffle=True, num_workers=2)
		self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=4,
            	                             shuffle=False, num_workers=2)
	
		if self.train:
			self.Train()

		if self.test:
			#self.Test()
			self.testPerClass()

	#Generates a new training set containing a subset of the 100 classes
	def genTrainSet(self):
		new_data=[]
		new_labels=[]
		for index in range(len(self.trainset.train_labels)):
			if(self.trainset.train_labels[index] in self.classes):
				new_data.append(self.trainset.train_data[index])
				new_labels.append(self.classDict[self.trainset.train_labels[index]])
		new_data = np.array(new_data)
		self.trainset.train_data = new_data
		self.trainset.train_labels = new_labels

	#Generates a new test set containing a subset of the 100 classes
	def genTestSet(self):
		new_data=[]
		new_labels=[]
		for index in range(len(self.testset.test_labels)):
			if(self.testset.test_labels[index] in self.classes):
				new_data.append(self.testset.test_data[index])
				new_labels.append(self.classDict[self.testset.test_labels[index]])
		new_data = np.array(new_data)
		self.testset.test_data = new_data
		self.testset.test_labels = new_labels
	
	def Train(self):
		for epoch in range(self.numEpochs):  
			running_loss = 0.0
			for i, data in enumerate(self.trainloader, 0):
				# get the inputs
				inputs, labels = data

				# wrap them in Variable
				inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

				# zero the parameter gradients
				self.optimizer.zero_grad()

				# forward + backward + optimize
				outputs = self.net(inputs)
				loss = self.criterion(outputs, labels)
				loss.backward()
				self.optimizer.step()

				# print statistics
				running_loss += loss.data[0]
				if i % 1000 == 999:    # print every 1000 mini-batches and every mini-batch has 4 images
					print('[%d, %5d] loss: %.3f' %
						(epoch + 1, i + 1, running_loss / 1000))
					running_loss = 0.0
			if(epoch%10==0 or epoch==self.numEpochs-1):
				self.trainPerClass()
		print('Finished Training')

	#Outputs the training accuracy per class after every 10 epochs and writes results to a csv file
	def trainPerClass(self):
		class_correct = list(0. for i in range(self.numClasses))
		class_total = list(0. for i in range(self.numClasses))
		for data in self.trainloader:
			images, labels = data
			labels = labels.cuda()
			outputs = net(Variable(images.cuda()))
			_, predicted = torch.max(outputs.data, 1)
			c = (predicted == labels).squeeze()
			for i in range(4):
				label = labels[i]
				class_correct[label] += c[i]
				class_total[label] += 1

		for i in range(self.numClasses):
			print('Train accuracy of class %5s : %2d %%' % (
				self.reverseDict[i], 100 * class_correct[i] / class_total[i]))
		
		accuracyDict={}
		with open('train_results.csv', 'ab') as f:
			for i in range(self.numClasses):
				accuracyDict[self.reverseDict[i]] = 100 * class_correct[i] / class_total[i]
			writer = csv.DictWriter(f, accuracyDict.keys())
			writer.writerow({})
			writer.writeheader()
			writer.writerow(accuracyDict)

	def Test(self):
		correct = 0
		total = 0
		for data in self.testloader:
			images, labels = data
			outputs = net(Variable(images.cuda()))
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum()
		print total
		print correct
		print('Test accuracy of the network on the 1000 test images: %d %%' % (
			100 * correct / total))

	def testPerClass(self):
		class_correct = list(0. for i in range(self.numClasses))
		class_total = list(0. for i in range(self.numClasses))
		for data in self.testloader:
			images, labels = data
			labels = labels.cuda()
			outputs = net(Variable(images.cuda()))
			_, predicted = torch.max(outputs.data, 1)
			c = (predicted == labels).squeeze()
			for i in range(4):
				label = labels[i]
				class_correct[label] += c[i]
				class_total[label] += 1

		for i in range(self.numClasses):
			print('Accuracy of class %5s : %2d %%' % (
				self.reverseDict[i], 100 * class_correct[i] / class_total[i]))
		
		accuracyDict={}
		with open('test_results.csv', 'ab') as f:
			for i in range(self.numClasses):
				accuracyDict[self.reverseDict[i]] = 100 * class_correct[i] / class_total[i]
			writer = csv.DictWriter(f, accuracyDict.keys())
			writer.writerow({})
			writer.writeheader()
			writer.writerow(accuracyDict)


class Net(nn.Module):
    def __init__(self, numClasses):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, numClasses)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
	numClasses = 5
	numExp = 5
	classes = random.sample(range(0,100), numClasses)
	print classes
	for i in range(2, numClasses+1):
		subsets = list(itertools.combinations(classes,i))
		for S in subsets:
			for j in range(numExp):
				print S
				net = Net(numClasses)
				net.cuda()
				fit = DistributionFit(transform = transform, net = net, numClasses = len(S), classes = S,
				numEpochs = 5, optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9))

	# see how to add the training accuracies and also number of epochs
	#write these to csv file
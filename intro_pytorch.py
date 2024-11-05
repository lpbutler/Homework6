import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
	"""
	TODO: implement this function.

	INPUT: 
		An optional boolean argument (default value is True for training dataset)

	RETURNS:
		Dataloader for the training set (if training = True) or the test set (if training = False)
	"""
	custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

	train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=custom_transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle=True)
	test_set = datasets.FashionMNIST('./data', train=False, transform=custom_transform)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64)
	
	#TODO: do i need to put the classes of this project down?
	#classes = (’T-shirt/top’, ’ Trouser’, ’Pullover’, ’Dress’, ’Coat’, ’Sandal’, ’Shirt’, ’Sneaker’, ’Bag’,’ Ankle Boot’)

	if training == True:
		return train_loader
	else:
		return test_loader

def build_model():
	"""
	TODO: implement this function.

	INPUT: 
		None

	RETURNS:
		An untrained neural network model
	"""
	model = nn.Sequential(
	nn.Flatten(),
	nn.Linear(28*28, 128, bias=True),
	nn.ReLU(),
	nn.Linear(128, 64, bias=True),
	nn.ReLU(),
	nn.Linear(64, 10, bias=True)
	)
	return model



def train_model(model, train_loader, criterion, T):
	"""
	TODO: implement this function.

	INPUT: 
		model - the model produced by the previous function
		train_loader  - the train DataLoader produced by the first function
		criterion   - cross-entropy 
		T - number of epochs for training

	RETURNS:
		None
	"""
	model.train()
	
	opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	for e in range(T):
		epochLoss = 0.0
		correct = 0
		total = 0
		for i, data in enumerate(train_loader, 0):
			inputs, labels = data
			opt.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			opt.step()
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			epochLoss += loss.item()
		percent = (correct/total)*100
		a = round(percent, 2)
		l = round((epochLoss/len(train_loader)), 3)
		print("Train Epoch: " + str(e) + " Accuracy: " + str(correct) + "/" + str(total) + "(" + str(a) + "%)" + " Loss: " + str(l))

def evaluate_model(model, test_loader, criterion, show_loss = True):
	"""
	TODO: implement this function.

	INPUT: 
		model - the the trained model produced by the previous function
		test_loader    - the test DataLoader
		criterion   - cropy-entropy 

	RETURNS:
		None
	"""
	model.eval()
	correct = 0;
	total = 0;
	with torch.no_grad():
		for data in test_loader:
			images, labels = data
			outputs = model(images)
			loss = criterion(outputs, labels)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
		if show_loss == True:
			al = loss.item()
			averageLoss = round(al, 4)
			print("Average loss: " + str(averageLoss))
		percent = (correct/total)*100
		a = round(percent, 2)
		print("Accuracy: " + str(a) + "%")


def predict_label(model, test_images, index):
	"""
	TODO: implement this function.

	INPUT: 
		model - the trained model
		test_images   -  a tensor. test image set of shape Nx1x28x28
		index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


	RETURNS:
		None
	"""
	#TODO: obtain logits
	classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
	image = test_images[index].unsqueeze(0)
	logits = model(image)
	prob = F.softmax(logits, dim = 1)
	topProb, topClass = torch.topk(prob, 3)
	topProb = topProb.detach().numpy().tolist()[0]
	topClass = topClass.detach().numpy().tolist()[0]
	for i in range(3):
		p = round((topProb[i]*100),2) 
		print(classes[topClass[i]] + ": " + str(p) + "%")
		

if __name__ == '__main__':
	'''
	Feel free to write your own test code here to exaime the correctness of your functions. 
	Note that this part will not be graded.
	'''
	train_loader = get_data_loader(True)
	test_loader = get_data_loader(False)
	model = build_model()
	criterion = nn.CrossEntropyLoss()
	train_model(model, train_loader, criterion, 5)
	evaluate_model(model, test_loader, criterion, show_loss = False)
	evaluate_model(model, test_loader, criterion, show_loss = True)
	test_images = next(iter(test_loader))[0]
	predict_label(model, test_images, 1)




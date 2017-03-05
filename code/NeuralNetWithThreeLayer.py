# CS440
# P2
# Mingyu Zhu (BUid: U13914732)
# Team: Sailung Yeung, Jiadong Chen, Zhengyuan Jin

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import erf
import time

# record the start time of running the program
start = time.time()

class NeuralNet:
    """
    This class implements a simple 3 layer neural network.
    """
    
    def __init__(self, hidden_dim, input_dim, output_dim, epsilon):
        """
        Initializes the parameters of the neural network to random values
        """
        
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)/ np.sqrt(hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        self.epsilon = epsilon
        
    #--------------------------------------------------------------------------
    
    def compute_cost(self,X, y):
        """
        Computes the total loss on the dataset
        """
        num_samples = len(X)
        # Do Forward propagation to calculate our predictions
        z = X.dot(self.W) + self.b
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        # Calculate the cross-entropy loss
        cross_ent_err = -np.log(softmax_scores[range(num_samples), y])
        data_loss = np.sum(cross_ent_err)
        return 1./num_samples * data_loss
        
    
    #--------------------------------------------------------------------------
 
    def predict(self,x):
        """
        Makes a prediction based on current model parameters
        """
        
        # Do Forward Propagation
        z1 = x.dot(self.W1) + self.b1
        output1 = sigmoid_func(z1)
        z2 = output1.dot(self.W2) + self.b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)        
        
    #--------------------------------------------------------------------------
        
    def fit(self,h,X,y,num_epochs):
        """
        Learns model parameters to fit the data
        hdim = number of nodes in the hidden layer
        """
        num_samples = len(X)
        
        #For each epoch
        for i in range(0, num_epochs):    
            
            
            # Do Forward Propagation   
            input0 = X.dot(self.W1) + self.b1# weighted input
            hidden0 = sigmoid_func(input0) # hidden layer output after sigmoid function
            output0 = hidden0.dot(self.W2) + self.b2 #final output by the output layer
            softmax_scores = np.exp(output0) / np.sum(np.exp(output0), axis=1, keepdims=True)
            
            # Do Back Propagation
            # beta is the back propogation error
            beta3 = softmax_scores
            beta3[range(num_samples), y] -= 1 # to find the first beta
            # calculate the dot product of output and 
            # back propogation error to get the change of weight
            dW2 = (hidden0.T).dot(beta3)
            db2 = np.dot(np.ones(y.shape).T, beta3) 
            # to find the second beta for hidden layer
            beta2 = beta3.dot(self.W2.T)*(1-np.power(hidden0,2)) 
            # get the change of weight for hidden layer
            dW1 = np.dot(X.T, beta2)
            db1 = np.dot(np.ones(y.shape).T, beta2)
            
            
            # uncomment the following to use regularization lambda 
            # in order to deal with overfitting
            #dW2 += reg_lambda *self.W2
            #dW1 += reg_lambda *self.W1
            
            # Update model parameters using gradients  
            # Update weight and bias
            self.W1 += -epsilon* dW1
            self.b1 += -epsilon* db1
            self.W2 += -epsilon* dW2
            self.b2 += -epsilon* db2

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

#def sigmoid_func(x):
#    #return np.exp(-x)/((1+np.exp(-x))**2)
#    return 1/(1+np.exp(-x))
#    
#def sigmoid_func2(x):
#    return x / (1+x**2)**0.5
    
# used one of the sigmoid function: erf(sqrt(pi)/x * x)
# https://upload.wikimedia.org/wikipedia/commons/6/6f/Gjl-t%28x%29.svg
def sigmoid_func(x):
    x2 = (math.pi**0.5)*x/2
    return erf(x2) 

def plot_decision_boundary(pred_func):
    """
    Helper function to print the decision boundary given by model
    """
    # Set min and max values
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

#Train Neural Network on
linear = False

#A. linearly separable data
if linear:
    #load data
    X = np.genfromtxt('C:/Users/myzhu/Desktop/P2/code/DATA/ToyLinearX.csv', delimiter=',')
    y = np.genfromtxt('C:/Users/myzhu/Desktop/P2/code/DATA//ToyLinearY.csv', delimiter=',')
    y = y.astype(int)
    #plot data
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()
#B. Non-linearly separable data
else:
    #load data
    X = np.genfromtxt('C:/Users/myzhu/Desktop/P2/code/DATA/ToyMoonX.csv', delimiter=',')
    y = np.genfromtxt('C:/Users/myzhu/Desktop/P2/code/DATA//ToyMoonY.csv', delimiter=',')
    y = y.astype(int)
    #plot data
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

input_dim = 2 # input layer dimensionality
output_dim = 2 # output layer dimensionality
hidden_dim = 5 # hidden layer dimensionality

# Gradient descent parameters 
epsilon = 0.01
num_epochs = 5000
reg_lambda = 0.01 # use for solve overfitting



# Fit model
#----------------------------------------------
#Uncomment following lines after implementing NeuralNet
#----------------------------------------------
NN = NeuralNet(hidden_dim, input_dim, output_dim, epsilon)
NN.fit(hidden_dim,X,y,num_epochs)
#
# Plot the decision boundary
plot_decision_boundary(lambda x: NN.predict(x))
plt.title("Neural Net Decision Boundary")

# printout running time
print("It took " + str(time.time()-start) + " seconds.")
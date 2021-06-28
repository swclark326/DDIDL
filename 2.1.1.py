import torch
import numpy as np
x=torch.arange(12)              #creates tensor (vector) of first 12 numbers
X=x.reshape(3,4)                #reshapes vector x into a 3x4 tensor (matrix)
X.shape                         #gives dimension of tensor (for any type of tensor), output is a 1x1 tensor
x.numel()                       #this numel command is a function and outputs the int valuee of the number of entries. ie, 3x4 matrix outputs 12
torch.zeros((3,4))              #creates a 3x4 tensor of all zeros. command also works in higher dimensions
torch.ones((2,3,5))             #Creates a 2x3x4 tensor of all 1's. command works in higher/lower dimensions
torch.rand((1,2,5))             #creates a 1x2x5 tensor with random numbers in [0,1)
torch.tensor([[1,2,3],[4,5,6]]) #entry for a 2x3 tensor with manual entries.

###########################
#list of elementwise operation 
a=torch.tensor([1,4,5.])
b=torch.tensor([2,3, -1])
a+b   
a-b
a*b
a**b                        # a_i ^ b_i
torch.exp(x)                # elementwise expbenentiation of tensor x with base 'e'
###########################

A=torch.tensor([[1,4,5.]])
B=torch.tensor([[2,3, -1]])
torch.cat((A,B),dim=0)       # Concatination: torch.cat  requres you to tell it if you want to merge along which dimension (starting with 0!)
torch.cat((A,B),dim=1) 

A==B  # returns a tensor with boolean entries based on whether each element is 

####Broadcasting ############
# this one seems strange. If I try to add a 2x1 and 1x3 matrix the dimensions do not work.... However pytorch still does it...
# pytorch forces them to become 2x3 matrices by augmenting each vector with itself in teh appropriate dimension,
# then it adds elementwise

a+b.reshape(3,1)

####### Indexing and Slicing #######

a[2]                        #represents the second indexed element (so actually the third) in the tensor a
a[1:3]
#print(a[1:3])               # selects the second and third since 1:3 recovers integers in [1,3)  elements from the tensor

Y=torch.tensor([[0,1,2,3,4], [5,6,7,8,9]])
Y[0,3]= 1000                  # sets this single element to 1000

XX = torch.arange(12, dtype=torch.float32).reshape((3, 4))
#print(XX)
XX[0:2, 0:2]=9
#print(XX)
XX[0:2, :]=9                #
#print(XX)

#### In place vairable Definitions to save memory
# this does not really make sense to me yet.....

#print(id(a))
a+=b # This does not change teh id number for the variable a
#print(a)
#print('id(a):',id(a))

a=a+b # this changes the id number but performs the same operation, this means that we are using more memory internally to perform the same opperation
#print(a)
#print('id(a):',id(a))

c=torch.zeros_like(a)
#print( id(c), ':id(c) before')
c[:]=a+b                # In place, addition of [:] cuases the id the be reused for some reason
#print( id(c),':id(c): with in place variable def')
c=a+b                   # Wihtout [:] defines a new variable id number
#print( id(c), ':id(c) without in place variable def')

######### Conversions ##########
# converting to Numpy, standard python classes, and tensors!

Xnumpy= X.numpy()
type(Xnumpy)        #numpy.ndarray
type(X)             #torch.tensor
#1x1 tensors and numpy.ndarray can also be converted to float, int, etc with the standard python commands
onebyone=torch.tensor([1.2])
onebyone # torch tensor
onebyone.numpy() # numpy.ndarray
float(onebyone) # float, etch
 
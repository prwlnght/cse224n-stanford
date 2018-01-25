import numpy as np
import numpy.matlib
from q2_sigmoid import sigmoid, sigmoid_grad

"""Example - making a single layered network"""


#function to shuffle in unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(a.shape[0])
    return a[p], b[p]


# define data and targets

data = np.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])

classAND = np.array([data[:, 0] & data[:, 1]]).transpose()
classOR = np.array([data[:, 0] | data[:, 1]]).transpose()
classXOR = np.array([np.bitwise_xor(data[:, 0],  data[:, 1])]).transpose()



#choose a class
classes = classOR

#making noisy observations

nRepats = 30;
data = np.matlib.repmat(data, nRepats, 1)
classes = np.matlib.repmat(classes, nRepats, 1)
data = data + .15*np.random.rand(data.shape[0], data.shape[1])
data, classes = unison_shuffled_copies(data, classes)
[n_obs, n_features] = [data.shape[0], data.shape[1]]
n_output = 1
l_rate = 0.1
n_iterations = 100 #dont quite understand this

#initializing random weights
w_layer_1 = np.random.random([1,2])
b_layer_1 = np.random.random(1)

#initiazatlions for visualizations (todo for later)


err = np.zeros([1,n_iterations])

for i in range(n_obs):
    input = data[i]
    target = classes[i]


    #1.FORWARD PROPAGATE
    z_out = w_layer_1.dot(input) + b_layer_1
    a_out = sigmoid(z_out)

    #BACKWARD PROPAGATE
    #Calculate error derivative wrt. output
    delta_out = sigmoid_grad(z_out).dot(a_out-target)

    delta_out_w = delta_out.dot(input)
    delta_out_b = delta_out

    #update parameters
    w_layer_1 = w_layer_1 - l_rate


#activations



#initialize model parameters



print data


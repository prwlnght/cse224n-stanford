import numpy as np
import numpy.matlib
from q2_sigmoid import sigmoid, sigmoid_grad
import matplotlib.pyplot as plt
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
classes = classXOR

#making noisy observations

nRepats = 30;
data = np.matlib.repmat(data, nRepats, 1)
classes = np.matlib.repmat(classes, nRepats, 1)
data = data + .15*np.random.rand(data.shape[0], data.shape[1])
data, classes = unison_shuffled_copies(data, classes)
[n_obs, n_features] = [data.shape[0], data.shape[1]]
n_output = 1
l_rate = .5
n_iterations = 100

n_hidden = 3

#initializing random weights
w_layer_1 = np.random.random([n_output,n_hidden])
b_layer_1 = np.random.random(1)
w_hid = np.random.random([n_hidden, n_features])
b_hid = np.random.random([n_hidden])

#initiazatlions for visualizations (todo for later)


err = np.zeros(n_obs)
w1s = np.zeros(n_iterations)
w2s = np.zeros(n_iterations)
bs = np.zeros(n_iterations)
mse = np.zeros(n_iterations)

for j in range(n_iterations):
    for i in range(n_obs):
        input = data[i]
        target = classes[i]


        #1.FORWARD PROPAGATE
        z_hid = w_hid.dot(input) + b_hid
        a_hid = sigmoid(z_hid)
        z_out = w_layer_1.dot(a_hid) + b_layer_1
        a_out = sigmoid(z_out)

        #BACKWARD PROPAGATE
        #Calculate error derivative wrt. output
        delta_out = sigmoid_grad(a_out).dot(a_out-target)
        delta_hid = sigmoid_grad(a_hid)*np.sum(delta_out*w_layer_1)


        #calculate gradeitn wrt parameters
        delta_out_w = delta_out*a_hid
        delta_out_b = delta_out
        delta_out_hid_w = np.outer(delta_hid, input)
        delta_out_hid_b = delta_hid*1

        #update parameters
        w_layer_1 = w_layer_1 - l_rate*delta_out_w
        b_layer_1 = b_layer_1 - l_rate*delta_out_b

        w_hid = w_hid - l_rate*delta_out_hid_w
        b_hid = b_hid - l_rate*delta_out_hid_b

        #calculate error function
        err[i] = .5*((a_out- target)**2)
        #bs[i] = delta_out_b
        print err[i]

        #plot params
    mse[j] = np.mean(err)
    w1s[j] = w_layer_1[0][1]
    w2s[j] = w_layer_1[0][1]
    bs[j] = b_layer_1

plt.plot(mse)
plt.show()


#activations



#initialize model parameters



print data


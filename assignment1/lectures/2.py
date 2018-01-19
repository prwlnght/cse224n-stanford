import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import timeit

# la = np.linalg
#
# words = ["I", "like", "enjoy", "deep", "learning", "NLP", "flying", "."]
#
# fig = plt.figure()
# plt.axis([-1, 1, -1, 1])
#
# X = np.array([
#     [0, 2, 1, 0, 0, 0, 0, 0],
#     [2, 0, 0, 1, 0, 1, 0, 0],
#     [1, 0, 0, 0, 0, 0, 1, 0],
#     [0, 1, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 1],
#     [0, 1, 0, 0, 0, 0, 0, 1],
#     [0, 0, 1, 0, 0, 0, 0, 1],
#     [0, 0, 0, 0, 1, 1, 1, 0],
# ])
#
# #compute the SVD
#
# U, s, Vh = la.svd(X, full_matrices=False)
#
# for i in xrange(len(words)):
#     plt.text(Vh[i, 4], Vh[i, 5], words[i])
#
# plt.show()


"""now looking into efficienty issues"""

setup_statement = """\
from numpy import random

N = 500 #number of windows to classify
d = 300 #dimensionality of each widnow
C = 5 #number of classes-entities
W = random.rand(C, d)
wordvectors_list = [random.rand(d,1) for i in range(N)]
wordvectors_one_matrix = random.rand(d,N)
"""

s1 = """\
#print "this"
[W.dot(wordvectors_list[i]) for i in range(N)]
"""

s2 = """
W.dot(wordvectors_one_matrix)
"""

print(timeit.timeit(s1, setup=setup_statement))
print(timeit.timeit(s2, setup=setup_statement))


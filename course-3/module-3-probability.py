import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


for n in [2.5, 0.3, 2.8, 0.5]:
    print('{0} - {1}'.format(n, sigmoid(n)))

print sigmoid(2.5) * (1 - sigmoid(0.3)) * sigmoid(2.8) * sigmoid(0.5)

print 2.5 * (1 - sigmoid(2.5)) + 0.3 * (0 - sigmoid(0.3)) + \
      2.8 * (1 - sigmoid(2.8)) + 0.5 * (1 - sigmoid(0.5))

import numpy
import math

# joint distribution[y][x]
joint_distribution = numpy.array(
    [[1/8, 1/16, 1/16, 1/4],
    [1/16, 1/8, 1/16, 0],
    [1/32, 1/32, 1/16, 0],
    [1/32, 1/32, 1/16, 0]]
)

print(joint_distribution[0])


def entropy(probability_distribution):
    entropy = 0
    for prob in probability_distribution:
        entropy += prob * math.log(prob ,2)
    return -entropy

def joint_entropy(joint_distribution):
    len_y = joint_distribution.shape[0]
    len_x = joint_distribution.shape[1]
    entropy = 0
    for y in range(len_y):
        for x in range(len_x):
            entropy += joint_distribution[y, x] * math.log(joint_distribution[y, x], 2)
    return -entropy


def conditional_entropy(joint_distribution):
    """
    Computes the conditional entropy H(x|y), using
    an array where y are the rows and x are the colums,
    like so : joint distribution[y][x]
    """
    len_y = joint_distribution.shape[0]
    len_x = joint_distribution.shape[1]
    entropy = 0
    for y in range(len_y):
        prob_y = sum(joint_distribution[y])
        for x in range(len_x):
            entropy += joint_distribution[y,x] * math.log(joint_distribution[y, x]/prob_y, 2)
    return -entropy


def mutual_information(joint_distribution):
    """
    Computes the mutual information I(x|y), using
    a numpy array where y are the rows and x are the colums,
    like so : joint distribution[y][x]
    """
    len_x = joint_distribution.shape[1]
    prob_x= [sum(joint_distribution[:,i]) for i in range(len_x)]

    return entropy(prob_x) - conditional_entropy(joint_distribution)



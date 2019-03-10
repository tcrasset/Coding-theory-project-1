import numpy
import math


def entropy(probability_distribution):
    """
    Computes H(X), the entropy of a random variable X, given its probability
    distribution.
    """
    entropy = 0
    for prob in probability_distribution:
        entropy += prob * math.log(prob ,2)
    return -entropy

def joint_entropy(joint_distribution):
    """
    Computes H(X,Y), the joint entropy of two discrete random variable X
    and Y, given their joint probability distribution.
    """
    len_y = joint_distribution.shape[0]
    len_x = joint_distribution.shape[1]
    entropy = 0
    for y in range(len_y):
        for x in range(len_x):
            entropy += joint_distribution[y, x] * math.log(joint_distribution[y, x], 2)
    return -entropy


def conditional_entropy(joint_distribution):
    """
    Computes H(X|Y), the conditional entropy of two discrete random variable X
    and Y, given their joint probability distribution.
    """
    len_y = joint_distribution.shape[0]
    len_x = joint_distribution.shape[1]
    entropy = 0
    for y in range(len_y):
        prob_y = sum(joint_distribution[y,:])
        for x in range(len_x):
            entropy += joint_distribution[y,x] * math.log(joint_distribution[y, x]/prob_y, 2)
    return -entropy


def mutual_information(joint_distribution):
    """
    Computes I(X;Y), the mutual information between two discrete random variable X
    and Y, given their joint probability distribution.
    """
    len_y = joint_distribution.shape[0]
    len_x = joint_distribution.shape[1]
    entropy = 0
    for y in range(len_y):
        prob_y = sum(joint_distribution[y,:])
        for x in range(len_x):
            prob_x = sum(joint_distribution[:,x])
            entropy += joint_distribution[y,x] * math.log(joint_distribution[y, x]/(prob_x * prob_y), 2)
    return -entropy



if __name__ == "__main__":
    # Joint distribution[y][x]
    joint_distribution = numpy.array(
        [[1/8, 1/16, 1/16, 1/4],
        [1/16, 1/8, 1/16, 0],
        [1/32, 1/32, 1/16, 0],
        [1/32, 1/32, 1/16, 0]]
    )


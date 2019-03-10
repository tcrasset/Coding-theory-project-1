import numpy as np
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
    Computes H(X,Y), the joint entropy of two discrete random variables
    X and Y, given their joint probability distribution.
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
    Computes H(X|Y), the conditional entropy of two discrete random
    variables X and Y, given their joint probability distribution.
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
    Computes I(X;Y), the mutual information between two discrete random
    variables X and Y, given their joint probability distribution.
    """
    len_y = joint_distribution.shape[0]
    len_x = joint_distribution.shape[1]
    entropy = 0
    for y in range(len_y):
        prob_y = sum(joint_distribution[y,:])
        for x in range(len_x):
            prob_x = sum(joint_distribution[:,x])
            entropy += joint_distribution[y,x] * math.log(joint_distribution[y, x]/(prob_x * prob_y), 2)
    return entropy


def cond_joint_entropy(joint_distribution):
    """
    Computes H(X,Y|Z), the conditional joint entropy of three discrete
    random variable X and Y, given their joint probability distribution.
    """
    len_y = joint_distribution.shape[0]
    len_x = joint_distribution.shape[1]
    len_z = joint_distribution.shape[2]
    entropy = 0
    for y in range(len_y):
        for x in range(len_x):
            for z in range(len_z):
                prob_z = sum(joint_distribution[:,:,z])
                entropy += joint_distribution[y,x,z] * math.log(joint_distribution[y,x,z]/prob_z, 2)
    return -entropy


def cond_mutual_information(joint_distribution):
    """
    Computes I(X;Y|Z), the conditional mutual information between three
    discrete random variables X and Y, given their joint probability
    distribution.
    """
    len_y = joint_distribution.shape[0]
    len_x = joint_distribution.shape[1]
    len_z = joint_distribution.shape[2]
    entropy = 0
    for y in range(len_y):
        for x in range(len_x):
            for z in range(len_z):
                prob_z = sum(joint_distribution[:,:,z])
                prob_x_z = sum(joint_distribution[:,x,z])
                prob_y_z = sum(joint_distribution[y,:,z])
                entropy += joint_distribution[y,x,z] * math.log((joint_distribution[y, x,z]*prob_z)/(prob_x_z * prob_y_z), 2)
    return entropy


def compute_probas_from_joint(joint_xy):
    """
    """
    len_y = joint_xy.shape[0]
    len_x = joint_xy.shape[1]
    probas_Y = np.zeros(len_y)
    probas_X = np.zeros(len_x)
    
    # Compute P(X)
    for x in range(len_x):
        probas_X[x] = sum(joint_xy[:,x])
    # Compute P(Y)
    for y in range(len_y):
        probas_Y[y] = sum(joint_xy[y,:])
    return probas_X, probas_Y


def compute_probas_from_table(table, joint_xy):
    """
    """
    len_y = table.shape[0]
    len_x = table.shape[1]
    len_z = 2
    probas_Z = np.zeros(len_z)
    joint_xz = np.zeros((len_x, len_z))
    joint_yz = np.zeros((len_y, len_z))
    joint_xyz = np.zeros((len_y, len_x, len_z))

    # Get P(x) and P(Y)
    probas_X, probas_Y = compute_probas_from_joint(joint_xy)
    
    # Compute P(Z)
    count = 0
    for y in range(len_y):
        for x in range(len_x):
            if table[y,x] == 0:
                count += 1
    probas_Z[0] = count/(len_x*len_y)
    probas_Z[1] = 1 - probas_Z[0]

    # Compute joint between X and Z
    for x in range(len_x):
        count = 0
        for y in range(len_y):
            if table[y,x] == 0:
                count += 1
        joint_xz[x,0] = (count/len_y)*probas_X[x]
        joint_xz[x,1] = (1-(count/len_y))*probas_X[x]

    # Compute joint between Y and Z
    for y in range(len_y):
        count = 0
        for x in range(len_x):
            if table[y,x] == 0:
                count += 1
        joint_yz[y,0] = (count/len_x)*probas_Y[y]
        joint_yz[y,1] = (1-(count/len_x))*probas_Y[y]

    # Compute joint between X, Y and Z
    for y in range(len_y):
        for x in range(len_x):
            if table[y,x] == 0:
                joint_xyz[y,x,0] = joint_xy[y,x]
            else:
                joint_xyz[y,x,1] = joint_xy[y,x]

    return (probas_Z, joint_xz, joint_yz, joint_xyz)



if __name__ == "__main__":
    # Joint distribution[y, x]
    joint_xy = np.array(
        [[1/8, 1/16, 1/16, 1/4],
        [1/16, 1/8, 1/16, 0],
        [1/32, 1/32, 1/16, 0],
        [1/32, 1/32, 1/16, 0]])
    # Table for w
    table_w = np.identity(4, int)
    # Table for z
    table_z = np.ones((4,4), int)
    np.fill_diagonal(table_z, 0)

    p_x, p_y = compute_probas_from_joint(joint_xy)
    print(p_x)
    print(p_y)

    p_z, joint_xz, joint_yz, joint_xyz = compute_probas_from_table(table_w,joint_xy)
    print(p_z)
    print(joint_xz)
    print(joint_yz)
    print(joint_xyz)

import numpy as np
import math


def entropy(probability_distribution):
    """
    Computes H(X), the entropy of a random variable X, given its probability
    distribution.
    """
    entropy = 0
    for prob in probability_distribution:
        if prob != 0:
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
            if joint_distribution[y,x] != 0:
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
        prob_y = np.sum(joint_distribution[y,:])
        for x in range(len_x):
            if (joint_distribution[y,x] != 0) and (prob_y != 0):
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
        prob_y = np.sum(joint_distribution[y,:])
        for x in range(len_x):
            prob_x = np.sum(joint_distribution[:,x])
            if (joint_distribution[y,x] != 0) and (prob_y != 0) and (prob_x != 0):
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
                prob_z = np.sum(joint_distribution[:,:,z])
                if (joint_distribution[y,x,z] != 0) and (prob_z != 0):
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
                prob_z = np.sum(joint_distribution[:,:,z])
                prob_x_z = np.sum(joint_distribution[:,x,z])
                prob_y_z = np.sum(joint_distribution[y,:,z])
                if (joint_distribution[y,x,z] != 0) and (prob_x_z != 0) and (prob_y_z != 0):
                    entropy += joint_distribution[y,x,z] * math.log((joint_distribution[y, x,z]*prob_z)/(prob_x_z * prob_y_z), 2)
    return entropy


def compute_probas_from_joint(joint_xy):
    """
    Compute the probability distributions P(X) and P(Y) from the joint
    probability distribution P(X,Y).
    """
    len_y = joint_xy.shape[0]
    len_x = joint_xy.shape[1]
    probas_Y = np.zeros(len_y)
    probas_X = np.zeros(len_x)
    
    # Compute P(X)
    for x in range(len_x):
        probas_X[x] = np.sum(joint_xy[:,x])
    # Compute P(Y)
    for y in range(len_y):
        probas_Y[y] = np.sum(joint_xy[y,:])
    return probas_X, probas_Y


def compute_probas_from_table(table, joint_xy):
    """
    Compute the probability distributions P(Z) as well as the joint
    probability distributions P(X,Z), P(Y,Z) and P(X,Y,Z) given a
    table of relations between X, Y and Z as given in the statement,
    and the joint probability distribution P(X,Y).
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

    # Compute the joint probability distributions
    p_x, p_y = compute_probas_from_joint(joint_xy)
    p_w, joint_xw, joint_yw, joint_xyw = compute_probas_from_table(table_w,joint_xy)
    p_z, joint_xz, joint_yz, joint_xyz = compute_probas_from_table(table_z,joint_xy)
    joint_wz = np.ones((2,2), int)
    np.fill_diagonal(joint_wz, 0)
    # print("Probability distribution of X :")
    # print(p_x)
    # print("Probability distribution of Y :")
    # print(p_y)
    # print("Probability distribution of W :")
    # print(p_w)
    # print("Probability distribution of Z :")
    # print(p_z)
    # print("Joint probability distribution between X and W :")
    # print(joint_xw)
    # print("Joint probability distribution between Y and W :")
    # print(joint_yw)
    # print("Joint probability distribution between X, Y and W :")
    # print(joint_xyw)
    # print("Joint probability distribution between X and Z :")
    # print(joint_xz)
    # print("Joint probability distribution between Y and Z :")
    # print(joint_yz)
    # print("Joint probability distribution between X, Y and Z :")
    # print(joint_xyz)

    # # Compute the entropy of the random variables
    # print("Entropy of X : {:.3f}".format(entropy(p_x)))
    # print("Entropy of Y : {:.3f}".format(entropy(p_y)))
    # print("Entropy of Z : {:.3f}".format(entropy(p_w)))
    # print("Entropy of W : {:.3f}\n".format(entropy(p_z)))

    # # Compute the joint entropy
    # print("Joint entropy of X and Y : {:.3f}".format(joint_entropy(joint_xy)))
    # print("Joint entropy of X and W : {:.3f}".format(joint_entropy(joint_xw)))
    # print("Joint entropy of Y and W : {:.3f}".format(joint_entropy(joint_yw)))
    # print("Joint entropy of W and Z : {:.3f}\n".format(joint_entropy(joint_wz)))

    # # Compute the conditional entropy
    # print("Conditional entropy of X knowing Y : {:.3f}".format(conditional_entropy(joint_xy)))
    # print("Conditional entropy of W knowing X : {:.3f}".format(conditional_entropy(joint_xw)))
    # print("Conditional entropy of Z knowing W : {:.3f}".format(conditional_entropy(np.transpose(joint_wz))))
    # print("Conditional entropy of W knowing Z : {:.3f}\n".format(conditional_entropy(joint_wz)))

    # # Compute the mutual information
    # print("Mutual information between X and Y : {:.3f}".format(mutual_information(joint_xy)))
    # print("Mutual information between X and W : {:.3f}".format(mutual_information(joint_xw)))
    # print("Mutual information between Y and Z : {:.3f}".format(mutual_information(joint_yz)))
    # print("Mutual information between W and Z : {:.3f}\n".format(mutual_information(joint_wz)))

    # Compute the conditional joint entropy
    """
    !!! joint_xyz not correct I think because when computing p_z in the function by making the sum I get [0.6875 0.3125]
    while it should be [0.75 0.25] !!!
    """
    print("Conditional joint entropy of X and Y knowing Z : {:.3f}\n".format(cond_joint_entropy(joint_xyz)))

    # Compute the conditional mutual information
    """
    !!! Same for here, function seems ok but joint_xyz may not be correct !!!
    """
    print("Conditional mutual information of X and Y knowing Z : {:.3f}\n".format(cond_mutual_information(joint_xyz)))
    
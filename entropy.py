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


def conditional_entropy(joint_distribution, distribution_cond):
    """
    Computes H(X|Y), the conditional entropy of two discrete random
    variables X and Y, given their joint probability distribution and
    the probability distribution of the condition.
    """
    len_y = joint_distribution.shape[0]
    len_x = joint_distribution.shape[1]
    entropy = 0
    for y in range(len_y):
        for x in range(len_x):
            if (joint_distribution[y,x] != 0) and (distribution_cond[y] != 0):
                entropy += joint_distribution[y,x] * math.log(joint_distribution[y, x]/distribution_cond[y], 2)
    return -entropy


def mutual_information(joint_distribution, distribution_var1, distribution_var2):
    """
    Computes I(X;Y), the mutual information between two discrete random
    variables X and Y, given their joint probability distribution and
    the probability distributions of the two variables.
    """
    len_y = joint_distribution.shape[0]
    len_x = joint_distribution.shape[1]
    entropy = 0
    for y in range(len_y):
        for x in range(len_x):
            if (joint_distribution[y,x] != 0 and
                distribution_var1[x] != 0 and
                distribution_var2[y] != 0):
                entropy += joint_distribution[y,x] * math.log(joint_distribution[y, x]/(distribution_var1[x] * distribution_var2[y]), 2)
    return entropy


def cond_joint_entropy(cond_distribution, distribution_cond):
    """
    Computes H(X,Y|Z), the conditional joint entropy of the discrete
    random variables X and Y knowing Z, given their joint probability
    distribution and the probability distribution of the condition.
    """
    len_z = cond_distribution.shape[2]
    entropy = 0
    for z in range(len_z):
        entropy += distribution_cond[z] * joint_entropy(cond_distribution[:,:,z])
    return -entropy


def cond_mutual_information(cond_distribution, cond_var1, cond_var2, distribution_cond):
    """
    Computes I(X;Y|Z), the conditional mutual information between the
    discrete random variables X and Y knowing Z, given their conditional
    probability distribution P(X,Y|Z), the conditional probability
    distributions P(X|Z) and P(Y|Z), and the probability distribution of
    the condition P(Z).
    """
    len_z = cond_distribution.shape[2]
    entropy = 0
    for z in range(len_z):
        entropy += distribution_cond[z] * mutual_information(cond_distribution[:,:,z], cond_var1[:,z], cond_var2[:,z])
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
    joint_xz = np.zeros((len_z, len_x))
    joint_yz = np.zeros((len_z, len_y))
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
        joint_xz[0,x] = (count/len_y)*probas_X[x]
        joint_xz[1,x] = (1-(count/len_y))*probas_X[x]

    # Compute joint between Y and Z
    for y in range(len_y):
        count = 0
        for x in range(len_x):
            if table[y,x] == 0:
                count += 1
        joint_yz[0,y] = (count/len_x)*probas_Y[y]
        joint_yz[1,y] = (1-(count/len_x))*probas_Y[y]

    # Compute conditional P(X,Y|Z)
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
    joint_wz = np.asarray([[0, 0.25],
                            [0.75, 0]])
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
    # # print("Joint probability distribution between X, Y and W :")
    # # print(joint_xyw)
    # print("Joint probability distribution between X and Z :")
    # print(joint_xz)
    # print("Joint probability distribution between Y and Z :")
    # print(joint_yz)
    # # print("Joint probability distribution between X, Y and Z :")
    # # print(joint_xyz)

    # # Q7 : Compute the entropy of the random variables
    # print("Entropy of X : {:.3f}".format(entropy(p_x)))
    # print("Entropy of Y : {:.3f}".format(entropy(p_y)))
    # print("Entropy of Z : {:.3f}".format(entropy(p_w)))
    # print("Entropy of W : {:.3f}\n".format(entropy(p_z)))

    # # Q8 : Compute the joint entropy
    # print("Joint entropy of X and Y : {:.3f}".format(joint_entropy(joint_xy)))
    # print("Joint entropy of X and W : {:.3f}".format(joint_entropy(joint_xw)))
    # print("Joint entropy of Y and W : {:.3f}".format(joint_entropy(joint_yw)))
    # print("Joint entropy of W and Z : {:.3f}\n".format(joint_entropy(joint_wz)))

    # # Q9 : Compute the conditional entropy
    # print("Conditional entropy of X knowing Y : {:.3f}".format(conditional_entropy(joint_xy, p_y)))
    # print("Conditional entropy of W knowing X : {:.3f}".format(conditional_entropy(np.transpose(joint_xw), p_x)))
    # print("Conditional entropy of Z knowing W : {:.3f}".format(conditional_entropy(np.transpose(joint_wz), p_w)))
    # print("Conditional entropy of W knowing Z : {:.3f}\n".format(conditional_entropy(joint_wz, p_z)))

    # # Q10 : Compute the mutual information
    # print("Mutual information between X and Y : {:.3f}".format(mutual_information(joint_xy, p_x, p_y)))
    # print("Mutual information between X and W : {:.3f}".format(mutual_information(joint_xw, p_x, p_w)))
    # print("Mutual information between Y and Z : {:.3f}".format(mutual_information(joint_yz, p_y, p_z)))
    # print("Mutual information between W and Z : {:.3f}\n".format(mutual_information(joint_wz, p_w, p_z)))

    # # Q11 : Compute the conditional joint entropy
    # """
    # print("Conditional joint entropy of X and Y knowing W : {:.3f}".format(cond_joint_entropy(joint_xyw)))
    # print("Conditional joint entropy of W and Z knowing X : {:.3f}\n".format(cond_joint_entropy(joint_xyw)))

    # # Q11 : Compute the conditional mutual information
    # print("Conditional mutual information of X and Y knowing Z : {:.3f}".format(cond_mutual_information(joint_xyz)))
    # print("Conditional mutual information of W and Z knowing X : {:.3f}\n".format(cond_mutual_information(joint_xyw)))

    # # Q13 : Entropy of a single square
    # prob = np.full(9, 1/9)
    # print("Entropy of a single square : {:.3f}".format(entropy(prob)))

    # # Q14 : Entropy of the subgrid
    # result = 0
    # i = 6
    # while i >= 1:
    #     prob = np.full(i, 1/i)
    #     result += entropy(prob)
    #     i -= 1
    # print(result)



    
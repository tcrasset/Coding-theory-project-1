import numpy as np
import math


#-----------------------------------------------------------------------------------------------
# QUESTIONS 7 to 11
#-----------------------------------------------------------------------------------------------
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


def cond_joint_entropy(joint_distribution, distribution_cond):
    """
    Computes H(X,Y|Z), the conditional joint entropy of the discrete
    random variables X and Y knowing Z, given their joint probability
    distribution and the probability distribution of the condition.
    """
    len_y = joint_distribution.shape[0]
    len_x = joint_distribution.shape[1]
    len_z = joint_distribution.shape[2]
    entropy = 0
    for y in range(len_y):
        for x in range(len_x):
            for z in range(len_z):
                if (joint_distribution[y,x,z] != 0 and
                distribution_cond[z] != 0):
                    entropy += joint_distribution[y,x,z] * math.log(joint_distribution[y,x,z]/distribution_cond[z], 2)
    return -entropy


def cond_mutual_information(joint_distribution, distribution_cond, joint_var1, joint_var2):
    """
    Computes I(X;Y|Z), the conditional mutual information between the
    discrete random variables X and Y knowing Z, given their conditional
    probability distribution P(X,Y|Z), the conditional probability
    distributions P(X|Z) and P(Y|Z), and the probability distribution of
    the condition P(Z).
    """
    len_y = joint_distribution.shape[0]
    len_x = joint_distribution.shape[1]
    len_z = joint_distribution.shape[2]
    entropy = 0
    for y in range(len_y):
        for x in range(len_x):
            for z in range(len_z):
                if (joint_distribution[y,x,z] != 0 and
                distribution_cond[z] != 0 and
                joint_var1[z,x] != 0 and
                joint_var1[z,y] != 0):
                    entropy += joint_distribution[y,x,z] * math.log((joint_distribution[y,x,z]*distribution_cond[z])/(joint_var1[z,x] * joint_var2[z,y]), 2)
    return entropy


#-----------------------------------------------------------------------------------------------
# Compute probabilities from given tables
#-----------------------------------------------------------------------------------------------
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
    sum_pxy = 0
    for y in range(len_y):
        for x in range(len_x):
            if table[y,x] == 1:
                sum_pxy += joint_xy[y,x]
    probas_Z[0] = sum_pxy
    probas_Z[1] = 1 - probas_Z[0]

    # Compute joint between X and Z
    for x in range(len_x):
        sum_y0 = 0
        sum_y1 = 0
        for y in range(len_y):
            if table[y,x] == 1: # Z = 1
                sum_y1 += joint_xy[y][x]
            if table[y,x] == 0: # Z = 0
                sum_y0 += joint_xy[y][x]  
        joint_xz[0,x] = sum_y0
        joint_xz[1,x] = sum_y1

    # Compute joint between Y and Z
    for y in range(len_y):
        sum_x0 = 0
        sum_x1 = 0
        for x in range(len_x):
            if table[y,x] == 1: # Z = 1
                sum_x1 += joint_xy[y][x]
            if table[y,x] == 0: # Z = 0
                sum_x0 += joint_xy[y][x]  
        joint_yz[0,y] = sum_x0
        joint_yz[1,y] = sum_x1


    # Compute joint P(X,Y,Z)
    for y in range(len_y):
        for x in range(len_x):
            if table[y,x] == 0:
                joint_xyz[y,x,0] = joint_xy[y,x]
            else:
                joint_xyz[y,x,1] = joint_xy[y,x]

    return (probas_Z, joint_xz, joint_yz, joint_xyz)

def compute_joint_wzx(joint_wz, p_x):
    """
    Compute the joint probability distribution P(W,Z,X) from P(W,Z) and P(X)
    knowing that P(W,Z|X) = P(W,Z)
    """
    len_z = joint_wz.shape[0]
    len_w = joint_wz.shape[1]
    len_x = len(p_x)
    joint_wzx = np.zeros((len_z, len_w, len_x))
    for z in range(len_z):
        for w in range(len_w):
            for x in range(len_x):
                joint_wzx[z,w,x] = joint_wz[z,w]*p_x[x]
    print("joint_wzx",joint_wzx[0,:,:])
    print("joint_wzx",joint_wzx[1,:,:])
    return joint_wzx
    

#-----------------------------------------------------------------------------------------------
# Functions used for answering Question 15
#-----------------------------------------------------------------------------------------------
def union(lst1, lst2, lst3):
    """
    Compute the union of lists without repetitionwithout zero.
    """
    final_list = list(set(lst1) | set(lst2) |set(lst3))
    final_list = [x for x in final_list if x!=0]
    return final_list


def entropy_unsolved_sudoku(sudoku):
    """
    Compute the entropy of the given ensolved sudoku.
    """
    grid_index_arr = np.zeros((9,9))
    n = []
    #Create grid index for every cell
    for r in range(sudoku.shape[0]):
        for c in range(sudoku.shape[1]):
            grid_index_arr[r,c] = int(c / 3 + r - r % 3)

    entropy = 0
    for r in range(sudoku.shape[0]):
        for c in range(sudoku.shape[1]):
            if(sudoku[r,c] ==0): #Empty cells only
                row = sudoku[r,:]
                col = sudoku[:,c]
                grid_index = c / 3 + r - r % 3
                subgrid = sudoku[np.where(grid_index_arr == grid_index)]
                #Digits that could be filled in column, row and subgrid
                n_digits = union(row,col,subgrid)
                n.append(9 - len(n_digits))

    entropy = sum([math.log(x,2) for x in n])
    return entropy


#-----------------------------------------------------------------------------------------------
# To print question number on the terminal
#-----------------------------------------------------------------------------------------------
def print_in_a_frame(message, symbol):
    size = len(message)
    print("\n")
    print(symbol * (size + 4))
    print('* {:<{}} *'.format(message, size))
    print(symbol * (size + 4))


#-----------------------------------------------------------------------------------------------
# QUESTIONS 12 to 15
#-----------------------------------------------------------------------------------------------
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
    joint_wz = np.asarray([[0, 5/16],
                            [11/16, 0]])
    joint_wzx = compute_joint_wzx(joint_wz, p_x)
    # print_in_a_frame("Probability distributions", '=')
    # print("Probability distribution of X :\n", p_x)
    # print("Probability distribution of Y :\n", p_y)
    # print("Probability distribution of W :\n", p_w)
    # print("Probability distribution of Z :\n", p_z)
    # print("Joint probability distribution between X and W :\n", joint_xw)
    # print("Joint probability distribution between Y and W :\n", joint_yw)
    # print("Joint probability distribution between X and Z :\n", joint_xz)
    # print("Joint probability distribution between Y and Z :\n", joint_yz)
    # print("Joint probability distribution between W and Z :\n", joint_wz)
    # print("Joint probability distribution between X, Y and W :\n", joint_xyw)
    # print("Joint probability distribution between X, Y and Z :\n", joint_xyz)
    # print("Joint probability distribution between W, Z and X :\n", joint_wzx)

    # Q12 : Verify exercises by hand
    print_in_a_frame("Question 12", '=')
    # Entropy
    print("Entropy of X : {:.3f}".format(entropy(p_x)))
    print("Entropy of Y : {:.3f}".format(entropy(p_y)))
    print("Entropy of Z : {:.3f}".format(entropy(p_w)))
    print("Entropy of W : {:.3f}\n".format(entropy(p_z)))
    # Joint entropy
    print("Joint entropy of X and Y : {:.3f}".format(joint_entropy(joint_xy)))
    print("Joint entropy of X and W : {:.3f}".format(joint_entropy(joint_xw)))
    print("Joint entropy of Y and W : {:.3f}".format(joint_entropy(joint_yw)))
    print("Joint entropy of W and Z : {:.3f}\n".format(joint_entropy(joint_wz)))
    # Conditional entropy
    print("Conditional entropy of X knowing Y : {:.3f}".format(conditional_entropy(joint_xy, p_y)))
    print("Conditional entropy of W knowing X : {:.3f}".format(conditional_entropy(np.transpose(joint_xw), p_x)))
    print("Conditional entropy of Z knowing W : {:.3f}".format(conditional_entropy(joint_wz, p_w)))
    print("Conditional entropy of W knowing Z : {:.3f}\n".format(conditional_entropy(np.transpose(joint_wz), p_z)))
    # Mutual information
    print("Mutual information between X and Y : {:.3f}".format(mutual_information(joint_xy, p_x, p_y)))
    print("Mutual information between X and W : {:.3f}".format(mutual_information(joint_xw, p_x, p_w)))
    print("Mutual information between Y and Z : {:.3f}".format(mutual_information(joint_yz, p_y, p_z)))
    print("Mutual information between W and Z : {:.3f}\n".format(mutual_information(joint_wz, p_w, p_z)))
    # Conditional joint entropy
    print("Conditional joint entropy of X and Y knowing W : {:.3f}".format(cond_joint_entropy(joint_xyw, p_w)))
    print("Conditional joint entropy of W and Z knowing X : {:.3f}\n".format(cond_joint_entropy(joint_wzx, p_x)))
    # Conditional mutual information
    print("Conditional mutual information of X and Y knowing W : {:.3f}".format(cond_mutual_information(joint_xyw, p_w, joint_xw, joint_yw)))
    print("Conditional mutual information of W and Z knowing X : {:.3f}".format(cond_mutual_information(joint_wzx, p_x, np.transpose(joint_xw), np.transpose(joint_xz))))

    # Q13 : Entropy of a single square
    print_in_a_frame("Question 13", '=')
    prob = np.full(9, 1/9)
    print("Entropy of a single square : {:.3f}".format(entropy(prob)))

    # Q14 : Entropy of the subgrid
    print_in_a_frame("Question 14", '=')
    entropy_subgrid = 0
    i = 6
    while i >= 1:
        prob = np.full(i, 1/i)
        entropy_subgrid += entropy(prob)
        i -= 1
    print("Entropy of the given subgrid : {:.3f}".format(entropy_subgrid))

    # Q15 : Entropy of unsolved sudoku grid
    print_in_a_frame("Question 15", '=')
    sudoku = np.load('sudoku.npy')
    print("Sudoku \n", sudoku)
    entropy_sudoku = entropy_unsolved_sudoku(sudoku)
    print("Entropy of the given unsolved sudoku : {:.3f}\n".format(entropy_sudoku))
    
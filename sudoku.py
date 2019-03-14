import numpy as np
import math

def union(lst1, lst2, lst3): 
    # Compute the union of  lists without repetition
    # without zero
    final_list = list(set(lst1) | set(lst2) |set(lst3))
    final_list = [x for x in final_list if x!=0]
    try:
        final_list.remove(c)
    except ValueError:
        pass

    return final_list 





if __name__=='__main__':
    sudoku = np.load('sudoku.npy')
    print("Sudoku \n",sudoku)
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
    print(entropy)


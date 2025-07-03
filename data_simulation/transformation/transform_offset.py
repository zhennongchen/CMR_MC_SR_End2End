import numpy as np

def transform_full_matrix_offset_center(matrix, shape):
    
    # Check dimensions
    if matrix.ndim != 2:
        raise ValueError("The transformation matrix must have 2 dimensions.")

    if matrix.shape[0] != (len(shape) + 1):
        raise ValueError(
            "The first dimension of the transformation matrix must be equal to len(shape) + 1."
        )

    if matrix.shape[1] != (len(shape) + 1):
        raise ValueError(
            "The second dimension of the transformation matrix must be equal to len(shape) + 1."
        )

    shape = np.array([float(s) / 2.0 + 0.5 for s in shape])
    # print('shape is: ',shape)
   
    for_mat = np.eye(len(shape) + 1)
    # print('for_mat: ',for_mat)
    rev_mat = np.eye(len(shape) + 1)
    # print('rev_mat: ',rev_mat)
   

    for_mat[:-1, -1] = +shape
    # print('for_mat: ',for_mat)

    rev_mat[:-1, -1] = -shape
    # print('rev_mat: ',rev_mat)


    return np.dot(np.dot(for_mat, matrix), rev_mat)


def transform_full_matrix_offset_heart(matrix, heart_center):
    
    # Check dimensions
    if matrix.ndim != 2:
        raise ValueError("The transformation matrix must have 2 dimensions.")

    if matrix.shape[0] != (len(heart_center) + 1):
        raise ValueError(
            "The first dimension of the transformation matrix must be equal to len(shape) + 1."
        )

    if matrix.shape[1] != (len(heart_center) + 1):
        raise ValueError(
            "The second dimension of the transformation matrix must be equal to len(shape) + 1."
        )

    shape = np.asarray(heart_center)
   
   
    for_mat = np.eye(len(shape) + 1)
  
    rev_mat = np.eye(len(shape) + 1)
   
   

    for_mat[:-1, -1] = +shape

    rev_mat[:-1, -1] = -shape


    return np.dot(np.dot(for_mat, matrix), rev_mat)
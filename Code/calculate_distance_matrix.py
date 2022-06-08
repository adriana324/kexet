import tmd
import numpy as np

"""
STEP 3:
GENERATE THE TMD DISTANCE MATRIX OF THE DATASET
"""

#Original file with the persistence images (one for each row)
X = np.loadtxt("X_train_all_types.txt")

#Create a N x N matrix filled with zeros, N = #presistance images
distance_matrix = np.zeros((len(X),len(X)))

for i in range(len(X)): 
    for j in range(i,len(X)):
        p_img1_flatten = X[i]
        p_img2_flatten = X[j]
        
        p_img1 = p_img1_flatten.reshape((100,100)) #Reshape the flatten presistance image to the original 100 x 100 
        p_img2 = p_img2_flatten.reshape((100,100)) #Reshape the flatten presistance image to the original 100 x 100 

        distance = np.sum(np.abs(tmd.analysis.get_image_diff_data(p_img1, p_img2)))
        distance_matrix[i][j] = distance
        distance_matrix[j][i] = distance 



np.savetxt("distance_matrix.txt", distance_matrix,  fmt='%s')

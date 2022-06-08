#Some parts of the code is taken from:
#https://github.com/AddedK/kexjobb/blob/master/src/createDataset.py 

"""
STEP 2:
CREATE ONE DATASET (TXT-FILE) FROM THE SEPARATE DATASETS OF PERSISTENCE IMAGES. 
CREATE TWO LABELS (TXT-FILE) FOR THE DATASET. ONE MORE SPECIFIC CONTAINING BOTH 
SPECIES AND CELL TYPE AND ANOTHER MORE GENERAL CONTAINING ONLY THE CELL TYPE.
"""

#from cProfile import label
import numpy as np
import fileinput
from os import walk

X_train_list = []
nr_of_labels = []
label_names = []

#Extract the file name of eaach file in given directory. 
#Then create a string that will be used to create the labels of 
#the data. Containg the name of the species + cell type
def extract_file_name():
    #Returns the filename in given directory and extend X_train_list with the filenames
    for(_,_,filenames) in walk ("working_dic/"):
        X_train_list.extend(filenames)
        break

    #Load each file and get the file name. 
    #Create a string, containg both the name of the species and celltype. 
    #Add it to a list. 
    for file_name in X_train_list:
        if file_name.endswith(".txt"):
            file = np.loadtxt("working_dic/" + file_name)
            nr_of_labels.append(len(file))
            label_names.append(file_name.split(".")[0]) #Exlude .txt 


#Create two label files. One more specific, containg species, cell type and color.
#Another more general containing only the cell type and color. 
def create_labels():
    colors = ["red", "blue", "green", "orange", "purple", "yellow"]
    labels_specific = []
    labels_general = []

    labels_specific.append("label color")
    labels_general.append("label color")
    j = 0
    for label_amount, label_name in zip(nr_of_labels, label_names):
        for i in range(label_amount):
            color = colors[j]
            labels_specific.append(label_name + " " +color) #Append both species and cell type + color
            celltype = label_name.split("_")[1] # Extract the cell type 

            if celltype == "pyramidal": 
                general_color = 'red'
            elif celltype == "granule":
                general_color = "blue"

            labels_general.append(celltype + " " + general_color) #Append only the cell type + color
        
        j+=1
    np.savetxt("specific_labels.txt", labels_specific,  fmt='%s') 
    np.savetxt("general_labels.txt", labels_general,  fmt='%s')


#Store all the data in one big file. 
def create_dataset():
    new_train_list = []
    for train_list in X_train_list:
        if train_list.endswith(".txt"):
            new_train_list.append("working_dic/" + train_list)


    with open('X_train_all_types.txt', 'w') as file:
        input_lines = fileinput.input(new_train_list)
        file.writelines(input_lines)

extract_file_name()
create_labels()
create_dataset()





import pandas as pd 
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture 
from matplotlib.lines import Line2D
import matplotlib.cm as cm

# Dimension reduction and clustering libraries
import umap
import hdbscan
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


"""
STEP 4:
DIMENSIONALITY REDUCTION WITH PCA AND UMAP ON THE DATA.
HDBSCAN CLUSTERING ON THE DIM. REDUCED DATA. 
VISUALIZATION OF THE RESULTING CLUSTERS, BOTH SPECIFIC (SPECIES + CELL TYPE)
AND MORE GENERAL (ONLY CELL TYPE) 
"""

#Load the distance_matrix
X = np.loadtxt("distance_matrix.txt")

#Convert matrix and vector to Pandas DataFrame 
feat_cols = ['pixel' + str(i) for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feat_cols)
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))

#Extract top 40 principal components
pca_40 = PCA(n_components=40)
pca_result_40 = pca_40.fit_transform(df[feat_cols].values)
print('Cumulative explained variation for 40 principal components: {}'.format(
    np.sum(pca_40.explained_variance_ratio_)))


#Configure UMAP hyperparameters and transform the data. 
umap_embedding = umap.UMAP(
    n_neighbors=50,
    min_dist=0,     #min_dist 0, good for density based clustering methods
    n_components=2, #Reduce to 2 dimensions
    random_state=42,
).fit_transform(pca_result_40)

#Extract the colors from the label files (both specific and more general - used for dataset 2) 
specific_labels = pd.read_csv("specific_labels.txt", delim_whitespace=True)
colors_specific = specific_labels.loc[:,"color"]

general_labels = pd.read_csv("general_labels.txt", delim_whitespace=True)
colors_general = general_labels.loc[:,"color"]


#Function used to visulize the results of the dimensionality reduction for dataset 1 
def plotDataSet1():
      plt.figure(1)
      plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c= colors_specific, s=1.5, cmap='Spectral')

      #Create a list of legend elemntes for dataset 1 - figure 1
      legend_elements = [Line2D([0], [0], marker='o', color='w', label='human pyramidal',
                              markerfacecolor='red', markersize=5), 
                        Line2D([0], [0], marker='o', color='w', label='rat pyramidal',
                              markerfacecolor='blue', markersize=5),
                        Line2D([0], [0], marker='o', color='w', label='mouse pyramidal',
                              markerfacecolor='green', markersize=5)]
      #Plot legend
      plt.legend(handles=legend_elements, loc='upper right')
     
      #Title of the figure 
      plt.suptitle('Dataset 1: Pyramidal cells layer 2-3 from humans, mice and rats', fontsize=10)
      
      #Change the info displayed on the x and y axis
      hideAxisInfo()

      


#Visulize the results of the dimensionality reduction for dataset 1.
#Both with the more specific labeling and the more general one. 
def plotDataSet2(): 
      #Visulize the results of the dimensionality reduction with specfic labeling 
      plot1 = plt.figure(1)
      plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c= colors_specific, s=1.5, cmap='Spectral')

      #Create a list of legend elemntes for dataset 2 - figure 1
      legend_elements = [Line2D([0], [0], marker='o', color='w', label='rat pyramidal', 
                          markerfacecolor='red', markersize=5), 
                        Line2D([0], [0], marker='o', color='w', label='rat granule',
                          markerfacecolor='blue', markersize=5),
                        Line2D([0], [0], marker='o', color='w', label='mouse pyramidal',
                          markerfacecolor='green', markersize=5),
                        Line2D([0], [0], marker='o', color='w', label='mouse granule',
                          markerfacecolor='orange', markersize=5)]

      #Plot legend
      plt.legend(handles=legend_elements, loc='lower right')
      
      #Title of the first figure 
      plot1.suptitle('Dataset 2: Granule and pyramidal cells from mice and rats', fontsize=10)

      #Change the info displayed on the x and y axis
      hideAxisInfo()



      ##Visulize the results of the dimensionality reduction with more general labeling
      plot3 = plt.figure(3)
      plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c= colors_general, s=1.5, cmap='Spectral')


      # create a list of legend elements for dataset 2 (more general) 
      legend_elements = [Line2D([0], [0], marker='o', color='w', label='pyramidal',
                          markerfacecolor='red', markersize=5), 
                    Line2D([0], [0], marker='o', color='w', label='granule',
                          markerfacecolor='blue', markersize=5)]

      #plot legend
      plt.legend(handles=legend_elements, loc='lower right')

      #Title for third figure in dataset 2 (more general) 
      plot3.suptitle('Dataset 2: Granule and pyramidal cells from mice and rats', fontsize=10)

      hideAxisInfo()




#Cluster the data and try to find groups when looking both at the celltype and species. 
def plotClusterSpecific(figure_title):
      #HBSSCAN clustering
      # min_cluster_size : sets the smallest size grouping that you wish to consider a cluster
      # min_samples: the minimum number of neighbours to a core point. The higher this is, 
      # the more points are going to be discarded as noise/outliers. 
      hdbscan_labels = hdbscan.HDBSCAN(
            min_samples=10,
            min_cluster_size=30,
      ).fit_predict(umap_embedding)

      plot2 = plt.figure(2)

      specmap = cm.get_cmap('Spectral', max(hdbscan_labels)+1)
      colors2 = specmap(range(0, max(hdbscan_labels)+1))
      legend_elements2 = []
      
      #Legend for the clusters 
      for x in range(0, max(hdbscan_labels) + 1):
            legend_elements2.append( Line2D([0], [0], marker='o', color='w', label='Cluster ' + str(x+1), markerfacecolor=colors2[x], markersize=5))
      
      #Title of the figure 
      plot2.suptitle(figure_title, fontsize=10)

      #Plot legend 
      plt.legend(handles=legend_elements2, loc='upper right')


      #Plot the clustering  
      clustered = (hdbscan_labels >= 0)
      print(hdbscan_labels[clustered])
      plt.scatter(umap_embedding[~clustered, 0],
                  umap_embedding[~clustered, 1],
                  color=(0.5, 0.5, 0.5),
                  s=1.5,
                  alpha=0.5)
      plt.scatter(umap_embedding[clustered, 0],
                  umap_embedding[clustered, 1],
                  c=hdbscan_labels[clustered],
                  s=1.5,
                  cmap='Spectral', 
			label=hdbscan_labels[clustered]
			)
      hideAxisInfo()

      #Get adjusted rand score and mutual infromation score
      clustered = (hdbscan_labels >= 0)
      print("RAND SCORE:")
      print(adjusted_rand_score(colors_specific[clustered], hdbscan_labels[clustered]))
      print("MUTUAL INFORMATION SCORE:")
      print(adjusted_mutual_info_score(colors_specific[clustered], hdbscan_labels[clustered]))


#Plot a more general clustering, that is only looking at the cell type. 
def plotClusterGeneral(): 
      #A more general clustering with HBSSCAN
      hdbscan_labels2 = hdbscan.HDBSCAN(
      min_samples=30,
      min_cluster_size=30,
      ).fit_predict(umap_embedding)

      plot4 = plt.figure(4)

      #Legends for the general clustering 
      specmap2 = cm.get_cmap('Spectral', 2)
      colors4 = specmap2(range(0, max(hdbscan_labels2)+1))
      legend_elements4 = []
      for x in range(0, max(hdbscan_labels2) + 1):
            legend_elements4.append( Line2D([0], [0], marker='o', color='w', label='Cluster ' + str(x+1), 
            markerfacecolor=colors4[x], markersize=5) )

      plt.legend(handles=legend_elements4, loc='lower right')

      #Title for second cluster of dataset 2
      plot4.suptitle('Dataset 2: More general clusters detected with HDBSCAN', fontsize=10)


      second_clustering = (hdbscan_labels2>= 0)
      plt.scatter(umap_embedding[~second_clustering, 0],
                  umap_embedding[~second_clustering, 1],
                  color=(0.5, 0.5, 0.5),
                  s=1.5,
                  alpha=0.5)
      plt.scatter(umap_embedding[second_clustering, 0],
                  umap_embedding[second_clustering, 1],
                  c=hdbscan_labels2[second_clustering],
                  s=1.5,
                  cmap='Spectral')
      
      hideAxisInfo()
      
      #Print rand score and mutual info score 
      clustered = (hdbscan_labels2 >= 0)
      print("RAND SCORE:")
      print(adjusted_rand_score(colors_general[clustered], hdbscan_labels2[clustered]))
      print("MUTUAL INFORMATION SCORE:")
      print(adjusted_mutual_info_score(colors_general[clustered], hdbscan_labels2[clustered]))


#Function that hides the numbers for the x and y-axis and add axis names
def hideAxisInfo():
      ax = plt.gca()
      ax.axes.xaxis.set_ticklabels([])
      ax.axes.yaxis.set_ticklabels([])

      plt.xlabel('UMAP 1')
      plt.ylabel('UMAP 2')


#Plot both the clustering and dim. reduction of dataset 1 
def getResultsDataset1():
      #Plot the clustering results
      fig_title = 'Dataset 1: Clusters detected with HDBSCAN'
      plotDataSet1()
      plotClusterSpecific(fig_title)


#Plot both the clustering and dim. reduction of dataset 2
def getResultsDataset2():
      fig_title = 'Dataset 2: Clusters detected with HDBSCAN'
      plotDataSet2()
      plotClusterSpecific(fig_title)
      plotClusterGeneral()



#Plot the results
getResultsDataset1()
#getResultsDataset2()
plt.show()


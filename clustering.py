from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as sklearnPCA
import json
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import scipy as sp
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage


fig = plt.figure()

x=[]
y=[]
z=[]
orgc=[]
clusternum=0;

def create_group(x0,y0,z0,xr0,yr0,zr0,samples):
    global x,y,z,orgc,clusternum
    x+= np.random.normal(loc=x0, scale=xr0, size=samples).tolist()
    y+=np.random.normal(loc=y0, scale=yr0, size=samples).tolist()
    z+=np.random.normal(loc=z0, scale=zr0, size=samples).tolist()
    cluster=np.zeros(samples);
    cluster.fill(clusternum)
    orgc+=cluster.tolist()
    clusternum=clusternum+1;

radius=10

create_group(50,0,0,radius,radius,radius,50)
create_group(0,50,0,radius,radius,radius,100)
create_group(0,0,0,radius,radius,radius,150)
create_group(200,0,0,5,5,5,200)
create_group(200,150,0,5,5,5,200)


rawdata={}
rawdata['x']=x;
rawdata['y']=y;
rawdata['z']=z;


print "*"*10
data = pd.DataFrame(rawdata)
print "*"*10
print data.describe()
print "*"*10

###################################
#ORIGINAL DATA
###################################

print "#ORG"*10
cx = fig.add_subplot(221, projection='3d')
cx.scatter(x, y, z, c=[orgc], marker='o')

cx.set_xlabel('X Label')
cx.set_ylabel('Y Label')
cx.set_zlabel('Z Label')
cx.set_title("Original")

###################################
# PCA
###################################

print "#PCA"*10

sklearn_pca = sklearnPCA(n_components=3)
#print sklearn_pca
sklearn_transf = sklearn_pca.fit_transform(data)
#print sklearn_transf

threed=False
if threed:
    dx = fig.add_subplot(223, projection='3d')
    dx.scatter(sklearn_transf[:,0],sklearn_transf[:,1],sklearn_transf[:,2], c=[orgc], marker='o')
    dx.set_xlabel('X Label')
    dx.set_ylabel('Y Label')
    dx.set_zlabel('Z Label')
    dx.set_title("PCA")
else:
    dx = fig.add_subplot(223)
    dx.scatter(sklearn_transf[:,0],sklearn_transf[:,1], c=[orgc], marker='o')
    dx.set_xlabel('X Label')
    dx.set_ylabel('Y Label')
    dx.set_title("PCA")



#plt.plot(sklearn_transf[:,0],sklearn_transf[:,1], 'o', markersize=7, color=colorpalette[orgc], alpha=0.5, label='class1')

###################################
#K MEANS
###################################

print "#KME"*10

model = KMeans(n_clusters=2)
model.fit(data)

ax = fig.add_subplot(222, projection='3d')
ax.scatter(x, y, z, c=[model.labels_], marker='x')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title("K Means")

#plt.show()

###################################

print "#LIN"*10

linkage_matrix = linkage(data)

print linkage_matrix

clus=sp.cluster.hierarchy.fcluster(linkage_matrix, 1.153)

bx = fig.add_subplot(224, projection='3d')

bx.scatter(x, y, z, c=[clus], marker='x')

bx.set_xlabel('X Label')
bx.set_ylabel('Y Label')
bx.set_zlabel('Z Label')
bx.set_title("Linkage")

plt.show()

#dendrogram(data)

#den=dendrogram(linkage_matrix,
#           color_threshold=1,
#           show_leaf_counts=True,
#           leaf_rotation=90.,  # rotates the x axis labels
#           leaf_font_size=8.,  # font size for the x axis labels
#           )

#plt.figure(figsize=(25, 10))
#plt.title('Hierarchical Clustering Dendrogram')
#plt.xlabel('sample index')
#plt.ylabel('distance')
#plt.show()

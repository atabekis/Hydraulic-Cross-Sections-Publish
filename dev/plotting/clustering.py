import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from matplotlib.patches import Patch


X = np.load('../../data/pub_input.npy')

image = X[10]
labeled_image, num_features = label(image)
cluster_image = np.where(np.isin(labeled_image, np.unique(labeled_image)[1:][np.bincount(labeled_image.flat)[1:] > 1]), 1, 0)
cluster_image[(image == 1) & (cluster_image == 0)] = 2

plt.figure(figsize=(6, 6))
plt.imshow(cluster_image, cmap=plt.cm.colors.ListedColormap(['black', 'red', 'white']))
plt.axis('off')

legend_elements = [Patch(facecolor='red', edgecolor='r', label='Clusters > 1 pixel'),
                   Patch(facecolor='white', edgecolor='k', label='Single white pixels')]

plt.legend(handles=legend_elements, loc='upper right')
plt.savefig('../../fig/clusters.png', bbox_inches='tight', pad_inches=0)
plt.show()


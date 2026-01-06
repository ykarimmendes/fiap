### https://on.fiap.com.br/mod/conteudoshtml/view.php?id=536635&c=14529&sesskey=NZou2qTH3z
from PIL import Image
import glob
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.cluster import KMeans

img_G = mpimg.imread("cursofiap/kmeans_mamografia/mdb001.pgm")
img_D = mpimg.imread("cursofiap/kmeans_mamografia/mdb003.pgm")
img_F = mpimg.imread("cursofiap/kmeans_mamografia/mdb005.pgm")

###plotando imagens
#fig, axs = plt.subplots(1, 3, figsize=(20, 6))
#im1=axs[0].imshow(img_G, cmap="gray", vmin=0, vmax=255)
#im1=axs[1].imshow(img_D, cmap="gray", vmin=0, vmax=255)
#im1=axs[2].imshow(img_F, cmap="gray", vmin=0, vmax=255)
#plt.show()

##usa kmeans como filtro de segmentação de imagem
def filtro_kmeans(img, clusters):
    vectorized = img.reshape((-1,1))
    kmeans = KMeans(n_clusters=clusters, random_state=0,n_init=5)
    kmeans.fit(vectorized)

    centers = np.uint8(kmeans.cluster_centers_)
    segmented_data = centers[kmeans.labels_.flatten()]
    segmented_image = segmented_data.reshape((img.shape))
    return (segmented_image)

cluster = 3
img_G_segmentada = filtro_kmeans(img_G, cluster)
img_D_segmentada = filtro_kmeans(img_D, cluster)
img_F_segmentada = filtro_kmeans(img_F, cluster)

#plotando imagens
fig, axs = plt.subplots(1, 3, figsize=(20, 6))
im1=axs[0].imshow(img_G_segmentada, cmap="gray", vmin=0, vmax=255)
im1=axs[1].imshow(img_D_segmentada, cmap="gray", vmin=0, vmax=255)
im1=axs[2].imshow(img_F_segmentada, cmap="gray", vmin=0, vmax=255)
plt.show()



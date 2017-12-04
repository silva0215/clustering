import sys
import numpy as np
import os
import cv2
from sklearn.cluster import KMeans

args = sys.argv

imglist = []

target_dir = args[1]

for root, dirs, files in os.walk(target_dir):
	for file in files:
		if not file.startswith(".") and file.endswith(".jpg"):
			imglist.append(file)

akaze = cv2.AKAZE_create()

deslist = np.empty((0,61),int)
lenlist = []

for file in imglist:
	path = os.path.join(target_dir,file)
	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	kp, des = akaze.detectAndCompute(img, None)

	lenlist.append(des.shape[0])

	for d in des:
		deslist = np.append(deslist, d.reshape(1,61), axis=0)

km = KMeans(n_clusters = 500, random_state = 10)
kmeans_model = km.fit(deslist)
km_bovw = kmeans_model.labels_

bovwlist = np.empty((0,500),int)
rows = 0

for l in lenlist:
	dis_converted = km_bovw[rows : rows + l]
	rows = rows + l

	bovw = np.zeros((1,500))

	for i in dis_converted:
		bovw[0][i] += 1

	bovw /= bovw.size
	
	for b in bovw:
		bovwlist = np.append(bovwlist, b.reshape(1,500), axis=0)


km2 = KMeans(n_clusters = 2, random_state = 10)
kmeans_model2 = km2.fit(bovwlist)
km_bovw = kmeans_model2.labels_

for b,i in zip(km_bovw,imglist):
	print(str(i) + "," + str(b))
# SpectralClustering

from sklearn.cluster import SpectralClustering
from sklearn import metrics
import numpy as np
from sklearn import datasets

# 根據參數造一個3維數據組 X.shape = (2500, 3), y則是(2500, 1)代表label
X, y = datasets.make_blobs(n_samples = 2500, n_features = 3, centers = 5, cluster_std = [1, 0.8, 1.1, 2, 0.3], random_state = 17)

# 將X丟到SpectralClustering中做預測，括號可選部分參數，分群數:n_clusters, 迦瑪值:gamma = 0.1
y_pred = SpectralClustering(n_clusters = 5).fit_predict(X)

# 計算分類分數用的
print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred) )

# 將降維後的前兩個維度畫出來，其中c代表每一個點的顏色，在這裡y_pred是每個點的類別，因此同一類會有同一個顏色，alpha代表透明度
plt.scatter(X.T[0],X.T[1], c = y_pred, alpha=.8)


***====================我是分隔線======================***

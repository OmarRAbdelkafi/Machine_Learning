#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def K_means_algo():
    #Generation de données
    X, y = make_blobs(n_samples = 100, centers = 3, cluster_std=0.5, random_state=0)

    #Entrainer le modele de K-Means clustering
    model = KMeans(n_clusters = 3, n_init = 10, max_iter = 300)
    model.fit(X)

    print(model.inertia_) #inertia positif (sommes des distances entre kes centroides et leurs donnée du cluster, plus c'est petit mieu c), cette distance est exprimé avec l'unité des data (euros, cm etc)
    print(model.cluster_centers_) #position des centroids
    print(model.labels_) #equivalent de predict

    predictions = model.predict(X)
    plt.scatter(X[:,0], X[:,1], c=predictions)
    plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1], c='r')
    plt.show()

    #Pour déteter le bon nombre de cluster il faut détercter la zone de coude, la ou l'inertia devient faible mais sans créer un cluster par point car ça n'a plus de sens sinon
    inertia = []
    k_range = range(1,20)
    for k in k_range:
        model = KMeans(n_clusters = k)
        model.fit(X)
        inertia.append(model.inertia_)

    plt.plot(k_range, inertia)
    plt.xlabel('nombre de cluster')
    plt.ylabel('cout du modele')
    plt.show()

def Isolation_Forest_algo():
    '''
    Cet algorithme permet de détecter les anomalies dans un data set avec le principe de tracer des ligne aléatoire afin d'isoler l'echantillons corrompu
    '''
    X, y = make_blobs(n_samples = 100, centers = 1, cluster_std=0.1, random_state=0)
    X[-1, :] = np.array([2.25,5])
    plt.scatter(X[:,0], X[:,1])
    plt.show()

    from sklearn.ensemble import IsolationForest

    model = IsolationForest(contamination = 0.01) #1% de déchet dans le data set à identifier
    model.fit(X)
    prediction = model.predict(X)

    plt.scatter(X[:,0], X[:,1], c=prediction)
    plt.show()

def PCA_algo():
    '''
    Cette algorithme permet de réduire les dimension on compression n feature on x features plus petite, chaque x est une combinaison linaire des n features
    cela permet de réduire les dimension ce qui nous permet de les visualiser par exemple ou bien d'entrainer un modéle avec moin de variables
    '''
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA

    digits = load_digits()
    images = digits.images
    X = digits.data
    y = digits.target

    print(X.shape)

    '''
    REDUCTION À deux features
    '''

    model = PCA(n_components=2)
    X_reduced = model.fit_transform(X) #reduced est un tableau de 2 composante et chaque composante contient 64 valeurs, chaque composante est donc la combinaison linaire de 64 feature compressé

    plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y)
    plt.colorbar()
    plt.show()

    '''
    Afin de savoir quel est le bon nombre de composante, on peut réduire à 64 composantes
    puis on détecte quand quel est le nombre de composante qui nous permet de garder entre 95 et 99% de la variance
    '''

    model = PCA(n_components=X.shape[1])
    X_reduced = model.fit_transform(X)

    V = model.explained_variance_ratio_ #les % de variances pour chaque compostante
    print(np.cumsum(V)) #faire la somme jusqu'à 100%, on observe qu'apartir de la 40eme dimension on arrive à 99%
    print('nombre de dimension à garder avec un seuil 99% :', np.argmax(np.cumsum(V) > 0.99)  )

    #plt.plot(np.cumsum(V))
    #plt.show()

    #--> CC on peut réduire à 40 dimension en garden 99% de la variance de nos donnée

    model = PCA(n_components=40)
    X_reduced = model.fit_transform(X)

    V = model.explained_variance_ratio_ #les % de variances pour chaque compostante
    print(np.cumsum(V))


def main():
    K_means_algo()
    #Isolation_Forest_algo()
    #PCA_algo()

if __name__ == '__main__':
    main()

#coding:utf-8

import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

def coef_determination(y,pred):
    u = ((y-pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - u/v

def model(X, theta):
    '''
    produit matriciel entre X et theta
    '''
    return X.dot(theta)

def cost_fonction(X, y,theta):
    '''
    fonction cout du modéle d'apprentissage
    '''
    m = len(y) #nombre d'exemple dans le datasets
    return 1/(2*m) * np.sum((model(X,theta) - y)**2)

def grad(X, y, theta):
    '''
    fonction qui permet de calculer le gradient pour la descente
    '''
    m = len(y) #nombre d'exemple dans le datasets
    Trans = X.T #transposé de la matrice X
    return 1/m * Trans.dot(model(X,theta) - y)

def gradient_descent(X, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations) #tracer l'apprentissage
    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(X, y, theta)
        cost_history[i] = cost_fonction(X, y, theta)

    return theta, cost_history


def RegressionLineaire():
    '''
    l'objectif est de trouver la fonction cout sous la forme linéaire y = ax+b, donc de trouver a et b à partir du datasets
    '''
    x, y = make_regression(n_samples=100, n_features=1, noise=10)
    #plt.scatter(x, y)
    #plt.show()

    #redimensionner y
    y = y.reshape(y.shape[0],1)

    #matrice X, ajouter le biais à x, avec
    # un vecteur de 1 ce vecteur est la pour représenter le coef de b
    X = np.hstack((x,np.ones(x.shape)))

    #initialiser theta aléatoirement
    theta = np.random.randn(2,1) #le vecteur qui définie les coeff de a et b

    '''
    #pour voir la droite aléatoire au début par rapport au nuage de pt x,y
    droite_alea = model(X, theta)
    plt.scatter(x, y)
    plt.plot(x, droite_alea)
    plt.show()
    '''
    #param descente
    learning_rate = 0.01
    n_iterations = 500

    theta_finale, cost_history = gradient_descent(X, y, theta, learning_rate, n_iterations)

    prediction = model(X, theta_finale)


    plt.scatter(x,y)
    plt.plot(x, prediction, c='r')
    plt.show()


    '''
    plt.plot(range(n_iterations), cost_history)
    plt.show()
    '''

    #coeff de determination R pour notre prediction
    print(coef_determination(y,prediction))


def RegressionPolynomiale():
    '''
    l'objectif est de trouver la fonction cout sous la forme plynomiale y = ax²+bx+c, donc de trouver a, b et c à partir du datasets
    '''
    x, y = make_regression(n_samples=100, n_features=1, noise=10)
    y = y + abs(y/2) #pour avoir un data set pas linéaire
    #plt.scatter(x, y)
    #plt.show()

    #redimensionner y
    y = y.reshape(y.shape[0],1)

    #matrice X, pour passer à un modéle plynomiale
    #il faudra ajouter au début en plus une colone pour x²
    X = np.hstack((x,np.ones(x.shape)))
    #voici le changement
    X = np.hstack((x**2, X))

    #initialiser theta aléatoirement
    #le changement qu'on fait est de faire passer theta de 2 param à 3 param (ou lignes ici)
    theta = np.random.randn(3,1) #le vecteur qui définie les coeff de a, b et c

    '''
    #pour voir la droite aléatoire au début par rapport au nuage de pt x,y
    droite_alea = model(X, theta)
    plt.scatter(x, y)
    plt.plot(x, droite_alea)
    plt.show()
    '''

    #param descente
    learning_rate = 0.01
    n_iterations = 500

    theta_finale, cost_history = gradient_descent(X, y, theta, learning_rate, n_iterations)

    #création d'une courbe de prediction qui prédit notre model finale
    prediction = model(X, theta_finale)


    plt.scatter(x[:,0],y)
    plt.scatter(x[:,0],prediction, c = 'r')
    plt.show()



    plt.plot(range(n_iterations), cost_history)
    plt.show()


    #coeff de determination R pour notre prediction
    print(coef_determination(y,prediction))


def RegressionLineaireManyFeatures():
    '''
    l'objectif est de trouver la fonction cout sous la forme linéaire y = ax1+bx2+c,
    avec plusieur feature, ici 2 features par exemple
    '''
    #generer un nuage de point avec 2 features
    x, y = make_regression(n_samples=100, n_features=2, noise=10)
    plt.scatter(x[:,0], y)
    plt.show()

    #redimensionner y
    y = y.reshape(y.shape[0],1)

    #changement ici car il faut ajouter le biais (les 1) seulement à la fin
    X = np.hstack( ( x, np.ones((x.shape[0],1)) ) )

    #initialiser theta aléatoirement pour a, b et c
    theta = np.random.randn(3,1)

    #param descente
    learning_rate = 0.01
    n_iterations = 500

    theta_finale, cost_history = gradient_descent(X, y, theta, learning_rate, n_iterations)

    #création d'une courbe de prediction qui prédit notre model finale
    prediction = model(X, theta_finale)

    #coeff de determination R pour notre prediction
    print(coef_determination(y,prediction))


    plt.scatter(x[:,0],y)
    plt.scatter(x[:,0],prediction, c = 'r')
    plt.show()


    plt.plot(range(n_iterations), cost_history)
    plt.show()

    #3d visualusation
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x[:,0], x[:,1], y)
    ax.scatter(x[:,0], x[:,1], prediction, c = 'r')
    plt.show()

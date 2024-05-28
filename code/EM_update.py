import numpy as np
from typing import List, Tuple
import argparse
import matplotlib.pyplot as plt

""" EM Update Rules """
def generate_pi(seed_pi) -> float:
    np.random.seed(seed_pi)
    return np.random.uniform(low=0, high=1)

def generate_theta2d(seed_theta, dimension: int = 2, bound: float = 5) -> List:
    np.random.seed(seed_theta)
    return np.random.uniform(low=-bound, high=bound, size=dimension)

def generate_theta(d: int, seed_theta: int) -> List[float]:
    Mu = np.zeros(d)
    Sigma = np.eye(d)
    # generate theta^*
    np.random.seed(seed_theta)
    theta = np.random.multivariate_normal(Mu, Sigma)
    # theta /= np.linalg.norm(theta)
    return theta

def generate_data(N: int, sigma: float,
                  seed_data: int, seed_latent: int, seed_noise: int,
                  theta: List, pi: float) -> Tuple[List, List]:
    d = len(theta)
    Mu = np.zeros(d)
    Sigma = np.eye(d)
    # generate X
    np.random.seed(seed_data)
    X = np.random.multivariate_normal(Mu, Sigma, size=N)
    # generate e
    np.random.seed(seed_noise)
    e = np.random.normal(0, sigma, N)
    # generate z
    np.random.seed(seed_latent)
    z = np.sign(pi - np.random.rand(N))
    return X, z * (X @ theta) + e #np.einsum('ij,j->i', X, theta)

def EM_update(X: List, Y: List, 
              sigma: float,
              T: int,
              theta0: List[float],
              pi0: float) -> Tuple[List, List]:
    N, d = len(X), len(X[0])
    Mu = np.zeros(d)
    Sigma = np.eye(d)
    # inverse covariance of X
    H = np.linalg.inv(np.einsum('ij,ik->jk', X, X)/N)
    sq = sigma * sigma
    # update rule for theta, pi
    def update_theta(X, Y, theta, pi, sq, H) -> List[float]:
        v = np.arctanh(2*pi-1)
        vecs =  [(np.tanh((x.T@theta)*y/sq + v)*y)*x for x, y in zip(X, Y)]
        return H @ sum(vecs)/len(vecs)
    def update_pi(X, Y, theta, pi, sq) -> float:
        v = np.arctanh(2*pi-1)
        return 0.5 + 0.5 * np.mean([np.tanh((x.T@theta)*y/sq + v) for x, y in zip(X, Y)])
    # EM update
    list_theta, list_pi = [theta0], [pi0]
    for _ in range(T):
        theta, pi = list_theta[-1], list_pi[-1]
        list_theta.append(update_theta(X, Y, theta, pi, sq, H))
        list_pi.append(update_pi(X, Y, theta, pi, sq))
    return list_theta, list_pi
    
def justify_params(list_theta: List, list_pi: List, 
                   theta: List[float]) -> Tuple[List, List]:
    if theta.T@list_theta[-1] < 0:
        return [-t for t in list_theta], [1-p for p in list_pi]
    return list_theta, list_pi

def calc_error(list_theta: List, list_pi: List, 
               theta: List[float], pi: float) -> Tuple[List[float], List[float]]:
    return [np.linalg.norm(t-theta) for t in list_theta], \
            [np.abs(p-pi) for p in list_pi]

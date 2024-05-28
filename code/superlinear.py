from EM_update import generate_theta, generate_data, EM_update,\
    generate_theta2d, generate_pi,\
    justify_params, calc_error
""" import packages for visualization """
from typing import List, Tuple
import numpy as np
from numpy import sqrt, arccos, pi
from numpy import sin, cos, tan, arctan, log, mgrid, array
from numpy import linspace, meshgrid, vectorize
import matplotlib.pyplot as plt
## RGB color platte: https://www.rapidtables.com/web/color/RGB_Color.html

def angle(vec1: List[float], vec2: List[float]):
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    return vec1.T@vec2 / (norm1*norm2)

def get_phi(theta: List[float], theta_star: List[float]):
    return pi/2 - arccos(abs(angle(theta, theta_star)))

def map_phi(phi: float):
    return log(pi/2) + log(tan(phi)-pi/4)

def specify_theta(phi: float, theta_star: List[float], seed: int):
    d, e1 = len(theta_star), theta_star / np.linalg.norm(theta_star)
    theta0 = generate_theta(d, seed)
    theta0 -= (theta0.T @ e1)*e1
    theta0 /= np.linalg.norm(theta0)
    return sin(phi)*theta_star + cos(phi)*theta0

def superlinear_theta(list_theta: List, theta_star: List[float]):
    list_map = []
    for theta in list_theta:
        phi = get_phi(theta, theta_star)
        list_map.append(map_phi(phi))
    return array(list_map)
    

def run_experiment(SNR: float
                   ) -> Tuple[List, List]:
    # generating data samples
    num_sample = 5000
    dimension = 50
    theta_star, pi_star = generate_theta(dimension, seed_theta=100),\
                            generate_pi(seed_pi=100)
    theta_star /= np.linalg.norm(theta_star)
    norm_star = np.linalg.norm(theta_star)
    sigma = norm_star/SNR # almost no noise, SNR -> inf
    seed_data, seed_latent, seed_noise = 0, 0, 0
    X, Y = generate_data(num_sample, sigma,
                  seed_data, seed_latent, seed_noise,
                  theta_star, pi_star)
    # specify the initial phi for theta0
    phi0 = arctan(1.5)
    # EM trials with different initial theta0, pi0
    num_init = 50# 60
    T = 4 # number of iterations
    lists_map = []
    for i in range(num_init):
        seed, with_label = i + 345, (i == 0)
        # theta0 = generate_theta(dimension, seed)
        theta0 = specify_theta(phi0, theta_star, seed)
        pi0 = generate_pi(seed)
        # EM update
        list_theta, list_pi = EM_update(X, Y, sigma, T,
                                        theta0,pi0)
        lists_map.append(superlinear_theta(list_theta, theta_star))
    return sum(lists_map)/len(lists_map)
    
def show_theory():
    plt.plot(np.arange(9), 2*np.arange(9), color='#FF6666', 
             label=r'SNR$\to\infty$: theoretical lower bound')
    
def save_plot():
    plt.xlim([0, None])
    plt.ylim([0, None])
    plt.ylabel(r'$\log\{\frac{\pi}{2}(\tan \varphi^t - \frac{\pi}{4})\}$')
    plt.xlabel(r'$\log\{\frac{\pi}{2}(\tan \varphi^{t-1} - \frac{\pi}{4})\}$')
    # plt.title(r'Superlinear Convergence Rate of $\theta^t$')
    plt.grid(color='gray', linestyle='dashed')
    plt.legend(loc='upper left')
    plt.savefig('superlinear.png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    colors = ["#aab7e3", "#7990db", "#4064d9"]
    markers = ["v","^", "o"]
    for lgSNR, c, m in list(zip(range(6, 8+1),colors,markers)):
        SNR = pow(10., lgSNR)
        list_map = run_experiment(SNR)
        plt.plot(list_map[:-1],list_map[1:], "--", color=c, marker=m,
                 label=r"SNR=$10^{}$".format(lgSNR))
    show_theory()
    save_plot()
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
import warnings
warnings.filterwarnings('ignore')

def angle(vec1: List[float], vec2: List[float]):
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    return vec1.T@vec2 / (norm1*norm2)

def get_phi(theta: List[float], theta_star: List[float]):
    return pi/2 - arccos(abs(angle(theta, theta_star)))
    
def specify_theta(phi: float, theta_star: List[float], seed: int):
    d, e1 = len(theta_star), theta_star / np.linalg.norm(theta_star)
    theta0 = generate_theta(d, seed)
    theta0 -= (theta0.T @ e1)*e1
    theta0 /= np.linalg.norm(theta0)
    return sin(phi)*theta_star + cos(phi)*theta0

def run_experiment(pi_star: float
                   ) -> Tuple[List, List]:
    # generating data samples
    num_sample = 5000
    dimension = 50
    theta_star= generate_theta(dimension, seed_theta=100)
    theta_star /= np.linalg.norm(theta_star)
    norm_star, SNR = np.linalg.norm(theta_star), 1e8
    sigma = norm_star/SNR # almost no noise, SNR -> inf
    seed_data, seed_latent, seed_noise = 0, 0, 0
    X, Y = generate_data(num_sample, sigma,
                  seed_data, seed_latent, seed_noise,
                  theta_star, pi_star)
    # specify the initial phi for theta0
    phi0 = 0.3 # arctan(1.5)
    # EM trials with different initial theta0, pi0
    num_init = 50# 60
    T = 6 # number of iterations
    lists_phi, lists_pi = [], []
    f = lambda x: get_phi(x, theta_star)
    for i in range(num_init):
        seed, with_label = i + 345, (i == 0)
        theta0 = specify_theta(phi0, theta_star, seed)
        pi0 = generate_pi(seed)
        # EM update
        list_theta, list_pi = EM_update(X, Y, sigma, T,
                                        theta0,pi0)
        lists_phi.append(array(list(map(f, list_theta))))
        lists_pi.append(array(list_pi))
    return sum(lists_phi)/num_init, sum(lists_pi)/num_init
    
def show_theory():
    plt.plot(np.linspace(0.,1.,num=10+1), 
             np.linspace(0.,1.,num=10+1), color='#FF6666', 
             label=r'SNR$\to\infty$: theoretical error of $||\pi^t-\pi^*||_1$')
    
def save_plot():
    plt.xlim([pow(0.1, 4), 1])
    plt.ylim([pow(0.1, 4), 1])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$(1-\frac{2}{\pi}\varphi^{t-1})\cdot||\frac{1}{2}-\pi^*||_1$')
    plt.ylabel(r'$||\pi^t-\pi^*||_1$')
    # plt.title(r'Superlinear Convergence Rate of $\theta^t$')
    plt.grid(color='gray', linestyle='dashed')
    plt.legend(loc='upper left')
    plt.savefig('mixingweight.png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    colors = ["#aab7e3", "#7990db", "#4064d9"]
    markers = ["v","^", "o"]
    for pi_star, c, m in list(zip(np.linspace(0.6, 1., num=3),colors,markers)):
        list_phi, list_pi = run_experiment(pi_star)
        plt.plot(np.abs((1/2-pi_star)*(1-list_phi[:-1]*2/pi)),
                 np.abs(list_pi[1:]-pi_star), "--", color=c, marker=m,
                 label=r"$\pi^\star=[{},{}]$"
                 .format(round(pi_star, 1), round(1.-pi_star,1)))
    show_theory()
    save_plot()
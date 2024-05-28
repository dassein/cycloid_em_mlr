from EM_update import generate_theta, generate_data, EM_update,\
    generate_theta2d, generate_pi,\
    justify_params, calc_error
""" import packages for visualization """
import numpy as np
from numpy import sqrt, arccos, pi
from numpy import sin, cos, tan, arctan, log, mgrid, array
from typing import List, Tuple
import matplotlib.pyplot as plt

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

def run_experiment(pi_star: float
                   ) -> Tuple[List, List]:
    # generating data samples
    num_sample = 5000
    dimension = 50
    theta_star= generate_theta(dimension, seed_theta=100)
    theta_star /= np.linalg.norm(theta_star)
    norm_star, SNR = np.linalg.norm(theta_star), 1e6
    sigma = norm_star/SNR # almost no noise, SNR -> inf
    seed_data, seed_latent, seed_noise = 0, 0, 0
    X, Y = generate_data(num_sample, sigma,
                  seed_data, seed_latent, seed_noise,
                  theta_star, pi_star)
    # specify the initial phi for theta0
    phi0 = 0.3 # arctan(1.5)
    # EM trials with different initial theta0, pi0
    num_init = 50# 60
    T = 10 # number of iterations
    lists_dists_theta, lists_errs_pi = [], []
    f = lambda x: get_phi(x, theta_star)
    for i in range(num_init):
        seed, with_label = i + 345, (i == 0)
        theta0 = specify_theta(phi0, theta_star, seed)
        pi0 = generate_pi(seed)
        # EM update
        list_theta, list_pi = EM_update(X, Y, sigma, T,
                                        theta0,pi0)
        list_theta, list_pi = justify_params(list_theta, list_pi, theta_star)
        dists_theta, errs_pi = calc_error(list_theta, list_pi, theta_star, pi_star)
        lists_dists_theta.append(array(dists_theta))
        lists_errs_pi.append(array(errs_pi))
    return sum(lists_dists_theta)/num_init, sum(lists_errs_pi)/num_init

## draw the scatter plot for each iterations of theta^t if d >= 2
def draw_thetas(list_theta: List, theta: List[float]) -> None:
    x_values, y_values = [t[0] for t in list_theta],\
                        [t[1] for t in list_theta]
    # Create the scatter plot
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.scatter(theta[0], theta[1], c='red', marker='o', s=50)
    plt.scatter(x_values, y_values)
    plt.axis('equal')
    # Link adjecent points with lines
    for i in range(len(x_values)-1):
        plt.plot([x_values[i], x_values[i+1]], [y_values[i], y_values[i+1]], c='blue')
    # Set the axis labels and title

    plt.xlabel(r'1st component of $\theta\in \mathbb{R}^d$')
    plt.ylabel(r'2nd component of $\theta\in \mathbb{R}^d$')
    plt.title(r'Scatter plot with $\theta^t$ and $\theta^*$')
    plt.grid()
    # Show the plot
    plt.show()

"""visualization""" 
# 1. the distance between theta^* and theta^t
# 2. the absolute error between pi^* and pi^t
def save_plot(ax1, ax2) -> None:
    ax1.set_title(r"Iterations vs Errors $||\theta^t-\theta^*||_2$ and $||\pi^t-\pi^*||_1$")
    # ax1.set_xlim([0, T])
    ax1.set_ylim([pow(0.1, 8), 2.5])
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$||\theta^t-\theta^*||_2$')
    # ax1.set_title(r'distance between $\theta^t$ and $\theta^*$')
    ax1.legend(loc='lower left')
    ax1.grid()
    # ax2.set_xlim([0, T])
    ax2.set_ylim([pow(0.1, 4), 1])
    ax2.set_yscale('log')
    ax2.set_xlabel(r'iteration $t$')
    ax2.set_ylabel(r'$||\pi^t-\pi^*||_1$')
    # ax2.set_title(r'distance between $\pi^t$ and $\pi^*$')
    ax2.legend(loc='lower left')
    ax2.grid()
    plt.savefig("dists.png", bbox_inches='tight', dpi=600)
    plt.show()


if __name__ == "__main__":
    colors = ["#aab7e3", "#7990db", "#4064d9"]
    markers = ["v","^", "o"]
    fig = plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    for pi_star, c, m in list(zip(np.linspace(0.6, 0.995, num=3),colors,markers)):
        dists_theta, errs_pi = run_experiment(pi_star)
        ax1.plot(range(len(dists_theta)),
                 dists_theta, "--", color=c, marker=m,
                 label=r"$\pi^\star=[{},{}]$"
                 .format(round(pi_star, 1), round(1.-pi_star,1)))
        ax2.plot(range(len(errs_pi)),
                 errs_pi, "--", color=c, marker=m,
                 label=r"$\pi^\star=[{},{}]$"
                 .format(round(pi_star, 1), round(1.-pi_star,1)))
    save_plot(ax1, ax2)
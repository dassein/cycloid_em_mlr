""" 
e_1 determined by theta^*
e_2 determined by each different theta^0
project theta^t / |theta^*| on the plane span{theta^*, theta^t}
we can express the projected vector as x^t e_1 + y^t e_2
hence, we can draw the coordinates of (x^t, y^t)
our theoretical analysis shows:
    (x^t, y^t) converges to the cycloid curve of a radius 1/pi
"""
from EM_update import generate_theta, generate_data, EM_update,\
    generate_theta2d, generate_pi,\
    justify_params, calc_error
""" import packages for visualization """
from typing import List, Tuple
import numpy as np
from numpy import sqrt, arccos, pi
from numpy import linspace, meshgrid, vectorize
import matplotlib.pyplot as plt
## RGB color platte: https://www.rapidtables.com/web/color/RGB_Color.html

def draw_trajectory_theoretical() -> None:
    rangex = linspace(-1, 1, num=100, endpoint=True)
    rangey = linspace(0, 2/pi, num=100, endpoint=True)
    X, Y = meshgrid(rangex,rangey)
    func = lambda x,y: sqrt((y*pi/2)*(1-y*pi/2)) - (abs(x)*pi/2-arccos(sqrt(abs(y)*pi/2)))
    f = vectorize(func)
    cs = plt.contour(X, Y, f(X, Y), [0], colors=['#FF6666'])
    cs.collections[0].set_label(r'theoretical trajectory of $\theta^t$')
    plt.scatter(1, 0, facecolors='none', edgecolors='#FF0000', marker='*', s=60,zorder=2, \
    label=r'true value $\theta^*, -\theta^*$')
    plt.scatter(-1, 0, facecolors='none', edgecolors='#FF0000', marker='*', s=60,zorder=2)
    plt.axis('equal')
    plt.xlabel(r'horizontal component of $\theta^0$ ($\parallel \theta^*$)')
    plt.ylabel(r'vertical component of $\theta^0$ ($\perp \theta^*$)')

def draw_trajectory_empirical(list_theta: List, theta_star: List[float],
                              with_label: bool):
    norm_star = np.linalg.norm(theta_star)
    e1 = theta_star / norm_star
    e2 = list_theta[0] - (e1.T@list_theta[0])*e1
    e2 /= np.linalg.norm(e2)
    x_values, y_values = [], []
    for theta in list_theta:
        x = e1.T@theta / norm_star
        y = e2.T@theta / norm_star
        x_values.append(x)
        y_values.append(y)
    # Link adjecent points with lines
    label_init = r'initial value $\theta^0$' if with_label else None
    plt.scatter(x_values[0], y_values[0], facecolors='none', edgecolors='#3399FF', marker='o', s=30, label=label_init)
    for i in range(len(x_values)-1):
        label = r'empirical trajectory of $\theta^t$' if ((i == 0) and with_label) else None
        tr = plt.plot([x_values[i], x_values[i+1]], [y_values[i], y_values[i+1]], c='#99ccff',zorder=1, label=label)


def save_plot():
    plt.title(r'Trajectories of $\theta^t$ and $\theta^*$')
    plt.grid(color='gray', linestyle='dashed')
    plt.ylim([0, None])
    plt.legend(loc='upper right')
    plt.savefig('trajectory_d>2.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
# generating data samples
    num_sample = 5000
    dimension = 50
    theta_star, pi_star = generate_theta(dimension, seed_theta=100),\
                            generate_pi(seed_pi=100)
    SNR, norm_star = 1e8, np.linalg.norm(theta_star)
    sigma = norm_star/SNR # almost no noise, SNR -> inf
    seed_data, seed_latent, seed_noise = 0, 0, 0
    X, Y = generate_data(num_sample, sigma,
                  seed_data, seed_latent, seed_noise,
                  theta_star, pi_star)
    # EM trials with different initial points theta0, pi0
    num_init = 60
    T = 100
    for i in range(num_init):
        seed, with_label = i + 345, (i == 0)
        theta0 = generate_theta(dimension, seed)
        pi0 = generate_pi(seed)
        # EM update
        list_theta, list_pi = EM_update(X, Y, sigma, T,
                                        theta0,pi0)
        draw_trajectory_empirical(list_theta, theta_star, with_label)
    draw_trajectory_theoretical()
    save_plot()
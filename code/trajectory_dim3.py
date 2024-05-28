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
from numpy import sin, cos, mgrid, array
from numpy import linspace, meshgrid, vectorize
import matplotlib.pyplot as plt
## RGB color platte: https://www.rapidtables.com/web/color/RGB_Color.html

def draw_trajectory_theoretical() -> None:
    ax = plt.gca()
    ax.view_init(20, 145) 
    # ax.view_init(30, 135) 
    plot_cycloid(ax)
    plot_cube(ax)

def plot_cycloid(ax):
    r = 1/np.pi #radius
    u, v = mgrid[0:2*np.pi:80j, 0:2*np.pi:40j]
    x = 1 - r * (u- sin(u))
    y = r * (1- cos(u)) * cos(v)
    z = r * (1- cos(u)) * sin(v)
    cs = ax.plot_surface(x, y, z, color='#FF6666', alpha=0.3, linewidth=2, antialiased=False, rstride=1, cstride=1,
                         label=r'theoretical trajectory of $\theta^t$')
    cs._facecolors2d = cs._facecolor3d
    cs._edgecolors2d = cs._edgecolor3d
    ax.set_xlabel(r'horizontal component of $\theta^0$ ($\parallel \theta^*$)')
    ax.set_zlabel(r'vertical component of $\theta^0$ ($\perp \theta^*$)')

def plot_cube(ax):
    # draw theta^\star
    ax.scatter([1], [0], [0],facecolors='none', edgecolors='#FF0000', marker='*', s=60,zorder=2, \
       label=r'true value $\theta^*, -\theta^*$')
    ax.scatter([-1], [0], [0], facecolors='none', edgecolors='#FF0000', marker='*', s=60,zorder=2)
    # draw cube
    range_x, range_y, range_z = [-1, 1], [-1, 1], [-1, 1.5]
    ax.set_xlim(range_x)
    ax.set_ylim(range_y)
    ax.set_zlim(range_z)
    ax.set_ylabel(None)
    ax.set_box_aspect(aspect=(
        range_x[1]-range_x[0], 
        range_y[1]-range_y[0], 
        range_z[1]-range_z[0]))
    ax.set_xticks(np.arange(range_x[0], range_x[1]+0.1, 0.5))
    ax.set_yticks(np.arange(range_y[0], range_y[1]+0.1, 0.5))
    ax.set_zticks(np.arange(range_z[0], range_z[1]+0.1, 0.5))
    
def draw_trajectory_empirical(list_theta: List, theta_star: List[float],
                              with_label: bool):
    norm_star = np.linalg.norm(theta_star)
    e1 = theta_star / norm_star
    e2 = list_theta[0] - (e1.T@list_theta[0])*e1
    e2 /= np.linalg.norm(e2)
    x_values, y_values, z_values = [], [], []
    for theta in list_theta:
        x = e1.T@theta / norm_star
        z = e2.T@theta / norm_star
        y = np.linalg.norm(theta/norm_star - x*e1 - z*e2)
        x_values.append(x)
        y_values.append(y)
        z_values.append(z)
    ax = plt.gca()
    label_init = r'initial value $\theta^0$' if with_label else None
    ax.scatter([x_values[0]], [y_values[0]], [z_values[0]], facecolors='none', edgecolors='#3399FF', marker='o', s=30, label=label_init)
    label = r'empirical trajectory of $\theta^t$' if with_label else None
    tr = ax.plot(x_values, y_values, z_values, c='#99ccff',zorder=1, label=label) 

def save_plot():
    plt.title(r'Trajectories of $\theta^t$ and $\theta^*$')
    plt.grid(color='gray', linestyle='dashed')
    plt.legend(loc='upper right')
    plt.savefig('trajectory_d=3.png', bbox_inches='tight', dpi=1200)
    plt.show()


if __name__ == "__main__":
    # generating data samples
    num_sample = 5000
    dimension = 3
    theta_star, pi_star = generate_theta(dimension, seed_theta=100),\
                            generate_pi(seed_pi=100)
    theta_star /= np.linalg.norm(theta_star)
    SNR, norm_star = 1e8, np.linalg.norm(theta_star)
    sigma = norm_star/SNR # almost no noise, SNR -> inf
    seed_data, seed_latent, seed_noise = 0, 0, 0
    X, Y = generate_data(num_sample, sigma,
                  seed_data, seed_latent, seed_noise,
                  theta_star, pi_star)
    # EM trials with different initial theta0, pi0
    num_init = 10# 60
    T = 100
    plt.axes(projection='3d')
    for i in range(num_init):
        seed, with_label = i + 345, (i == 0)
        theta0 = generate_theta(dimension, seed)
        theta0 /= np.linalg.norm(theta0)
        pi0 = generate_pi(seed)
        # EM update
        list_theta, list_pi = EM_update(X, Y, sigma, T,
                                        theta0,pi0)
        draw_trajectory_empirical(list_theta, theta_star, with_label)
    draw_trajectory_theoretical()
    save_plot()
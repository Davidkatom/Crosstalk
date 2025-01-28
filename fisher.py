from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact

# Define the function
def vw(t1, t2, a, w):
    term1 = (np.exp(a * t1) + np.cos(t1 * w))**2
    term2 = (np.exp(a * t2) + np.cos(t2 * w))**2
    term3 = (t1**2 * np.cos(t1 * w)**2) / term1**2 + (t2**2 * np.cos(t2 * w)**2) / term2**2
    term4 = (1/np.sin((t1 - t2) * w))**2
    return np.sqrt(term1 * term2 * term3 * term4 / (t1**2 * t2**2))

# Create a grid of points
t1_values = np.linspace(0, 5, 100)
t2_values = np.linspace(0, 5, 100)
t1_grid, t2_grid = np.meshgrid(t1_values, t2_values)

# Function to create the plot
def plot_func(a, w):
    # Calculate the function values
    vw_values = vw(t1_grid, t2_grid, a, w)

    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(t1_grid, t2_grid, vw_values)

    # Set labels
    ax.set_xlabel('t1')
    ax.set_ylabel('t2')
    ax.set_zlabel('vw')

    # Show the plot
    plt.show()
    return fig
# Create interactive widgets
fig = interact(plot_func, a=(0.1, 2.0, 0.1), w=(1, 5, 0.1))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Number of points
n_points = 100

# Generate points from a normal distribution with a fixed standard deviation
sigma = 1
points = np.random.normal(0, sigma, n_points)  # Initial points with mu = 0

# Create figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Histogram for the distribution
count, bins, ignored = ax1.hist(points, 30, density=True, alpha=0.6, color='g')

# Range of mu values for the animation
mu_values = np.linspace(-5, 5, 1000)


# Function to update the plot
def update(frame):
    mu = mu_values[frame]

    # Update distribution plot
    ax1.clear()
    ax1.hist(points, 30, density=True, alpha=0.6, color='g')
    ax1.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    ax1.set_title('Normal Distribution')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Probability Density')

    # Update likelihood plot
    ax2.clear()
    likelihoods = np.array(
        [np.sum(-np.log(np.sqrt(2 * np.pi * sigma ** 2)) - ((points - m) ** 2 / (2 * sigma ** 2))) for m in mu_values])
    ax2.plot(mu_values, likelihoods, color='b')
    ax2.axvline(x=mu, color='r', linestyle='--')
    ax2.set_title('Log-Likelihood Function')
    ax2.set_xlabel('Mu')
    ax2.set_ylabel('Log-Likelihood')


# Creating the animation
ani = animation.FuncAnimation(fig, update, frames=len(mu_values), repeat=True)

# Save the animation to a file
output_file = 'normal_distribution_animation.mp4'
ani.save(output_file, writer='ffmpeg', fps=90)

output_file

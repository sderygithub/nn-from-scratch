import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np


def plot_decision_boundary(pred_func, X, y):
    """

    Usage:
        >>> model = nn.Sequential(
            nn.Linear.random(n_input, n_hidden),
            F.Tanh(),
            nn.Linear.random(n_hidden, n_output),
            F.Softmax(),)
        >> plot_decision_boundary(lambda x: model.predict(x), x, y)
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def render_training(model, X, y, num_iter=1000, filepath="training.mp4"):
    """

    Usage:
        >>> model = nn.Sequential(
            nn.Linear.random(n_input, n_hidden),
            F.Tanh(),
            nn.Linear.random(n_hidden, n_output),
            F.Softmax(),)
        >>> render_training(model, x, y, num_iter=100, filepath='animation.mp4')
    """
    fig, ax = plt.subplots()

    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    def update(i):
        """"""
        # Fetch data from generator
        train(model, X, y)

        # Predict the function value for the whole gid
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Update the title
        plt.title(r't = %1.2e' % i)

        # Plot the contour and training examples
        cont = plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

        return cont,

    ani = animation.FuncAnimation(fig, update, interval=25,
                                  blit=False, save_count=num_iter)
    ani.save(filepath, writer='ffmpeg')

import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class Space_2D:
    """Class to span 2D space"""

    def __init__(self, shape):
        """initialisation with a shape"""

        self.shape = shape
        self.generate_Grid()

    def generate_Grid(self):
        """generates a grid for the space with a 1 by 1 spacing"""

        self.gridx, self.gridy= np.mgrid[0:self.shape[0]:1, 0:self.shape[1]:1]

    def generate_Points(self, scale):
        """generates a set of points for the space with a 1/scale spacing"""

        self.pointsx, self.pointsy = np.mgrid[0:self.shape[0]:1/scale, 0:self.shape[1]:1/scale]

    def generate_Value_oct(self, freq):
        """generates a single octave of value noise for a given frequency"""

        T = 1/freq

        scaled_gridx, scaled_gridy = np.mgrid[0:self.shape[0]*T:T, 0:self.shape[1]*T:T]
        scaled_gridnodes = np.vstack([scaled_gridx.ravel(), scaled_gridy.ravel()]).T

        grid_values = np.random.uniform(low=-1, high=1, size=np.shape(scaled_gridnodes)[0])

        return sp.interpolate.griddata(scaled_gridnodes, grid_values, (self.pointsx, self.pointsy), method = "cubic")

    def generate_Value(self, octaves):
        """generates value noise by summing several octaves of noise"""

        self.data = np.empty_like(self.pointsx)

        for i in range(1, octaves):
            self.values += (0.5**i)*self.generate_Value_oct(2*i)

    def plot_Data(self, **kwargs):
        """creates a plot of the data
        kwargs: 
        elevation - creates an elevation plot
        """

        fig = plt.figure()

        if kwargs["elevation"] == True:
            ax = fig.add_subplot(1,2,1, projection='3d')
            ax.plot_surface(self.pointsx,self.pointsy,self.data, cmap=cm.viridis, linewidth=0, antialiased=False)

            ax = fig.add_subplot(122)
            ax.imshow(test_Space.values, cmap=cm.viridis)
        else:
            ax = fig.add_subplot(1,1,1, projection='3d')
            ax.plot_surface(self.pointsx,self.pointsy,self.data, cmap=cm.viridis, linewidth=0, antialiased=False)

        plt.show()
        plt.savefig("testnoise.png")

            
if __name__ == "__main__":
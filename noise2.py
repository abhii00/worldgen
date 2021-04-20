import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class Space_2D:
    def __init__(self, shape, points_scale, value_octaves):
        self.shape = shape

        self.generate_Grid()
        self.generate_Points(points_scale)
        self.generate_Value(value_octaves)

    def generate_Grid(self):
        self.gridx, self.gridy= np.mgrid[0:self.shape[0]:1, 0:self.shape[1]:1]

    def generate_Points(self, scale):
        self.pointsx, self.pointsy = np.mgrid[0:self.shape[0]:1/scale, 0:self.shape[1]:1/scale]

    def generate_Value_oct(self, scale):
        scaled_gridx, scaled_gridy = np.mgrid[0:self.shape[0]*scale:scale, 0:self.shape[1]*scale:scale]
        scaled_gridnodes = np.vstack([scaled_gridx.ravel(), scaled_gridy.ravel()]).T

        grid_values = np.random.uniform(low=-1, high=1, size=np.shape(scaled_gridnodes)[0])

        return sp.interpolate.griddata(scaled_gridnodes, grid_values, (self.pointsx, self.pointsy), method = "cubic")

    def generate_Value(self, octaves):
        self.values = np.empty_like(self.pointsx)

        for i in range(1, octaves):
            self.values += (0.5**i)*self.generate_Value_oct(2*i)
            
if __name__ == "__main__":
    test_Space = Space_2D((10,10),250,6)

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(test_Space.pointsx,test_Space.pointsy,test_Space.values, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax = fig.add_subplot(122)
    ax.imshow(test_Space.values, cmap=cm.viridis)
    plt.show()
    plt.savefig("testnoise.png")
    
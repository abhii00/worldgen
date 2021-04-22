import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class Space_2D:
    """class to span 2D space.

    Parameters: 
        shape (tuple): the shape of the grid
        points_scale (float): the scale between the points and the grid
    Returns:
        None
    """

    def __init__(self, shape, points_scale):
        """initialisation"""

        self.shape = shape
        self.points_scale = points_scale
        self.gridx, self.gridy = self.generate_Grid()
        self.pointsx, self.pointsy = self.generate_Points()

    def generate_Grid(self):
        """generates a grid with a 1 by 1 spacing.

        Parameters:
            None
        Returns:
            Pair of NxN arrays with 1x1 spacing
        """

        return np.mgrid[0:self.shape[0]:1, 0:self.shape[1]:1]

    def generate_Points(self, **kwargs):
        """generates a set of points with a 1/scale spacing.

        Parameters:
            scale (float): 1/spacing of points
        Returns:
            Pair of NxN arrays with kxk spacing where k = 1/scale
        """

        scale = kwargs.get("scale",self.points_scale)

        return np.mgrid[0:self.shape[0]-1:1/scale, 0:self.shape[1]-1:1/scale]

    def generate_Value_oct(self, freq):
        """generates simple value noise.
        
        Parameters:
            freq (int): frequency of the noise
        Returns:
            NxN array of value noise
        """

        #time period of noise
        period = 1/freq

        #create a scaled grid
        scaled_gridx, scaled_gridy = np.mgrid[0:self.shape[0]*period:period, 0:self.shape[1]*period:period]
        scaled_gridnodes = np.vstack([scaled_gridx.ravel(), scaled_gridy.ravel()]).T

        #generate random values for grid
        grid_values = np.random.uniform(low=-1, high=1, size=np.shape(scaled_gridnodes)[0])

        #interpolate for points from grid
        return sp.interpolate.griddata(scaled_gridnodes, grid_values, (self.pointsx, self.pointsy), method = "cubic")

    def generate_Noise(self, method, **kwargs):
        """generates noise.

        Parameters:
            method (str): pick between Gaussian, Value
            octaves (Value) (int): number of summed sets of value noise
            mu (Gaussian) (float): mean of gaussian
            sigma (Gaussian) (float): standard distribution of gaussian
        Returns:
            NxN array of noise
        """

        #initialise data array
        self.data = np.empty_like(self.pointsx)

        #Gaussian Noise
        if method == "Gaussian":
            mu = kwargs.get("mu", 0)
            sigma = kwargs.get("sigma", 1)

            self.data = np.random.normal(loc=mu, scale=sigma, size=np.shape(self.pointsx))

        #Value Noise
        elif method == "Value":
            octaves = kwargs.get("octaves", 8)
            if octaves > 10: octaves = 10

            for i in range(1, octaves):
                #generate single octave
                oct_data = self.generate_Value_oct(0.1*i)

                #calculate amplitude for octave
                oct_amp = 0.5**i

                #calculate the octave with the reduced amplitude
                oct_value = oct_data*oct_amp
                self.data += oct_value

                print("Octave {} generated with max value {:.3f} and min value {:.3f}.".format(i,oct_value.max(),oct_value.min()))
        
        #rescale to [-1,1]
        self.data = np.interp(self.data, (self.data.min(), self.data.max()), (-1, +1))

    def fill_Level(self, level):
        """fills data to certain level between -1 and 1.
        
        Parameters:
            level (float): the level between -1 and 1 to fill below
        Returns:
            None
        """

        below_fill_indices = self.data < level
        self.data[below_fill_indices] = level

    def plot_Data(self, **kwargs):
        """creates a plot of the data.

        Parameters: 
            elevation (bool): creates elevation plot
            color (cm.x): sets color map
            crop (float): crops to given fraction if specified
            file_name (str): output image file name
        Returns:
            None
        """

        fig = plt.figure()

        #color map
        color = kwargs.get("color", cm.viridis)

        #crop
        crop = kwargs.get("crop", False)
        if crop:
            crop = kwargs["crop"]
            px, py = self.generate_Points(scale=self.points_scale*crop)
            d = self.data[:np.shape(px)[0], :np.shape(px)[0]]
        else:
            px = self.pointsx
            py = self.pointsy
            d = self.data
            

        #elevation
        elevation = kwargs.get("elevation", False)

        #plot 3d + elevation
        colorcount = 100
        if elevation == True:
            ax1 = fig.add_subplot(1,2,1, projection='3d')
            ax1.plot_surface(px, py, d, cmap=color, linewidth=0, antialiased=False, rcount=colorcount, ccount=colorcount)
            ax1.axis("off")

            ax2 = fig.add_subplot(122)
            ax2.imshow(d, cmap=color)

        #plot 3d
        else:
            ax1 = fig.add_subplot(1,1,1, projection='3d')
            ax1.plot_surface(px, py, d, cmap=color, linewidth=0, antialiased=False, rcount=colorcount, ccount=colorcount)
            ax1.axis("off")

        #save to file
        file_name = kwargs.get("file_name", False)
        if type(file_name) == str:
            fig.savefig(file_name, dpi=fig.dpi, bbox_inches='tight')

        plt.show()

    def save_Data(self, file_name):
        """writes data to .csv file to 4sf

        Parameters:
            file_name (str): save file name with extension
        Returns:
            None
        """
        np.savetxt(file_name, self.data, delimiter=",", fmt="%.3f")

    def load_Data(self, file_name):
        """loads data from file
        
        Parameters:
            file_name (str): load file name with extension
        Returns:

        """
        self.data = np.loadtxt(file_name, delimiter=",")

if __name__ == "__main__":
    test_Space = Space_2D(shape=(10,10), points_scale=150)
    test_Space.generate_Noise(method="Value", octaves=6)
    test_Space.fill_Level(-0.25)
    test_Space.plot_Data(elevation=True, color=cm.viridis, crop = 0.5, file_name="testnoise.png")
    test_Space.save_Data("testnoise.csv")
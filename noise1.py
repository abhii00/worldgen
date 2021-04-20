import numpy as np
import matplotlib.pyplot as plt

class DataSet:
    """class to hold dataset"""
    def __init__(self, axis_start, axis_stop, **kwargs):
        """
        initialisation
        kwargs: 
            axis_division - number of axis divisions,
            axis_steps - number of axis steps
        """

        #define characteristics
        if "name" in kwargs:
            self.name = kwargs["name"]
        else: 
            self.name = ""
        if "color" in kwargs:
            self.color = kwargs["color"]
        else:
            self.color = (np.random.rand(), np.random.rand(), np.random.rand())

        #define axis characteristics
        if "axis_division" in kwargs: 
            self.axis_division = kwargs["axis_division"]
        if "axis_steps" in kwargs: 
            self.axis_steps = kwargs["axis_steps"]

        #create axis
        if "axis_division" in kwargs:
            self.axis = np.arange(axis_start, axis_stop, self.axis_division)
        elif "axis_steps" in kwargs:
            self.axis = np.linspace(axis_start, axis_stop, self.axis_steps)     
        self.axis_shape = np.shape(self.axis)

        self.data = []

    def generate_random(self, method, **kwargs):
        """
        generates random data using several methods
        method:
        Gaussian: X ~ N(mu, sigma**2)
        Perlin: chunks
        """

        #sample from Gaussian distribution
        if method == "Gaussian":
            if self.name == "":
                self.name = "Gaussian ({:.1f},{:.1f})".format(kwargs["mu"], kwargs["sigma"])
            self.data = np.random.normal(loc=kwargs["mu"], scale=kwargs["sigma"], size=self.axis_shape)

        #generate Perlin noise
        elif method == "Perlin":
            if self.name == "":
                self.name = "Perlin"

            #generate chunks and random gradient vectors for each chunk
            grid = np.linspace(self.axis[0], self.axis[-1], kwargs["chunks"])
            chunk_size = grid[1] - grid[0]
            grid_gradients = np.random.uniform(low=-1,high=1,size=np.shape(grid)[0])
            
            for i, x in enumerate(self.axis):
                n_indexes = [0,0]
                #calculate which box x is in
                n_indexes[0] = np.argmin(np.abs(grid-x))
                if n_indexes[0] != np.shape(grid)[0] - 1:
                    n_indexes[1] = n_indexes[0]+1
                else: 
                    n_indexes[1] = n_indexes[0]
                
                #calculate distance to x, and multiply with gradient
                n_locations = [grid[n_i] for n_i in n_indexes]
                n_gradients = [grid_gradients[n_i] for n_i in n_indexes]
                n_distances = 2*[x-n_locations[0], n_locations[1]-x]/chunk_size      
                n_values = [n_gradients[i]*n_distances[i] for i in [0,1]]

                #linearly interpolate and store
                x_value = np.interp(x, n_locations, n_values)
                self.data.append(x_value)

    def calculate_fft(self):
        """calculates the fft of the data"""
        self.fft = np.fft.fft(self.data)
        self.fft_shape = np.shape(self.fft)

    def perfect_filter(self, low_f_frac, up_f_frac):
        """filters the data using percentages of the total frequency points, keeps frequencies in between low_f_frac and up_f_frac"""
        self.calculate_fft()
        self.fft[:int(round(low_f_frac*self.fft_shape[0]))] = complex(0)
        self.fft[:int(round(up_f_frac*self.fft_shape[0]))] = complex(0)
        self.data = np.fft.ifft(self.fft)

    def simple_plot(self):
        """plots the data against the axis"""
        plt.plot(self.axis, self.data, self.color)
        plt.show()
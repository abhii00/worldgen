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

            #generate random vectors for gradients of each node
            chunk_size = int(round(self.axis_shape[0]/kwargs["chunks"]))
            node_locations = np.linspace(self.axis[0], self.axis[-1], chunk_size)
            node_gradients = np.random.uniform(low=-1, high=1, size=np.shape(node_locations))

            self.data = []
            #iterate over axis
            for i, point in enumerate(self.axis):
                #find nearest nodes, locations and gradients
                left_node_index = np.argmin(np.abs(node_locations-point))
                right_node_index = left_node_index + 1
                if right_node_index == np.shape(node_locations)[0]:
                    right_node_index = left_node_index
                left_node_location = node_locations[left_node_index]
                right_node_location = node_locations[right_node_index]
                left_node_gradients = node_gradients[left_node_index]
                right_node_gradients = node_gradients[right_node_index]

                #calculate distance vectors
                left_node_distance = point-left_node_location
                right_node_distance = right_node_location-point

                #calculate dot product with gradient vector
                left_node_value = left_node_distance*left_node_gradients
                right_node_value = right_node_distance*right_node_gradients

                #linearly interpolate for value
                value = np.interp(point, [left_node_location,right_node_location], [left_node_value,right_node_value])
                self.data.append(value)

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

def test_gaussian(plots):
    fig, axs = plt.subplots(plots)

    for i in range(0,plots):
        ax = axs[i]
        current_set = DataSet(name = "{}".format(i), axis_start=0, axis_stop=100, axis_division = 0.5)
        current_set.generate_random("Gaussian", mu=0, sigma=1)
        current_set.perfect_filter(low_f_frac=0, up_f_frac=i/plots)
        ax.plot(current_set.axis, current_set.data, color=current_set.color, label=current_set.name)

    plt.legend()
    plt.show()

def test_perlin():
    test_set = DataSet(name = "Perlin Test", axis_start=0, axis_stop=1000, axis_division = 0.5)
    test_set.generate_random(method="Perlin", chunks=30)

    fig, ax = plt.subplots()
    ax.plot(test_set.axis, test_set.data, color=test_set.color, label=test_set.name)
    plt.show()

if __name__ == "__main__":
    test_perlin()
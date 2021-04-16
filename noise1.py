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
        """

        #sample from Gaussian distribution
        if method == "Gaussian":
            if self.name == "":
                self.name = "Gaussian ({:.1f},{:.1f})".format(kwargs["mu"], kwargs["sigma"])
            self.data = np.random.normal(loc=kwargs["mu"], scale=kwargs["sigma"], size=self.axis_shape)

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

def test_1(plots):
    fig, axs = plt.subplots(plots)

    for i in range(0,plots):
        ax = axs[i]
        current_set = DataSet(name = "{}".format(i), axis_start=0, axis_stop=100, axis_division = 0.5)
        current_set.generate_random("Gaussian", mu=0, sigma=1)
        current_set.perfect_filter(low_f_frac=0, up_f_frac=i/plots)
        ax.plot(current_set.axis, current_set.data, color=current_set.color, label=current_set.name)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_1(5)
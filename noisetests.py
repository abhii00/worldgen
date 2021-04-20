from noise1.py import *

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
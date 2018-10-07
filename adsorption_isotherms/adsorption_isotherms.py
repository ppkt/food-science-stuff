import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


"""
Small program to calculate sorption isotherms using GAB model
"""


def gab_model(v2, c, k, x):
    """
    Model GAB

    :param v2: water activity
    :param c: model parameter 1
    :param k: model parameter 2
    :param x: monolayer moisture content
    :return:
    """
    return (x * c * k * v2) / ((1 - (k * v2)) * (1 - (k * v2) + (c * k * v2)))


if __name__ == '__main__':
    # read csv file
    data = pd.read_csv('isotherms.csv')
    x_data, y_data = data['v2'], data['v1']

    # fit model to experimental data
    popt, _ = curve_fit(gab_model, x_data, y_data, method='lm')

    # calculate R2 and print c, k, parameters
    new_y_data = gab_model(x_data, *popt)
    r2 = r2_score(y_data, new_y_data)
    print('r2:\t', r2)
    print('c:\t', popt[0])
    print('k:\t', popt[1])
    print('x:\t', popt[2])

    # plot curve
    plot_data=np.arange(0.0, x_data.iloc[-1] + 0.01, 0.01)
    plt.plot(plot_data, gab_model(plot_data, *popt))

    # plot experimental points
    plt.plot(x_data, y_data, 'o')

    # show plot
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

"""
Small program to calculate sorption isotherms using GAB model
"""


def gab_model(v2: float, c: float, k: float, x: float) -> float:
    """
    Model GAB - Guggenheim-Anderson-de Boer

    :param v2: water activity (model is accurate for values 0.0 - 0.75)
    :param c: model parameter 1
    :param k: model parameter 2
    :param x: monolayer moisture content
    :return:
    """
    return (x * c * k * v2) / ((1 - (k * v2)) * (1 - (k * v2) + (c * k * v2)))


def bet_model(v2: float, c: float, x: float) -> float:
    """
    Model BET - Brunauer–Emmett–Teller

    :param v2: water activity (model is accurate for values 0.0 - 0.5)
    :param c: model parameter
    :param x: monolayer moisture content
    :return:
    """
    return (x * c * v2) / ((1 - v2) * (1 + (c - 1) * v2))


def lewicki_model(v2, a, b, c):
    """
    Empirical model described by P. Lewicki

    :param v2: water activity (0.0 - 1.0)
    :param a: model parameter 1
    :param b: model parameter 2
    :param c: model parameter 3
    :return:
    """
    return (a / np.power(1 - v2, b)) - (a / (1 + np.power(v2, c)))


def peleg_model(v2, a, b, c, d):
    """
    Empirical model described by Peleg

    :param v2: water activitiy (0.0 - 1.0)
    :param a: model parameter 1
    :param b: model parameter 2
    :param c: model parameter 3
    :param d: model parameter 4
    :return:
    """
    return (a * np.power(v2, c)) + (b * np.power(v2, d))


if __name__ == '__main__':
    # read csv file
    data = pd.read_csv('isotherms.csv')
    x_data, y_data = data['v2'], data['v1']

    # model
    model = peleg_model

    # fit model to experimental data
    popt, _ = curve_fit(model, x_data, y_data, method='lm')

    # calculate R2 and print c, k, parameters
    new_y_data = model(x_data, *popt)
    r2 = r2_score(y_data, new_y_data)
    print('r2:\t', r2)
    print('parameters:\t', popt)

    # plot curve
    plot_data = np.linspace(0, x_data.iloc[-1], num=500)
    plt.plot(plot_data, model(plot_data, *popt))

    # plot experimental points
    plt.plot(x_data, y_data, 'o')

    # show plot
    plt.show()

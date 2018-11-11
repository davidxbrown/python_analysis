# -*- coding: utf-8 -*-
"""
In this file synthetic data is created and various statistical tools are tested.

"""

# What do I want to do
# create a 100 pt data set with range 0 to 1, with a certain amount of noise on each point
# plot r2 as a function of the noise
# profit

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def make_test_set(numPoints, sigmaX, sigmaY):
    """creates a random linear data set with stated noise and numPoints.
    range of data is from 0 to 1
    
    args:
        numPoints: number of points in data set
        sigmaX: standard deviation on x points
        sigmaY: standard deviation on y points
    
    returns:
        valuePairs: list of x,y values
    """
    minX = 0
    maxX = 1
    trueXY = np.linspace(minX, maxX, numPoints)
    yVals = [x + np.random.normal(scale=sigmaY) for x in trueXY]
    xVals = [x + np.random.normal(scale=sigmaX) for x in trueXY]

    valuePairs = zip(xVals, yVals)
    return valuePairs
    # print(list(valuePairs))


def plot_zip_pair(zippedList, settingsString):
    x, y = zip(*zippedList)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(slope, intercept, r_value, p_value, std_err)

    plt.figure(1)
    plt.title("Correlation Example")
    plt.text(-0.1, 0.75, settingsString + "r2 = " + str(r_value ** 2)[0:6], fontsize=16)
    plt.axis([-0.2, 1.2, -0.2, 1.2])
    plt.grid(True)
    plt.plot(x, y, "*")
    plt.show()


def test_plot(numPoints, sigmaX, sigmaY):
    vP = make_test_set(numPoints, sigmaX, sigmaY)
    settingsString = (
        "Points = "
        + str(numPoints)
        + "\nsigX = "
        + str(sigmaX)
        + "\nsigY = "
        + str(sigmaY)
        + "\n"
    )
    plot_zip_pair(vP, settingsString)


def calc_mult_sets(numSets, numPoints, sigmaX, sigmaYs):
    r2mean = []
    r2std = []
    p_mean = []
    p_std = []
    std_err_mean = []
    std_err_std = []
    for sigmaY in sigmaYs:
        r2 = []
        p = []
        std_errs = []
        for x in range(0, numSets):
            vp = make_test_set(numPoints, sigmaX, sigmaY)
            x, y = zip(*vp)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r2.append(r_value)
            p.append(p_value)
            std_errs.append(std_err)
        r2mean.append(np.mean(r2))
        r2std.append(np.std(r2))
        p_mean.append(np.mean(p))
        p_std.append(np.std(p))
        std_err_mean.append(np.mean(std_errs))
        std_err_std.append(np.std(std_errs))

    return zip(sigmaYs, r2mean, r2std, p_mean, p_std, std_err_mean, std_err_std)


def code_tbd_1(numPoints, sigmaY, numSets=100):
    """given error to range ratio, return expected max r2"""
    r2 = []
    for x in range(0, numSets):
        vp = make_test_set(numPoints, 0, sigmaY)
        x, y = zip(*vp)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r2.append(r_value)
    return np.mean(r2), np.std(r2)


def code_tbd_2():
    """given expected r2, return maximum error to range ratio"""


if __name__ == "__main__":
    numPoints = 101
    sigmaX = 0.00
    sigmaY = 0.1
    test_plot(numPoints, sigmaX, sigmaY)

    numSets = 100
    sigmaYs = np.logspace(-2, 1, 50)
    r2s = calc_mult_sets(numSets, numPoints, sigmaX, sigmaYs)

    sigmaY, r2mean, r2std, p_mean, p_std, std_err_mean, std_err_std = zip(*r2s)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=False)
    ax0 = axs[0, 0]
    ax0.set_title("r2 as a function of std/range")
    ax0.errorbar(sigmaY, r2mean, yerr=r2std, fmt="o")
    ax0.axis([-0.01, max(sigmaY), -0.1, 1.2])
    ax0.grid(True)

    ax1 = axs[0, 1]
    ax1.set_title("r2 as a function of std/range")
    ax1.errorbar(sigmaY, r2mean, yerr=r2std, fmt="o")
    ax1.axis([-0.01, 1, 0.4, 1.1])
    ax1.grid(True)

    ax3 = axs[1, 0]
    ax3.set_title("r2 std as function of  std/range")
    ax3.plot(sigmaY, r2std, "*")
    ax3.axis([-0.01, max(sigmaY), -0.1, max(r2std) * 1.2])
    ax3.grid(True)

    ax4 = axs[1, 1]
    ax4.set_title("r2 std as function of std/range")
    ax4.plot(sigmaY, r2std, "*")
    ax4.axis([-0.01, 1, 0, 0.2])
    ax4.grid(True)

    plt.show()

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=False)
    ax0 = axs[0, 0]
    ax0.set_title("p as a function of std/range")
    ax0.errorbar(sigmaY, p_mean, yerr=p_std, fmt="o")
    ax0.axis([-0.01, max(sigmaY), -0.1, 1.2])
    ax0.grid(True)

    ax1 = axs[0, 1]
    ax1.set_title("p as a function of std/range")
    ax1.errorbar(sigmaY, p_mean, yerr=p_std, fmt="o")
    ax1.axis([-0.01, 1, -0.1, 0.1])
    ax1.grid(True)

    ax3 = axs[1, 0]
    ax3.set_title("p std as function of  std/range")
    ax3.plot(sigmaY, p_std, "*")
    ax3.axis([-0.01, max(sigmaY), -0.1, max(p_std) * 1.2])
    ax3.grid(True)

    ax4 = axs[1, 1]
    ax4.set_title("p std as function of std/range")
    ax4.plot(sigmaY, p_std, "*")
    ax4.axis([-0.01, 1, 0, 0.2])
    ax4.grid(True)

    plt.show()

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=False)
    ax0 = axs[0, 0]
    ax0.set_title("std_err  as a function of std/range")
    ax0.errorbar(sigmaY, std_err_mean, yerr=std_err_std, fmt="o")
    ax0.axis([-0.01, max(sigmaY), -0.1, np.max(std_err_mean)])
    ax0.grid(True)

    ax1 = axs[0, 1]
    ax1.set_title("std_err as a function of std/range")
    ax1.errorbar(sigmaY, std_err_mean, yerr=std_err_std, fmt="o")
    ax1.axis([-0.01, 1, -0.1, 0.1])
    ax1.grid(True)

    ax3 = axs[1, 0]
    ax3.set_title("std_err as function of  std/range")
    ax3.plot(sigmaY, p_std, "*")
    ax3.axis([-0.01, max(sigmaY), -0.1, max(std_err_std) * 1.2])
    ax3.grid(True)

    ax4 = axs[1, 1]
    ax4.set_title("std_err as function of std/range")
    ax4.plot(sigmaY, std_err_std, "*")
    ax4.axis([-0.01, 1, 0, 0.2])
    ax4.grid(True)

    plt.show()

    print(code_tbd_1(101, 0.1, numSets=1000))

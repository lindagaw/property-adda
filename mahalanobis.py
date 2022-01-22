import os
import numpy as np
import pretty_errors

def load_data(city_name):
    xs = np.load('data//pre-processed//'+ city_name + '_finalized_x.npy')
    xs = np.expand_dims(xs, axis=1)
    ys = np.load('data//pre-processed//'+ city_name + '_finalized_y.npy')
    print("{} xs has shape {}".format(city_name, xs.shape))
    print("{} ys has shape {}".format(city_name, ys.shape))
    return xs, ys

beijing_xs, beijing_ys = load_data('beijing')
tianjin_xs, tianjin_ys = load_data('tianjin')
shenzhen_xs, shenzhen_ys = load_data('shenzhen')
guangzhou_xs, guangzhou_ys = load_data('guangzhou')

def empirical_mean(xs):
    xs = np.squeeze(xs)
    norms = []
    for x in xs:
        norm = np.linalg.norm(x)
        norms.append(norm)

    empirical_mean = np.mean(norms)
    return empirical_mean

def inv_empirical_cov(xs):
    xs = np.squeeze(xs)
    norms = []
    for x in xs:
        norm = np.linalg.norm(x)
        norms.append(norm)

    cov = np.cov(norms)
    return cov

def mahalanobis(x, mean, inv_cov):
    x = np.squeeze(x)
    norm = np.linalg.norm(x)
    diff = norm - mean
    return diff * mean * inv_cov

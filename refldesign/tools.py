import numpy as np
import tensorflow as tf

def approxmap(P, a, b, method="basic"):
    if method=="basic":
        Tx = b[np.argmax(P, axis=1)]
        x = a 
    elif method=="averaging":
        Tx = np.sum(P*np.expand_dims(b, axis=0), axis=1)/np.sum(P, axis=1)
        x = a
    else:
        print(f"Method not implemented : {method}.")
        raise
    return x, Tx

def ctransform(C, phi):
    return np.min(C - phi.reshape(-1, 1), axis=0)

def cbartransform(C, psi):
    return np.min(C - psi.reshape(1, -1), axis=1)

@tf.function
def to_col(v):
    return tf.reshape(v, [-1, 1])

@tf.function
def to_row(v):
    return tf.reshape(v, [1, -1])

@tf.function
def sinkhornplan(C, eps, alpha, beta):
    return tf.math.exp(-(C - to_col(alpha) - to_row(beta))/eps)

@tf.function
def n_inf(x):
    return tf.math.reduce_max(tf.math.abs(x))

@tf.function
def min_norm(x):
    return tf.math.reduce_min(tf.math.abs(x))

@tf.function
def vect(x):
    return tf.reshape(x, [-1])

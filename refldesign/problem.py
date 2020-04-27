import tensorflow as tf
import tensorflow_probability as tfp
from numpy import pi
tfd = tfp.distributions

def build_problem(input_n, output_n):
    #astart, aend = tf.cast(pi/4., tf.float64), tf.cast(3.*pi/4., tf.float64)
    astart, aend = tf.cast(0, tf.float64), tf.cast(1, tf.float64)
    #bstart, bend = tf.cast(5.*pi/4., tf.float64), tf.cast(7.*pi/4., tf.float64)
    bstart, bend = tf.cast(0, tf.float64), tf.cast(1, tf.float64)
    a = tf.linspace(astart, aend, input_n)
    b = tf.linspace(bstart, bend, output_n)
    
    locs, scales = tf.random.uniform((8,), 0.2, 0.8, dtype=tf.float64), tf.random.uniform((8,), 0.01, 0.10, dtype=tf.float64)

    mu = tf.zeros_like(a)
    for i in range(4):
        mu += tfd.Normal(locs[i], scales[i]).prob(tf.linspace(tf.cast(0., tf.float64), tf.cast(1., tf.float64), input_n))
    mu = mu/tf.math.reduce_sum(mu)
    nu = tf.zeros_like(a)
    for i in range(4,8):
        nu += tfd.Normal(locs[i], scales[i]).prob(tf.linspace(tf.cast(0., tf.float64), tf.cast(1., tf.float64), output_n))
    nu = nu/tf.math.reduce_sum(nu)
    #mu = tf.ones((input_n,), dtype=tf.float64)/input_n
    #nu = tfd.Normal(locs[4], scales[4]).prob(tf.linspace(tf.cast(0., tf.float64), tf.cast(1., tf.float64), output_n))

    mu = mu/tf.math.reduce_sum(mu)
    nu = nu/tf.math.reduce_sum(nu)

    A, B = tf.meshgrid(a, b, indexing='ij')
    #C = -tf.math.log(1 - tf.math.cos(A - B))
    C = (1/2)*tf.math.square(A - B)
    return C, a, b, mu, nu

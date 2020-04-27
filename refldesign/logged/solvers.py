import tensorflow as tf
import numpy as np
from ..tools import sinkhornplan, to_row, to_col, n_inf, vect, min_norm

def sorted1d(C, mu, nu):
    i, j = 0, 0
    n, m = mu.shape[0], nu.shape[0]
    wmu, wnu = mu.copy(), nu.copy()
    cost = 0
    P = np.zeros((n,m))
    while i < n and j < m:
        if wmu[i] < wnu[j] or j == m - 1:
            cost += wmu[i] * C[i, j]
            P[i, j] = wmu[i]
            wnu[j] -= wmu[i]
            wmu[i] = 0
            i += 1
        elif wnu[j] < wmu[i] or i == n - 1:
            cost += wnu[j] * C[i, j]
            P[i, j] = wnu[j]
            wmu[i] -= wnu[j]
            wnu[j] = 0
            j += 1
        else:
            cost += wmu[i] * C[i, j]
            P[i, j] = wmu[i]
            wnu[j] = 0
            wmu[i] = 0
            i += 1
            j += 1
    return cost, P

@tf.function
def shsoftmin(z, eps, axis=1):
    return -eps * tf.math.reduce_logsumexp(-z/eps, axis=axis)

#@tf.function
def sinkhorn_log(C, mu, nu, alpha0=None, beta0=None, eps=1/10, error=1e-9, maxiter=10000):
    runlog = dict()
    
    iternum = tf.TensorArray(tf.int64, size=0, dynamic_size=True)
    margerr = tf.TensorArray(tf.float64, size=0, dynamic_size=True)

    n=mu.shape[0]
    m=nu.shape[0]
    
    eps = tf.cast(eps, tf.float64)


    if alpha0 is None:
        alpha0 = tf.ones((n,), dtype=tf.float64)/n  
    if beta0 is None:
        beta0 = tf.ones((m,), dtype=tf.float64)/m
    alpha = alpha0
    beta = beta0

    ilog = 0
    
    for i in tf.range(maxiter):
        alpha = eps*tf.math.log(mu) + shsoftmin(C - tf.expand_dims(beta, axis=0), eps, axis=1)
        beta = eps*tf.math.log(nu) + shsoftmin(C - tf.expand_dims(alpha, axis=1), eps, axis=0)
        if i % 5 == 0:
            K = sinkhornplan(C, eps, alpha, beta)
            marginal_mu = tf.math.reduce_sum(K, axis = 1)
            diff_mu = n_inf(vect(mu) - marginal_mu)
            iternum = iternum.write(ilog, tf.cast(i, tf.int64))
            margerr = margerr.write(ilog, tf.cast(diff_mu, tf.float64))
            if diff_mu < error:
                break
            ilog += 1

    runlog["iternum"] = iternum.stack()
    runlog["margerr"] = margerr.stack()
    return alpha, beta, runlog



def sinkhorn_stabilized(C, mu, nu, eps, tau=1e3, alpha0=None, beta0=None):
    runlog = dict()
    
    iternum = tf.TensorArray(tf.uint32, size=0, dynamic_size=True)
    margerr = tf.TensorArray(tf.uint32, size=0, dynamic_size=True)

    n, m = mu.shape[0], nu.shape[0]
    
    if alpha0 is None:
        alpha0 = tf.ones((n,), dtype=tf.float64)/n
    if beta0 is None:
        beta0 = tf.ones((m,), dtype=tf.float64)/m
    
    alpha, beta = to_col(alpha0), to_col(beta0)
    
    u = tf.ones((n,1), dtype=tf.float64)
    v = tf.ones((m,1), dtype=tf.float64)
    
    mu = to_col(mu)
    nu = to_col(nu)
    
    niters = 0
    
    K = sinkhornplan(C, eps, alpha, beta)
    
    while True:
        u = mu/(K@v)
        v = nu/(tf.transpose(K)@u)    
        niters += 1

        if n_inf(u) > tau or n_inf(v) > tau:
            alpha += eps * tf.math.log(u)
            beta += eps* tf.math.log(v)
            
            u = tf.ones((n,1), dtype=tf.float64)
            v = tf.ones((m,1), dtype=tf.float64)
            K = sinkhornplan(C, eps, alpha, beta)
        
        if niters % 5 == 0:
            Kf = sinkhornplan(C, eps, alpha + eps*tf.math.log(u), beta + eps*tf.math.log(v))
            mga = tf.reduce_sum(Kf, axis=1)
            diffmu = n_inf(vect(mu) - mga)
            iternum.write(ilog, i)
            margerr.write(ilog, diff_mu)
            if diffmu < 1e-9:
                break
                
    alpha += eps * tf.math.log(u)
    beta += eps* tf.math.log(v)
    K = sinkhornplan(C, eps, alpha, beta)
    return vect(alpha), vect(beta)

def geom_eps(eps0, lbd, niter):
    eps = tf.constant(eps0, dtype=tf.float64)
    for i in range(niter):
        yield eps
        eps *= lbd

def epsilon_scaling(C, mu, nu, niter, eps0=1., lbd=0.75, tau=1e3, epscor=True):
    n=mu.shape[0]
    m=nu.shape[0]

    alpha = tf.zeros((n,), dtype=tf.float64)
    beta = tf.zeros((m,), dtype=tf.float64)
    alphasnocor = []
    alphas = []
    betasnocor = []
    betas = []
    epss = []

    runlog = dict()

    totiters = tf.constant([], tf.int64)
    totmargerrs = tf.constant([], tf.float64)


    for eps in geom_eps(eps0, lbd, niter):
        epss.append(eps)
        alphan, betan, runlog = sinkhorn_log(C, mu, nu, eps=eps, alpha0=alpha, beta0=beta, maxiter=10)
        if len(totiters) > 0:
            totiters = tf.concat([totiters, runlog["iternum"]+totiters[-1]], axis=0)
        else:
            totiters = tf.concat([totiters, runlog["iternum"]], axis=0)
        totmargerrs = tf.concat([totmargerrs, runlog["margerr"]], axis=0)

        alpha, beta = alphan, betan
        alphasnocor.append(alpha-alpha[0])
        betasnocor.append(beta+alpha[0])
        
        if epscor:
            alpha = alpha - eps*(1-lbd)*tf.math.log(mu)
            beta = beta - eps*(1-lbd)*tf.math.log(nu)
            beta = beta + alpha[0]
            alpha = alpha - alpha[0]
        alphas.append(alpha)
        betas.append(beta)
    runlog["iternum"] = totiters
    runlog["margerr"] = totmargerrs
    return alphasnocor, alphas, betasnocor, betas, epss, runlog




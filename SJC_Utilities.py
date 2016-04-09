def heaviside(x):
    from numpy import sign
    y = 0.5 * ( sign(x) + 1.0 )
    return y


# Set IC
# Note: MATLAB behaves differently in this part.
def setIC(X_Mat, IC_Flag):
    from SJC_Utilities import heaviside
    from numpy import sin, pi
    from sys import exit
    if IC_Flag == 1:
        U_Mat = sin(pi * X_Mat)
    elif IC_Flag == 2:
        U_Mat = 0.5 - sin(pi * X_Mat)
    elif IC_Flag == 3:
        U_Mat = 1.0 \
            + heaviside(X_Mat+1.0/4.0) \
            - heaviside(X_Mat-1.0/4.0)
    else:
        exit('Initial Condition Error!')

    return U_Mat

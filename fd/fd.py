import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from time import sleep
###variable declarations
nx = 101
c = 2
x1 = 0.0
x2 = 2.0
y1 = -1.1
y2 = 1.1
dx = (x2 - x1) / (nx - 1)
sigma = 5E-2
dt = sigma * dx / c
tEnd = (x2 - x1) / c
n = 0
t = n * dt
x = np.linspace(x1,x2,nx)
# # set IC 1
# u = np.zeros(x.shape)
# Ind1_4 = 25
# Ind3_4 = 75
# u[:Ind1_4] = u[:Ind1_4] - 1
# u[Ind1_4:Ind3_4] = np.cos(np.pi * x[Ind1_4:Ind3_4])
# u[Ind3_4:] = u[Ind3_4:] + 1
# set IC 2
u = np.sin(np.pi * x)

CoefA_Vec = [0.0, \
    -567301805773.0/1357537059087.0, \
    -2404267990393.0/2016746695238.0, \
    -3550918686646.0/2091501179385.0, \
    -1275806237668.0/842570457699.0]
CoefB_Vec = [1432997174477.0/9575080441755.0, \
    5161836677717.0/13612068292357.0, \
    1720146321549.0/2090206949498.0, \
    3134564353537.0/4481467310338.0, \
    2277821191437.0/14882151754819.0]
CoefC_Vec = [0.0, \
    1432997174477.0/9575080441755.0, \
    2526269341429.0/6820363962896.0, \
    2006345519317.0/3224310063776.0, \
    2802321613138.0/2924317926251.0]
ResU = np.zeros(u.shape)

def getdF(u, dx):
    dF = np.zeros(u.shape)
# # Linear Convection Equation
# # 2nd order spatial discretization
#     dF[1:-1] = (u[2:] - u[:-2]) / (2*dx)
#     dF[0] = (u[1] - u[-1]) / (2*dx)
#     dF[-1] = (u[0] - u[-2]) / (2*dx)
# # Burgers equation
# # 2nd order spatial discretization
    dF[1:-1] = u[1:-1] * (u[2:] - u[:-2]) / (2*dx)
    dF[0] = u[0] * (u[1] - u[-1]) / (2*dx)
    dF[-1] = u[-1] * (u[0] - u[-2]) / (2*dx)
    return dF

plt.ion()
while t < tEnd:
    t = n * dt
    if np.mod(n, 80) == 0:
        print(('%.3d' % n) + ': ' + ('%.4f' % t) + ' / ' + ('%.4f' % tEnd))
        plt.plot(x, u, '.-')
        plt.axis((x1, x2, y1, y2))
        plt.draw()
        sleep(0.2)
        # plt.show()
        # FigName = ('%.4f' % t) + '.jpg'
        # plt.savefig(FigName)
        plt.clf()
    # u[1:]=u[1:]-dt*(u[1:]*(u[1:]-u[:-1])/dx)
    for Ind in range(5):
        dF = getdF(u, dx)
        print(dF[0], dF[-1])
        ResU = CoefA_Vec[Ind] * ResU - dt * (-dF)
        u = u - CoefB_Vec[Ind] * ResU
        # u[0] = -1.0
        # u[-1] = 1.0
    if u.max() > 1E2:
        exit('Divergence!')
    n = n + 1


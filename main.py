from SJC_Mesh import Mesh1D
from SJC_SpectralToolbox import Poly
from IC import setIC
# from SJC_Equation import ConvectionLinearEq
from SJC_Equation import ConvectionNonlinearEq
from SJC_TimeDiscretization import RungeKutta54_LS
from numpy import mod, save, load, arange, where, sin, pi
import matplotlib.pyplot as plt
plt.style.use('ggplot')
###########################################################

Range = [-0.5, 0.5]
OrderNMAX = 4
CellNMAX = 40
QuadType = 'LGL'

XMesh = Mesh1D(Range, OrderNMAX, CellNMAX, QuadType)
Local_Vec = XMesh.getSolutionPoints()
Global_Mat = XMesh.getGlobalCoordinates()

SJCPoly = Poly(Local_Vec)

# 1: Sine Wave
# 2: Quadratic Polynomial
# 3: Square Jump
IC_Flag = 2
U0_Mat = setIC(Global_Mat, IC_Flag)
ConvA = 1.0 # Needs correction for nonlinear case
CFL = 5E-2
TimeStep = CFL * XMesh.getCellSize() / abs(ConvA)
TimeEnd = (Range[1] - Range[0]) / ConvA
# TimeEnd = 1.0 / (4.0 * Range[1])
Time = 0.0
TimeInd = 0
U_Mat = U0_Mat

ArtDiffuFlag = 0
# Eq = ConvectionLinearEq(Global_Mat, U0_Mat, ConvA, ArtDiffuFlag)
Eq = ConvectionNonlinearEq(Global_Mat, U0_Mat, ArtDiffuFlag, Time)

Plot_YMin = -1.1
Plot_YMax = 1.1
while (Time < TimeEnd):
    Time = TimeInd * TimeStep
    if Time > TimeEnd:
        Time = TimeEnd
    if mod(TimeInd, 1) == 0:
        print(('%.4d' % TimeInd) + ': ' + ('%.4f' % Time) + ' / ' + ('%.4f' % TimeEnd))
        plt.plot(Global_Mat, U_Mat, '.-')
        plt.axis(( Range[0], Range[1], Plot_YMin, Plot_YMax ))
        FigName_Str = ('%03d' % TimeInd) + '.jpg'
        plt.savefig(FigName_Str)
        plt.clf()
    U_Mat = RungeKutta54_LS(U_Mat, TimeStep, Eq, XMesh, SJCPoly, Time)
    if U_Mat.max() > 1E3:
        exit('Divergence!')
    TimeInd = TimeInd + 1

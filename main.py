from SJC_Mesh import Mesh1D
from SJC_SpectralToolbox import Poly
from IC import setIC
from SJC_Equation import ConvectionEq
from SJC_TimeDiscretization import EulerForward, RungeKutta54_LS
from numpy import mod, amax
from sys import exit
import matplotlib.pyplot as plt
plt.style.use('ggplot')
###########################################################

Range = [-1.0, 1.0]
OrderNMAX = 8
CellNMAX = 80
QuadType = 'LGL'

XMesh = Mesh1D(Range, OrderNMAX, CellNMAX, QuadType)
Local_Vec = XMesh.getSolutionPoints()
Global_Mat = XMesh.getGlobalCoordinates()

SJCPoly = Poly(Local_Vec)

# 2: Shifted Sine Wave;
# 3: Square Jump
IC_Flag = 3
U0_Mat = setIC(Global_Mat, IC_Flag)

ConvA = 20.0
CFL = 5E-2
TimeStep = CFL * XMesh.getCellSize() / ConvA
TimeEnd = 1.0 / ConvA

if IC_Flag == 3:
    ArtDiffuFlag = 1
elif IC_Flag == 2:
    ArtDiffuFlag = 0
else:
    exit('IC Flag Error!')

Eq = ConvectionEq(Global_Mat, U0_Mat, ConvA, ArtDiffuFlag)

TimeInd = 0
Time = 0.0
U_Mat = U0_Mat
while (Time < TimeEnd):
    Time = TimeInd * TimeStep
    if Time > TimeEnd:
        Time = TimeEnd
    if mod(TimeInd, 10) == 0:
        print(('%.4f' % Time) + ' / ' + ('%.4f' % TimeEnd))
        plt.plot(Global_Mat, U_Mat, '.-')
        FigName_Str = ('%.4f' % Time) + '.jpg'
        plt.savefig(FigName_Str)
        plt.clf()
    # U_Mat = EulerForward(U_Mat, TimeStep, Eq, XMesh, SJCPoly)
    U_Mat = RungeKutta54_LS(U_Mat, TimeStep, Eq, XMesh, SJCPoly)
    if amax(U_Mat) > 1E3:
        exit('Divergence!')
    TimeInd = TimeInd + 1

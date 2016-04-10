###########################################################
# Define Auxiliary equation
# q = u_x
###########################################################
class AuxEq(object):

    def __init__(self, X_Mat, Sol_Mat):
        self.X = X_Mat
        self.Sol = Sol_Mat

    def getSolFace(self, XMesh, BCType=1):
        from numpy import ones
        SolFaceL_Vec = ones((XMesh.CellNMAX+1,))
        SolFaceL_Vec[0:-1] = self.Sol[0, :]
        SolFaceR_Vec = ones((XMesh.CellNMAX+1,))
        SolFaceR_Vec[1:] = self.Sol[-1, :]
        # Set Periodic BC
        if BCType == 1:
            SolFaceL_Vec[-1] = SolFaceL_Vec[0]
            SolFaceR_Vec[0] = SolFaceR_Vec[-1]
        else:
            exit('BC Error!')
        return SolFaceL_Vec, SolFaceR_Vec

    def getSolFluxFace(self, XMesh):
        SolFaceL_Vec, SolFaceR_Vec = self.getSolFace(XMesh)
        CoefK = 1.0
        SolFlux_Vec = CoefK * SolFaceR_Vec + (1.0 - CoefK) * SolFaceL_Vec
        SolFluxL_Vec = SolFlux_Vec[0:-1]
        SolFluxR_Vec = SolFlux_Vec[1:]
        return SolFluxL_Vec, SolFluxR_Vec

    def getSolDeriv(self, Poly):
        from numpy import dot
        # Without Correction Procedure, the Solution Derivative is just the following line.
        SolDeriv_Mat = dot(Poly.getLagrangePolyDeriv(), self.Sol)
        return SolDeriv_Mat

    def getSolDeriv_CPR(self, XMesh, Poly):
        from numpy import dot
        SolDeriv_Mat = self.getSolDeriv(Poly)
        # Construct Discontinuous Solution Polynomial
        SolDiscL_Vec = self.Sol[0, :]
        SolDiscR_Vec = self.Sol[-1, :]
        SolFluxL_Vec, SolFluxR_Vec = self.getSolFluxFace(XMesh)
        # Compute the derivative of continuous flux polynomial
        SolDeriv_CPR_Mat = 1.0 / XMesh.getJacobian() * ( SolDeriv_Mat \
            + dot(Poly.getRadauRightPoly().reshape((Poly.Order+1, 1)), \
                (SolFluxL_Vec - SolDiscL_Vec).reshape((1, XMesh.CellNMAX))) \
            + dot(Poly.getRadauLeftPoly().reshape((Poly.Order+1, 1)), \
                (SolFluxR_Vec - SolDiscR_Vec).reshape((1, XMesh.CellNMAX)) ) )
        return SolDeriv_CPR_Mat

###########################################################
class ConvectionEq(AuxEq):
###########################################################
# Define Convection equation
# u_t + a * u_x = 0
# a = const
# f(u) = a * u, f is flux
###########################################################
    def __init__(self, X_Mat, Sol_Mat, ConvectionA, Flag):
        self.X = X_Mat
        self.Sol = Sol_Mat
        self.A = ConvectionA
        self.Flux = ConvectionA * Sol_Mat
        self.ArtDiffuFlag = Flag

    def update(self, Sol_Mat):
        self.Sol = Sol_Mat
        self.Flux = self.A * Sol_Mat
###########################################################
    def getFluxFluxFace(self, XMesh):
        SolFaceL_Vec, SolFaceR_Vec = self.getSolFace(XMesh)
        # Compute True Flux values at the interfaces
        # Flux Difference Splitting expression
        Flux_Vec = 0.5 * ( \
            self.A * (SolFaceL_Vec + SolFaceR_Vec) \
            - abs(self.A) * (SolFaceL_Vec - SolFaceR_Vec) )
        FluxL_Vec = Flux_Vec[0:-1]
        FluxR_Vec = Flux_Vec[1:]
        return FluxL_Vec, FluxR_Vec

    def getFluxDeriv(self, Poly):
        from numpy import dot
        # Without Correction Procedure, the Flux is just the following line.
        FluxDeriv_Mat = dot(Poly.getLagrangePolyDeriv(), self.Flux)
        return FluxDeriv_Mat

    def getFluxDeriv_CPR(self, XMesh, Poly):
        from numpy import dot
        FluxDeriv_Mat = self.getFluxDeriv(Poly)
        # Construct Discontinuous Flux Polynomial
        FluxDiscL_Vec = self.Flux[0, :]
        FluxDiscR_Vec = self.Flux[-1, :]
        FluxL_Vec, FluxR_Vec = self.getFluxFluxFace(XMesh)
        # Compute the derivative of continuous flux polynomial
        FluxDeriv_CPR_Mat = 1.0 / XMesh.getJacobian() * ( FluxDeriv_Mat \
            + dot(Poly.getRadauRightPoly().reshape((Poly.Order+1, 1)), \
                (FluxL_Vec - FluxDiscL_Vec).reshape((1, XMesh.CellNMAX))) \
            + dot(Poly.getRadauLeftPoly().reshape((Poly.Order+1, 1)), \
                (FluxR_Vec - FluxDiscR_Vec).reshape((1, XMesh.CellNMAX)) ) )
        return FluxDeriv_CPR_Mat

    def getSolDeriv_CPR_ArtDiffu(self, XMesh, Poly):
        from numpy import ones, dot
        Vandermonde = Poly.getVandermondeLegendre()
        CellSize = XMesh.getCellSize()
        Epsilon_e = self.getSmoothIndicator(Vandermonde, CellSize)
        FluxDeriv_CPR_Mat = self.getSolDeriv_CPR(XMesh, Poly)
        FluxDeriv_CPR_ArtDiffu_Mat = \
            dot( ones((Poly.Order+1,1)), Epsilon_e.reshape((1, Epsilon_e.size)) ) \
            * FluxDeriv_CPR_Mat
        return FluxDeriv_CPR_ArtDiffu_Mat

    def getSolDeriv2_CPR_ArtDiffu(self, XMesh, Poly):
        Temp = self.Sol
        self.Sol = self.getSolDeriv_CPR_ArtDiffu(XMesh, Poly)
        SolDeriv2_CPR_ArtDiffu_Mat = self.getSolDeriv_CPR(XMesh, Poly)
        self.Sol = Temp # Ensure Solution in Eq is not changed.
        return SolDeriv2_CPR_ArtDiffu_Mat

    def getdF(self, ArtDiffuFlag, XMesh, Poly):
        if ArtDiffuFlag == 0:
            f_x = self.getFluxDeriv_CPR(XMesh, Poly)
            dF = f_x
        elif ArtDiffuFlag == 1:
            f_x = self.getFluxDeriv_CPR(XMesh, Poly)
            u_xx = self.getSolDeriv2_CPR_ArtDiffu(XMesh, Poly)
            dF = f_x - u_xx
        else:
            exit('Flag Error!')
        return dF

###########################################################
    def getSmoothIndicator(self, Vandermonde, CellSize):
        from numpy.linalg import solve
        from numpy import dot, inner, log10, shape, where, isinf, nan_to_num, sin, pi
        SolBasis = solve(Vandermonde, self.Sol)
        # Truncated Solution Matrix
        SolBasis[-1, :] = 0
        SolTrunc = dot(Vandermonde, SolBasis)
        # Smooth/Resolution Indicator
        CellNMAX = shape(self.Sol)[1]
        S_e = [ inner(self.Sol[:, Ind] - SolTrunc[:, Ind], \
                      self.Sol[:, Ind] - SolTrunc[:, Ind]) \
              / inner(self.Sol[:, Ind], self.Sol[:, Ind]) \
            for Ind in range(CellNMAX) ]
        s_e = log10(S_e)
        # clean 's_e' variable: get rid of -Inf values
        s_e[where(isinf(s_e))] = -6 # this value depends on k and s_0
        # or I can just convert nan to finite value
        # s_e = nan_to_num(s_e)
        # Parameters
        PolyOrder = shape(self.Sol)[0] - 1
        k = 1.5    # lenght of the activation ramp, not sure about this
        s_0 = -4    # s_0 ~ 1/(PolyOrder**4), not sure about this
        Epsilon_0 = CellSize / PolyOrder # * abs(self.A) / 10, not sure about this
        # Ranges
        Range1 = s_e < (s_0 - k)
        Range2 = [ (s_0 - k) < s_e[i] and s_e[i] < (s_0 + k) \
            for i in range(CellNMAX) ]
        Range3 = s_e > (s_0 + k)
        # Epsilon
        Epsilon_e = Epsilon_0 \
            * ( 0.0 * Range1 \
                + 0.5 * ( 1 + sin( pi*(s_e-s_0) / (2*k) ) ) * Range2 \
                + Range3 )
        return Epsilon_e


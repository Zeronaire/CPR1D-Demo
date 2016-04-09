class ConvectionEq(object):
###########################################################
# Define Convection equation
# u_t + a * u_x = 0
# a = const
# f(u) = a * u, f is flux
###########################################################
    def __init__(self, X_Mat, Sol_Mat, ConvectionA):
        self.X = X_Mat
        self.Sol = Sol_Mat
        self.A = ConvectionA
        self.Flux = ConvectionA * Sol_Mat

    def update(self, Sol_Mat):
        self.Sol = Sol_Mat
        self.Flux = self.A * Sol_Mat

    def getContinuousFlux(self, XMesh, Poly):
        from numpy import ones, abs, dot
        SolFaceL_Vec = ones((XMesh.CellNMAX+1,))
        SolFaceL_Vec[0:-1] = self.Sol[0, :]
        SolFaceR_Vec = ones((XMesh.CellNMAX+1,))
        SolFaceR_Vec[1:] = self.Sol[-1, :]
        # Set Periodic BC
        SolFaceL_Vec[-1] = SolFaceL_Vec[0]
        SolFaceR_Vec[0] = SolFaceR_Vec[-1]
        # Construct Discontinuous Flux Polynomial
        FluxDiscL_Vec = self.Flux[0, :]
        FluxDiscR_Vec = self.Flux[-1, :]
        # Compute True Flux values at the interfaces
        # Flux Difference Splitting expression
        Flux_Vec = 0.5 * ( \
            self.A * (SolFaceL_Vec + SolFaceR_Vec) \
            - abs(self.A) * (SolFaceL_Vec - SolFaceR_Vec) )
        FluxL_Vec = Flux_Vec[0:-1]
        FluxR_Vec = Flux_Vec[1:]
        FluxDeriv_Mat = dot(Poly.getLagrangePolyDeriv(), self.Flux)
        # Compute the derivative of continuous flux polynomial
        FluxContiDeriv_Mat = 1.0 / XMesh.getJacobian() * ( FluxDeriv_Mat \
            + dot(Poly.getRadauRightPoly().reshape((Poly.Order+1, 1)), \
                (FluxL_Vec - FluxDiscL_Vec).reshape((1, XMesh.CellNMAX))) \
            + dot(Poly.getRadauLeftPoly().reshape((Poly.Order+1, 1)), \
                (FluxR_Vec - FluxDiscR_Vec).reshape((1, XMesh.CellNMAX)) ) )
        return FluxContiDeriv_Mat

###########################################################
class ConvArtDiffEq(object):

    def __init__(self, X_Mat, Sol_Mat, ConvectionA, Diffusivity):
        self.X = X_Mat
        self.Sol = Sol_Mat
        self.A = ConvectionA
        self.Flux = ConvectionA * Sol_Mat
        self.E = Diffusivity

    def getSmoothIndicator(self, ):
        ut=V\u
        # u_hat: Truncated ut
        ut(P+1,:)=0
        u_hat=V*ut
        # Smooth/Resolution Indicator
        se = log10(dot(u-u_hat,u-u_hat)./dot(u,u))
        # clean 'se' variable: get rid of NaN or -Inf values
        se(se==-Inf)=-6
        # Parameters
        k = 1.5    # lenght of the activation ramp
        so = -4    # $s_0$
        epso = (h/P)*abs(a)/10
        # Ranges
        range1 = se<(so-k)
        range2 = (so-k)<se & se<(so+k)
        range3 = se>(so+k)
        # Epsilon
        epsilon = epso.*(0*range1 + 1/2*(1+sin(pi*(se-so)/(2*k))).*range2 + range3)

    def getContinuousFlux(self, XMesh, Poly):
        

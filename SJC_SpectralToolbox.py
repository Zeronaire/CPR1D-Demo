class Poly(object):

    def __init__(self, Node_Vec):
        self.Order = len(Node_Vec) - 1
        self.Nodes = Node_Vec

    def getLagrangeBasis(self):
        from numpy.polynomial.polynomial import Polynomial
        #########################
        # Construct Lagrange Basis Functions in the form of the Polynomial class
        # imported from numpy.
        # Output:
        #           List of Polynomial object. Each Polynomial object has 3 default
        #           parameters. The first is the coefficients, second is the domain,
        #           third is the window size. Details about the latter 2 parameters
        #           are in the definition of Polynomial class in numpy.
        x = self.Nodes
        PolyList_List = []
        for j in range(self.Order+1):
            Poly_Poly = 1.0
            for k in range(self.Order+1):
                if k == j:
                    continue
                Poly_Poly *= Polynomial([-x[k], 1.0]) / (x[j] - x[k])
            PolyList_List.append(Poly_Poly)
        return PolyList_List

    def getLagrangePolyDeriv(self):
        from numpy import zeros
        from numpy.polynomial.polynomial import polyval
        ###########################################################
        # Construct Lagrange Polynomial of order OrderNMAX
        NodesNMAX = self.Order + 1
        Basis_List = self.getLagrangeBasis()
        BasisDeriv_List = \
            [Basis_List[i].deriv(1) for i in range(NodesNMAX)]
        BasisDerivCoef_List = \
            [BasisDeriv_List[i].coef for i in range(NodesNMAX)]
        BasisDeriv_Mat = zeros((NodesNMAX, NodesNMAX))
        # Here raises a problem for the arrangement
        for Ind in range(NodesNMAX):
            BasisDeriv_Mat[:, Ind] = \
                polyval(self.Nodes, BasisDerivCoef_List[Ind])
        return BasisDeriv_Mat

    def getRadauRightPoly(self):
        from scipy.special import legendre
        from numpy.polynomial.polynomial import polyval
        from numpy import insert
        ###########################################################
        # Construct Right Radau Polynomial
        Temp1 = legendre(self.Order).deriv(1).coeffs
        Temp2 = legendre(self.Order+1).deriv(1).coeffs
        RadauR_Vec = (-1)**self.Order / 2.0 * (insert(Temp1, 0, 0) - Temp2)
        RadauRPoly_Vec = polyval(self.Nodes, RadauR_Vec[::-1])
        return RadauRPoly_Vec

    def getRadauLeftPoly(self):
        Temp_Vec = self.getRadauRightPoly()
        return -Temp_Vec[::-1]


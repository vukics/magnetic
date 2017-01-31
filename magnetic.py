import numpy as np
from scipy import special

import matplotlib.pyplot as plt
from matplotlib import cm



def symbolicForCurrentLoop() :
    import sympy as sp

    from sympy.functions.special.elliptic_integrals import elliptic_k, elliptic_e

    from sympy import sympify, lambdify, Eq, sqrt, diff

    z = sp.symbols('z', real=True)
    rho, R, m = sp.symbols('rho R m', positive=True)
    
    Bz=1/sqrt((R+rho)**2+z**2)*(elliptic_k(m)+(R**2-rho**2-z**2)/((R-rho)**2+z**2)*elliptic_e(m))
    Brho=z/(rho*sqrt((R+rho)**2+z**2))*(-elliptic_k(m)+(R**2+rho**2+z**2)/((R-rho)**2+z**2)*elliptic_e(m))
    
    mExpr=(4*R*rho)/((R+rho)**2+z**2)
    
    Bz=Bz.subs(m,mExpr)
    Brho=Brho.subs(m,mExpr)

    def _doLambdify(expr) :
        return lambdify((z,rho,R),expr,
                        modules=[{"elliptic_k": special.ellipk, "elliptic_e": special.ellipe},"numpy"])

    return _doLambdify(Bz), _doLambdify(Brho), _doLambdify(diff(Brho,rho)), _doLambdify(diff(Brho,z)), _doLambdify(diff(Bz,rho)), _doLambdify(diff(Bz,z))



def baseVectors(n) :
    """ Returns 3 orthognal base vectors, the first one colinear to n.
        At some point the routine could be written back to accept an array of base vectors

        Parameters
        -----------
        n: ndarray, shape (3)
            A vector giving direction of the basis

        Returns
        -----------
        n: ndarray, shape (3)
            The first vector of the basis
        l: ndarray, shape (3)
            The second vector of the basis
        m: ndarray, shape (3)
            The first vector of the basis

    """
    # normalize n
    n = n / np.linalg.norm(n)

    # choose two vectors perpendicular to n (choice is arbitrary since the solid is symetric about n)
    if  n[1]==0 and n[2]==0  : l = np.array((0, 1   , 0   ))
    else                     : l = np.array((0, n[2],-n[1]))

    l = l / np.linalg.norm(l)
    m = np.cross(n, l)
    return n, l, m



class CylindricallySymmetricSolid(object) :
    """
    Just to factor out common code from the current loop and the wire
    """
    
    def __init__(self,n,r0) :
        """
        Arguments
        ----------
            n: ndarray, shape (3, )
                The direction vector defining the axis of the cylindrical symmetry
            r0: ndarray, shape (3, )
                The location of the solid in units of d: [x y z]
        """
      
        self.r0 = r0;

        ### Translate the coordinates in the coil's frame
        n, l, m = baseVectors(n)
        
        # transformation matrix lab frame => own frame
        self.trans = np.vstack((l,m,n))
        # transformation matrix own frame => lab frame
        self.transInv = np.linalg.inv(self.trans)


    def calculateField(self,r_mg,calculateJacobian=False) :
        """
        Arguments
        ----------
            r_mg: meshgrid representing positions (in units of d) where the magnetic field is evaluated
                [[x1,x2,…],[y1,y2,…],[z1,z2,…]], shape (3,Nx,Ny,Nz)

        Returns
        --------
        B: ndarray, shape (3,Nx,Ny,Nz)
            a vector for the B field at each position specified in r in inverse units of (mu I) / (2 pi d)
            (for I in amps and d in meters and mu = 4 pi * 10^-7 we get Tesla)
        """

        r = np.transpose(r_mg,(2,1,3,0)) # shape (Nx,Ny,Nz,3)

        # point location from center of coil
        r = r - self.r0
        # transform vector to coil frame
        r = np.dot(r, self.transInv)

        #### calculate field

        # express the coordinates in polar form
        x = r[...,0]; y = r[...,1]; z = r[...,2]
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y,x)
        
        C=np.cos(phi); S=np.sin(phi); ZERO=np.zeros(phi.shape); ONE=np.ones(phi.shape)
        
        localTrans    = np.array((np.array((C, -S, ZERO)), np.array(( S, C, ZERO)), np.array((ZERO, ZERO, ONE)))) # shape (Nrow, Ncol, Nx, Ny, Nz) = (3,3,Nx,Ny,Nz)
        localTransInv = np.array((np.array((C,  S, ZERO)), np.array((-S, C, ZERO)), np.array((ZERO, ZERO, ONE))))
        
        fullTrans    = np.einsum('ij,jk...->ik...',self.transInv,localTrans)
        fullTransInv = np.einsum('ij,jk...->ik...',self.trans,localTransInv)
        
        # Rotate the field back in the lab’s frame. For this the axis representing space has to be rolled to the necessary position (and then rolled back)

        res = np.array(self.calculateFieldInOwnCylindricalCoordinates(rho,phi,z,calculateJacobian))
        field = np.einsum('ij...,j...->i...',fullTrans,res[:3])
        
        print(res.shape)

        if (calculateJacobian) :
            jacobian = res[3:].reshape((3,3)+res.shape[1:])
            print(jacobian.shape)
            jacobian = np.einsum('ij...,jk...->ik...',fullTrans,np.einsum('ij...,jk...->ik...',jacobian,fullTransInv))
            return (field,jacobian)
        else : return field



class CurrentLoop(CylindricallySymmetricSolid) :
    """
    Calculates the magnetic field of an arbitrary current loop using Eqs. (1) and (2) in Phys Rev A 35:1535-1546(1987)
    """
    
    def __init__(self,n,r0,R) :
        """
        Arguments
        ----------
            R: float
                The radius of the current loop
        """
        super(CurrentLoop,self).__init__(n,r0)

        self.R = R

        self.expressions = symbolicForCurrentLoop()


    def calculateFieldInOwnCylindricalCoordinates(self,rho,phi,z,calculateJacobian) :
        e = self.expressions
        Bz = e[0](z,rho,self.R)
        Brho = e[1](z,rho,self.R)
        # On the axis of the coil we get a division by zero here. This returns a NaN where the field is actually zero :
        Brho[np.isnan(Brho)] = 0; Brho[np.isinf(Brho)] = 0
        Bz  [np.isnan(Bz)]   = 0; Bz  [np.isinf(Bz)]   = 0

        ZERO = np.zeros(Brho.shape)

        res = (Brho, ZERO, Bz)
        return res if not calculateJacobian else res+(e[2](z,rho,self.R),ZERO,e[3](z,rho,self.R),ZERO,ZERO,ZERO,e[4](z,rho,self.R),ZERO,e[5](z,rho,self.R))



class InfiniteWire(CylindricallySymmetricSolid) :
    """
    Calculates the magnetic field of an infinite wire using Eqs. (4) and (5) in Phys Rev A 35:1535-1546(1987)
    """
    
    def __init__(self,n,r0,rhoLimit) :
        """
        Arguments
        ----------
            n: ndarray, shape (3, )
                The direction vector of the wire
            r0: ndarray, shape (3, )
                The location of the wire in units of d: [x y z]
            rhoLimit: float
                Minimal rho to calculate field for
        """
        super(InfiniteWire,self).__init__(n,r0)
        self.rhoLimit=rhoLimit


    def calculateFieldInOwnCylindricalCoordinates(self,rho,phi,z,calculateJacobian) :
        Bphi = 1./rho
        Bphi[rho<self.rhoLimit] = 0
        
        ZERO = np.zeros(Bphi.shape); res = (ZERO, Bphi, ZERO)
        
        if not calculateJacobian : return res
        else :
            dBphidrho = -1./rho**2
            dBphidrho[rho<self.rhoLimit] = 0
            return res+(ZERO, ZERO, ZERO, dBphidrho, ZERO, ZERO, ZERO, ZERO, ZERO)
            



class ArrayOfSources(object) :
    """
    Adds up the fields of its elements
    
    Recursivity is possible : an instance of ArrayOfSources can be an element of a larger-scale instance of ArrayOfSources
    """
    
    def __init__(self,arrayOfSources,relativeCurrents=None) :
        """
        Arguments
        ----------
            relativeCurrents: list of the same length as arrayOfSources
                The currents in the different objects
        """
        self.arrayOfSources = arrayOfSources
        self.relativeCurrents = np.ones(len(arrayOfSources)) if relativeCurrents==None else relativeCurrents
 
 
    def setCurrents(self,c) : self.relativeCurrents=c
    
    def calculateField(self,r_mg) :
        B = self.arrayOfSources[0].calculateField(r_mg)*self.relativeCurrents[0]
        for x, I in zip(self.arrayOfSources[1:],self.relativeCurrents[1:]) : B += x.calculateField(r_mg)*I
        return B



class Coil(ArrayOfSources) :
    """
    Coil with rectangular cross section
    
    At the moment, this is a simple array of current loops, so we completely disregard the fact that the transition matrices will be the same for all loops, that is, that Coil is itself a CylindricallySymmetricSolid.
    This is because multiple inheritance is rather tiresome in python.
    Hope this won’t cause too big an overhead…
    """
    
    def __init__(self,n,r0,R,w,nw,h,nh) :
        """
        Arguments
        ----------
            n: ndarray, shape (3, )
                The normal vector to the plane of the solid
            r0: ndarray, shape (3, )
                The location of the centre of the trap
            R: float
                Inner radius
            w: float
                The width of the rectangular cross section
            nw: float
                Number of loops along width
            h: float
                The height of the rectangular cross section
            nh: float
                Number of loops along height
        
        There are no relative currents here, as the same current will flow in all loops
        """
        super(Coil,self).__init__([CurrentLoop(n,r0+i*n,Rj) for i in np.linspace(-h/2.,h/2.,nh) for Rj in np.linspace(R,R+w,nw)])



class QuadrupoleTrap(ArrayOfSources) :
    """
    Two instances of Coil with the same orientation
    """
    
    def __init__(self,n,r0,R,w,nw,h,nh,d,relativeCurrents=None) :
        """
        Arguments
        ---------
            n, r0, R, w, nw, h, nh: same as for Coil
            d: float
                The distance of the two coils
        """
        n=np.array(n); r0=np.array(r0)
        super(QuadrupoleTrap,self).__init__([
            Coil( n,r0+d/2.*n,R,w,nw,h,nh),
            Coil(-n,r0-d/2.*n,R,w,nw,h,nh)
            ],relativeCurrents)



class IoffeWires(ArrayOfSources) :
    """
    Two instances of InfiniteWire with opposite orientation
    """
    
    def __init__(self,n,r0,m,d,rhoLimit,relativeCurrents=None) :
        """
        Arguments
        ---------
            n: ndarray, shape (3, )
                The diretion vector of the wires
            r0: ndarray, shape (3, )
                The location of the centre
            m: ndarray, shape (3, )
                Direction vector of the relative positions
            d: float
                The distance of the two wires
        """
        n=np.array(n); r0=np.array(r0); m=np.array(m)
        super(IoffeWires,self).__init__([
            InfiniteWire( n,r0+d/2.*m,rhoLimit),
            InfiniteWire(-n,r0-d/2.*m,rhoLimit)
            ],relativeCurrents)
        



def visualizeFieldMap(B,xcoord,ycoord,nLevels=40,Bmax=-1) :
    """
    Here, eventually some kwargs could be passed to enable tailoring the plot from the outside
    """
    Bnorm=np.linalg.norm(B,axis=0);
    if Bmax>0 : B[:,Bnorm>Bmax]=0; Bnorm=np.linalg.norm(B,axis=0);

    fig, axes = plt.subplots(ncols=3,nrows=2,figsize=(24,12))

    for i, title in zip(range(3),["$B_x$","$B_y$","$B_z$"]) :
        ax = axes.flatten()[i]
        temp = ax.contour(xcoord, ycoord, np.transpose(B[i]), cmap=cm.Spectral, linewidths=3);
        ax.set_title(title)
        fig.colorbar(temp, ax=ax)

    BnormAxis=axes[1,0]
    
    levels=np.linspace(0,Bnorm.max(),nLevels)

    temp=BnormAxis.contour(xcoord, ycoord, np.transpose(Bnorm), cmap=cm.Spectral, linewidths=3, levels=levels);
    BnormAxis.set_title("$|B|$")
    fig.colorbar(temp,ax=BnormAxis)

    xaxis=axes[1,1]; ymid=round(Bnorm.shape[1]/2)
    xaxis.plot(xcoord,Bnorm[:,ymid],linewidth=3,label="$|B|$")
    for i, s in zip(range(3),["x","y","z"]) : xaxis.plot(xcoord,B[i,:,ymid],label="$B_"+s+"$")
    xaxis.legend()
    
    yaxis=axes[1,2]; xmid=round(Bnorm.shape[0]/2)
    yaxis.plot(ycoord,Bnorm[xmid,:])
    for i in range(3) : yaxis.plot(ycoord,B[i,xmid,:])

    fig.tight_layout()
    plt.show()



def gradientOfNorm(B,Bnorm,Jacobian) :
    return np.einsum("i...,ij...",B,Jacobian)/Bnorm

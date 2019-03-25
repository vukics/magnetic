# Copyright (c) 2016-2017 András Vukics (http://github.com/vukics). Distributed under the MIT Licence (See accompanying file LICENSE)

import numpy as np
from scipy import special

import matplotlib.pyplot as plt
from matplotlib import cm

from math import copysign


def create3by3matrix(a00,a01,a02,a10,a11,a12,a20,a21,a22) : return np.array((np.array((a00,a01,a02)), np.array((a10,a11,a12)), np.array((a20,a21,a22))))

def matrixVector(m,v) : return np.einsum('ij...,j...->i...',m,v)

def filterIndeces(a) :
    try :
        while True :
            a=np.rollaxis(a,a.shape.index(1))[0]
    except :
        return a


def transformation(n) :
    """
        Parameters
        -----------
        n: ndarray, shape (3,)
            A vector giving direction of the basis

        Returns
        -----------
        transformation matrix lab frame => own frame / or boolean if n lies along the z direction

    """
    n = n / np.linalg.norm(n) # normalize n

    if  n[0]==0 and n[1]==0  : return copysign(1,n[2]) # n lies along the z direction

    # else choose two vectors perpendicular to n (choice is arbitrary since the solid is symetric about n)
    l = np.array((-n[1],n[0],0))

    l = l / np.linalg.norm(l)
    m = np.cross(n, l)
    return np.vstack((l,m,n))


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


currentLoopExpressions = symbolicForCurrentLoop()



class CylindricallySymmetricSolid(object) :
    """
    Just to factor out common code from the current loop and the wire
    """
    
    def __init__(self,n,r0) :
        """
        Arguments
        ----------
            n: bool or ndarray, shape (3,)
                Direction vector of the axis of the solid
                If boolean, then the axis lies along the z direction (True if direction vector points upwards)
            r0: ndarray, shape (3,)
                The location of the solid in units of d: [x y z]
        """

        
        if type(n)==bool : self.transformation=1 if n else -1
        elif n[0]==0 and n[1]==0 : self.transformation=1 if n[2]>0 else -1
        else : self.transformation=transformation(n)

        self.matrixTransformation=(not type(self.transformation)==int)
        
        if self.matrixTransformation : self.transformationInv=np.linalg.inv(self.transformation)

        self.r0 = r0;


    def calculateField(self,x,y,z) :
        """
        Arguments
        ----------
            x, y, z: coordinates that can be either floats or array_like types

        Returns
        --------
        B: ndarray, shape (3,Nx,Ny,Nz)
            a vector for the B field at each position specified in r in inverse units of (mu I) / (2 pi d)
            (for I in amps and d in meters and mu = 4 pi * 10^-7 we get Tesla)
        """
        
        r = np.meshgrid(*[ (xx if type(xx).__module__ == np.__name__ else (float(xx)) ) for xx in [x,y,z] ],indexing='ij')

        # point location from center of coil
        for i in range(3) : r[i]-=self.r0[i]
        
        # transform vector to coil frame
        if self.matrixTransformation : r=matrixVector(self.transformation,r)

        x = r[0]; y = r[1]; z = r[2]
        
        # express the coordinates in polar form
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y,x)
        
        C=np.cos(phi); S=np.sin(phi); ZERO=np.zeros(phi.shape); ONE=np.ones(phi.shape)
        
        ### self.direction (which should be now either a +/- 1, or a matrix), should be fused into localTrans
        localTrans = create3by3matrix(C, -S, ZERO,  S, C, ZERO, ZERO, ZERO, ONE) # shape (Nrow, Ncol, Nx, Ny, Nz) = (3,3,Nx,Ny,Nz)
        fullTrans = np.einsum('ij,jk...->ik...',self.transformationInv,localTrans) if self.matrixTransformation else localTrans*self.transformation

        # Rotate the field back in the lab’s frame.

        res = np.array(self.calculateFieldInOwnCylindricalCoordinates(rho,phi,z))
        
        return filterIndeces(matrixVector(fullTrans,res))



class CurrentLoop(CylindricallySymmetricSolid) :
    """
    Calculates the magnetic field of an arbitrary current loop using Eqs. (1) and (2) in Phys Rev A 35:1535-1546(1987)
    """
    
    def __init__(self,n,r0,R) :
        """
        Arguments
        ----------
            n: same as with CylindricallySymmetricSolid
            r0: ”
            R: float
                The radius of the current loop
        """
        super(CurrentLoop,self).__init__(n,r0)

        self.R = R


    def calculateFieldInOwnCylindricalCoordinates(self,rho,phi,z) :
        Bz = currentLoopExpressions[0](z,rho,self.R)
        Brho = currentLoopExpressions[1](z,rho,self.R)
        # On the axis of the coil we get a division by zero here. This returns a NaN where the field is actually zero :
        Brho[np.isnan(Brho)] = 0; Brho[np.isinf(Brho)] = 0
        Bz  [np.isnan(Bz)]   = 0; Bz  [np.isinf(Bz)]   = 0

        return (Brho, np.zeros(Brho.shape), Bz)


    def thermalPowerCoeff(self,r) :
        """
        L/A : To be multiplied by the resistivity and I**2 to get the power
        """
        return 2*self.R/r**2



class PointLikeDipole(CylindricallySymmetricSolid) :
    """
    Calculates the magnetic field of a point-like dipole
    using Eq. (5.41) of Jackson: Classical Electrodynamics. Wiley, New York, NY, 2nd ed., (1975)
    """
    
    def __init__(self,n,r0,m=1) :
        """
        Arguments
        ----------
            n: same as with CylindricallySymmetricSolid
            r0: ”
            m: float
                The dipole moment
        """
        super(PointLikeDipole,self).__init__(n,r0)
        
        self.m = m
    
    
    def calculateFieldInOwnCylindricalCoordinates(self,rho,phi,z) :
        distanceToTheFifth=(rho**2+z**2)**2.5
        Brho=3*self.m*rho*z/distanceToTheFifth/2
        Bz=self.m*(2*z**2-rho**2)/distanceToTheFifth/2
        Brho[np.isnan(Brho)] = 0; Brho[np.isinf(Brho)] = 0
        Bz  [np.isnan(Bz)]   = 0; Bz  [np.isinf(Bz)]   = 0
        
        return (Brho, np.zeros(Brho.shape), Bz)



class InfiniteWire(CylindricallySymmetricSolid) :
    """
    Calculates the magnetic field of an infinite wire using Eqs. (4) and (5) in Phys Rev A 35:1535-1546(1987)
    """
    
    def __init__(self,n,r0,rhoLimit) :
        """
        Arguments
        ----------
            n: same as with CylindricallySymmetricSolid
            r0: ”
            rhoLimit: float
                Minimal rho to calculate field for
        """
        super(InfiniteWire,self).__init__(n,[r0[0],r0[1],0])
        self.rhoLimit=rhoLimit


    def calculateFieldInOwnCylindricalCoordinates(self,rho,phi,z) :
        Bphi = 1./rho
        Bphi[rho<self.rhoLimit] = 0
        
        ZERO = np.zeros(Bphi.shape)
        
        return (ZERO, Bphi, ZERO)


    def thermalPowerCoeff(self,r) :
        raise NotImplementedError



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
 
 
    def setCurrents(self,*c) : self.relativeCurrents[:len(c)]=c


    def calculateField(self,x,y,z) :
        res = 0. if self.relativeCurrents[0]==0 else self.arrayOfSources[0].calculateField(x,y,z)*self.relativeCurrents[0]
        for source, I in zip(self.arrayOfSources[1:],self.relativeCurrents[1:]) :
            if not I==0 : res += source.calculateField(x,y,z)*I
        return res


    def thermalPowerCoeff(self,r) :
        res = self.arrayOfSources[0].thermalPowerCoeff(r)*self.relativeCurrents[0]**2
        for source, I in zip(self.arrayOfSources[1:],self.relativeCurrents[1:]) :
            res+=source.thermalPowerCoeff(r)*I**2
        return res



def visualizeFieldMap(B,xcoord,ycoord,nLevels=40,Bmax=-1) :
    """
    Here, eventually some kwargs could be passed to enable tailoring the plot from the outside
    """
    Bnorm=np.linalg.norm(B,axis=0)
    if Bmax>0 : B[:,Bnorm>Bmax]=0

    fig, axes = plt.subplots(ncols=3,nrows=2,figsize=(24,12))

    for i, title in zip(range(3),["$B_x$","$B_y$","$B_z$"]) :
        ax = axes.flatten()[i]
        try :
            temp = ax.contour(xcoord, ycoord, np.transpose(B[i]), cmap=cm.Spectral, linewidths=3);
            ax.set_title(title)
            fig.colorbar(temp, ax=ax)
        except ValueError as e :
            print("Axis "+title+":",e)

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
    yaxis.plot(ycoord,Bnorm[xmid,:],linewidth=3)
    for i in range(3) : yaxis.plot(ycoord,B[i,xmid,:])

    fig.tight_layout()
    plt.show()



def gradientOfNorm(fieldDeriv) :
    field=fieldDeriv[0]
    return matrixVector(fieldDeriv[1],field)/np.linalg.norm(field,axis=0)


# Copyright (c) 2016-2017 András Vukics (http://github.com/vukics). Distributed under the MIT Licence (See accompanying file LICENSE)

import magneticElements

import numpy as np

import math


def nz(n) : return np.array((0, 0, 1 if n else -1)) if type(n)==bool else n

def inverseOf(n) : return not n if type(n)==bool else -1*n

ArrayOfSources=magneticElements.ArrayOfSources

visualizeFieldMap=magneticElements.visualizeFieldMap


class Coil(ArrayOfSources) :
    """
    Coil with rectangular cross section
    
    Here, this is a simple array of current loops, so we completely disregard the fact that the transition matrices will be the same for all loops, that is, that Coil is itself a CylindricallySymmetricSolid.
    This is because multiple inheritance is rather tiresome in python.
    Hope this won’t cause too big an overhead…
    """
    
    def __init__(self,n,r0,R,w,nw,h,nh) :
        """
        Arguments
        ----------
            n: bool or ndarray, shape (3,)
                Direction vector of the axis of the solid
                If boolean, then the axis lies along the z direction (True if direction vector points upwards)
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
        super(Coil,self).__init__([magneticElements.CurrentLoop(n,r0+i*nz(n),Rj) for i in np.linspace(-h/2.,h/2.,nh) for Rj in np.linspace(R,R+w,nw)])


class PackedCoil(ArrayOfSources) :
    """
    cf. PackedCoilModel.pdf on how this is derived from (two intertwined) Coil(s)
    """
    
    def __init__(self,n,r0,R,w,nw,h,nh,Delta,moreOrLess) :
        super(PackedCoil,self).__init__([
            Coil(n,r0,R,w,nw,h,nh),
            Coil(n,r0,R+math.sqrt(3.)/2.*Delta,w-sqrt(3.)*Delta if moreOrLess else w,nw-math.copysign(1,Delta) if moreOrLess else nw,h-2.*abs(Delta),nh-1)
        ])


class TwoCoils(ArrayOfSources) :
    
    def __init__(self,n1,n2,r0,R,w,nw,h,nh,d,relativeCurrents=None,WhichCoil=Coil,**kwargs) :
        """
        Arguments
        ---------
            n1, n2 : the axis orientation in Coil 1 and 2, respectively
            r0, R, w, nw, h, nh: same as for Coil
            d: float
                The distance of the two coils
        """
        r0=np.array(r0)
        super(TwoCoils,self).__init__([
            WhichCoil(n1,r0+d/2.*nz(n1),R,w,nw,h,nh,**kwargs),
            WhichCoil(n2,r0-d/2.*nz(n1),R,w,nw,h,nh,**kwargs)
            ],relativeCurrents)
        
        self.d = d

    
    
class DipoleCoils(TwoCoils) :
    """
    Two instances of Coil with the same axis and same current directions
    """
    
    def __init__(self,n,r0,R,w,nw,h,nh,d,relativeCurrents=None,WhichCoil=Coil,**kwargs) :
        """
        Arguments
        ---------
            n, r0, R, w, nw, h, nh: same as for Coil
            d: float
                The distance of the two coils
        """
        super(DipoleCoils,self).__init__(n,n,r0,R,w,nw,h,nh,d,relativeCurrents,WhichCoil,**kwargs)



class QuadrupoleTrap(TwoCoils) :
    """
    Two instances of Coil with the same axis but opposite current directions
    """
    
    def __init__(self,n,r0,R,w,nw,h,nh,d,relativeCurrents=None,WhichCoil=Coil,**kwargs) :
        """
        Arguments
        ---------
            n, r0, R, w, nw, h, nh: same as for Coil
            d: float
                The distance of the two coils
        """
        super(QuadrupoleTrap,self).__init__(n,inverseOf(n),r0,R,w,nw,h,nh,d,relativeCurrents,WhichCoil,**kwargs)



class IoffeWires(ArrayOfSources) :
    """
    Two instances of InfiniteWire with opposite orientation
    """
    
    def __init__(self,n,r0,m,d,rhoLimit,relativeCurrents=None) :
        """
        Arguments
        ---------
            n: bool or ndarray, shape (3,)
                Direction vector of the wire
                If boolean, then it lies along the z direction (True if direction vector points upwards)
            r0: ndarray, shape (3, )
                The location of the centre
            m: ndarray, shape (3, )
                Direction vector of the relative positions
            d: float
                The distance of the two wires
        """
        r0=np.array(r0); m=np.array(m)
        super(IoffeWires,self).__init__([
            magneticElements.InfiniteWire(          n ,r0+d/2.*m,rhoLimit),
            magneticElements.InfiniteWire(inverseOf(n),r0-d/2.*m,rhoLimit)
            ],relativeCurrents)



class DipoleSimplex(ArrayOfSources) :
    """
    From two loops
    """
    def __init__(self,n,r0,R,d,relativeCurrents=None) :
        """
        Arguments
        ---------
            n, r0, R: same as for CurrentLoop
            d: float
                The distance of the two loops
        """
        r0=np.array(r0)
        super(DipoleSimplex,self).__init__([
            magneticElements.CurrentLoop(n,r0+d/2.*nz(n),R),
            magneticElements.CurrentLoop(n,r0-d/2.*nz(n),R)
            ],relativeCurrents)
        
        self.d = d


class QuadrupoleSimplex(ArrayOfSources) :
    """
    From two loops mostly for testing
    """
    def __init__(self,n,r0,R,d,relativeCurrents=None) :
        """
        Arguments
        ---------
            n, r0, R: same as for CurrentLoop
            d: float
                The distance of the two loops
        """
        r0=np.array(r0)
        super(QuadrupoleSimplex,self).__init__([
            magneticElements.CurrentLoop(          n ,r0+d/2.*nz(n),R),
            magneticElements.CurrentLoop(inverseOf(n),r0-d/2.*nz(n),R)
            ],relativeCurrents)
        
        self.d = d
    
    
    def coefficient(self) :
        A = self.d/2
        R = self.arrayOfSources[0].R
        return 3*math.pi*A*R**2/(R**2+A**2)**2.5, 5*math.pi*(4*A**2-3*R**2)/(6*(A**2+R**2)**2)



class HelmholtzCoil(ArrayOfSources) :
    def __init__(self,n,r0,R) :
        r0=np.array(r0)
        super(HelmholtzCoil,self).__init__([
            magneticElements.CurrentLoop(n,r0+R/2.*nz(n),R),
            magneticElements.CurrentLoop(n,r0-R/2.*nz(n),R)
            ])



class InfiniteWireWithSquareCrossSection(ArrayOfSources) :
    """
    Implemented as an array of 5 point-like wires
    """
    def __init__(self,n,r0,direction,rhoLimit) :
        """
        Arguments
        ----------
            n: bool or ndarray, shape (3,)
                Direction vector of the wire
                If boolean, then it lies along the z direction (True if direction vector points upwards)
            r0: ndarray, shape (2, )
                The location of the centre of the square in units of d: [x y]
            direction: ndarray, shape (2, )
                The location vector of any of the sides of the square from the centre in units of d: [x y]
            rhoLimit: float
                Minimal rho to calculate field for
        """
        r0=r0[:2]
        direction=direction[:2]; directionRot=np.array((-direction[1],direction[0]))
        diagonal=direction+directionRot; diagonalRot=direction-directionRot
        super(InfiniteWireWithSquareCrossSection,self).__init__([magneticElements.InfiniteWire(n,rIter,rhoLimit) for rIter in [r0,r0+diagonal,r0-diagonal,r0+diagonalRot,r0-diagonalRot]],
                                                                0.2*np.ones(5)
                                                                )
        self.rhoLimit=rhoLimit

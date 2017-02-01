import magnetic as mg
import numpy as np

from scipy.optimize import minimize
from scipy.misc import derivative

import matplotlib.pyplot as plt



# Everything is in mm and A, so the field is to be multiplied by 2 to get Gauss
def actualSetup() :
    MOT_innerRadius=17; MOT_outerRadius=32
    origo=np.zeros(3); otherCenter=np.array(origo); otherCenter[0]=MOT_outerRadius

    MOT=mg.QuadrupoleTrap(True,origo,MOT_innerRadius,MOT_outerRadius-MOT_innerRadius,13,10,12,42)
    transferCoil=mg.QuadrupoleTrap(True,otherCenter,13,11,7,6,7,24)
    compressionCoil=mg.QuadrupoleTrap(True,otherCenter,6,6,5,10,9,18)
    wires=mg.ArrayOfSources([mg.IoffeWires(True,otherCenter+np.array([3,0,0]),[0,1,0],1.6,.025),
                             mg.IoffeWires(True,otherCenter+np.array([3,0,0]),[0,1,0],4.6,.025)
                            ])

    return mg.ArrayOfSources([MOT,transferCoil,compressionCoil,wires],[5,0,0,12])


def setupWith6Wires() :
    MOT_innerRadius=17; MOT_outerRadius=32
    origo=np.zeros(3); otherCenter=np.array(origo); otherCenter[0]=MOT_outerRadius

    res = actualSetup()
    wires=mg.ArrayOfSources([mg.IoffeWires(True,otherCenter+np.array([3,0,0]),[0,1,0],1.6,.025),
                             mg.IoffeWires(True,otherCenter+np.array([3,0,0]),[0,1,0],4.6,.025),
                             mg.IoffeWires(True,otherCenter+np.array([3,0,0]),[0,1,0],7.6,.025)
                            ])
    res.arrayOfSources[3]=wires
    return res



Setup=actualSetup()

mu=.0671 # mK/G
m_g=.10175   # mK/mm. This is the average isotope mass (87AMU)

rhoCu=1.68e-5 # Ohm*mm

Bnorm = lambda x, y, z : np.linalg.norm(2*Setup.calculateField(x,y,z),axis=0)
potentialEnergy = lambda x, y, z : mu*Bnorm(x,y,z)+m_g*z

def potentialEnergyGradient(x,y,z) :
    gradientOfNorm=2*mg.gradientOfNorm(Setup.calculateField(x,y,z,True))
    gradientOfNorm[2]+=m_g
    return gradientOfNorm


def derivatives(f,x0,dx0,epsDeriv=1e-3) :
    return (abs(derivative(f,x0+dx0,dx=epsDeriv*abs(dx0))),
            derivative(f,x0+dx0,dx=epsDeriv*abs(dx0),n=2,order=5),
            derivative(f,x0+dx0,dx=epsDeriv*abs(dx0),n=3,order=5)
            )



# This is with gravitational potential energy
def quadrupoleCharacteristicsWithGravity(x0,z0,offset=1, # offset from the minimum when looking for the maximum
                                         epsDeriv=1e-3,stopIfMinimumNotFound=False) :
    minimum = minimize(lambda x : potentialEnergy(x[0],0,x[1]), (x0,z0))
    
    if stopIfMinimumNotFound :
        if not minimum.success : return minimum
    
    minx, minz = minimum.x
    fieldAtMinimum = Bnorm(minx,0,minz)
    maxxr = minimize(lambda x : -potentialEnergy(x,0,minz), minx+offset)
    maxxl = minimize(lambda x : -potentialEnergy(x,0,minz), minx-offset)
    maxy = minimize(lambda y : -potentialEnergy(minx,y,minz), offset)
    maxzu = minimize(lambda z : -potentialEnergy(minx,0,z), minz+offset)
    maxzl = minimize(lambda z : -potentialEnergy(minx,0,z), minz-offset)
    
    dx0r=epsDeriv*abs(maxxr.x[0]-minx)
    dx0l=epsDeriv*abs(maxxl.x[0]-minx)

    dy0=epsDeriv*abs(maxy.x[0])

    dz0u=epsDeriv*abs(maxzu.x[0]-minz)
    dz0l=epsDeriv*abs(maxzl.x[0]-minz)

    return ((minx, minz, fieldAtMinimum, minimum.fun),
            (maxxr.x[0], -maxxr.fun-minimum.fun),
            (maxxl.x[0], -maxxl.fun-minimum.fun),
            (maxy.x[0], -maxy.fun-minimum.fun),
            (maxzu.x[0], -maxzu.fun-minimum.fun),
            (maxzl.x[0], -maxzl.fun-minimum.fun),
            derivatives(lambda x : potentialEnergy(x,0,minz), minx,  dx0r),
            derivatives(lambda x : potentialEnergy(x,0,minz), minx, -dx0l),
            derivatives(lambda y : potentialEnergy(minx,y,minz), 0, dy0),
            derivatives(lambda z : potentialEnergy(minx,0,z), minz,  dz0u),
            derivatives(lambda z : potentialEnergy(minx,0,z), minz, -dz0l)
           )


def visualizeSequenceWithGravity(sequence,currents) :
    iTrans=np.arange(len(sequence))
    fig, axes = plt.subplots(ncols=3,nrows=2,figsize=(24,12))
    
    def plotLine(i,j,m,n,*args) : axes[i][j].plot(iTrans,[s[m][n] for s in sequence], *args)
        
    plotLine(0,0,0,0)
    plotLine(0,1,0,1)
    plotLine(0,2,0,2)
    for i, s in zip(range(6,11),['b','b:','g','r','r:']) : plotLine(1,0,i,0,s)
    for i, s in zip(range(6,11),['b','b:','g','r','r:']) : plotLine(1,1,i,1,s)
    for i, s in zip(range(1,6),['b','b:','g','r','r:']) : plotLine(1,2,i,1,s)

    for axis in axes.flatten()  :
        axisy2=axis.twinx()
        axisy2.plot(iTrans,currents[1],'0.7')
        axisy2.plot(iTrans,currents[0],color='0.7',linestyle=":")
        axisy2.plot(iTrans,currents[2],color='0.7',linestyle="-.")

    fig.tight_layout()
    plt.show()


def visualizeSequenceWithoutGravity(sequence,currents) :
    iTrans=np.arange(len(sequence))
    fig, axes = plt.subplots(ncols=3,nrows=2,figsize=(24,12))
    axes[0][0].plot(iTrans,[s[0][0] for s in sequence])
    axes[0][1].plot(iTrans,[s[5][0] for s in sequence], 'b')
    axes[0][1].plot(iTrans,[s[6][0] for s in sequence], 'b:')
    axes[0][1].plot(iTrans,[s[7][0] for s in sequence], 'g')
    axes[0][1].plot(iTrans,[s[8][0] for s in sequence], 'r')
    axes[0][2].plot(iTrans,[s[5][1] for s in sequence], 'b')
    axes[0][2].plot(iTrans,[s[6][1] for s in sequence], 'b:')
    axes[0][2].plot(iTrans,[s[7][1] for s in sequence], 'g')
    axes[0][2].plot(iTrans,[s[8][1] for s in sequence], 'r')
    axes[1][0].plot(iTrans,[s[0][1] for s in sequence])
    axes[1][1].plot(iTrans,[s[1][1]-s[0][1] for s in sequence], 'b')
    axes[1][1].plot(iTrans,[s[2][1]-s[0][1] for s in sequence], 'b:')
    axes[1][1].plot(iTrans,[s[3][1]-s[0][1] for s in sequence], 'g')
    axes[1][1].plot(iTrans,[s[4][1]-s[0][1] for s in sequence], 'r')
    for axis in axes.flatten()  :
        axisy2=axis.twinx()
        axisy2.plot(iTrans,currents[1],'0.7')
        axisy2.plot(iTrans,currents[0],color='0.7',linestyle=":")
        axisy2.plot(iTrans,currents[2],color='0.7',linestyle="-.")

    fig.tight_layout()
    plt.show()


def frequencyFromSecondDerivative(sd) : #  sd given in mK/mm^2
    return (9.555e4*sd)**.5/6.28



# This is without gravitational potential energy
def quadrupoleCharacteristicsWithoutGravity(x0,offset=1, # offset from the minimum when looking for the maximum
                                            epsDeriv=1e-3) :
    minx=minimize(lambda x : Bnorm(x,0,0), x0)
    minxx=minx.x[0]
    maxxr=minimize(lambda x : -Bnorm(x,0,0), minxx+offset)
    maxxl=minimize(lambda x : -Bnorm(x,0,0), minxx-offset)
    maxy=minimize(lambda y : -Bnorm(minxx,y,0), offset)
    maxz=minimize(lambda z : -Bnorm(minxx,0,z), offset)
    
    dx0r=epsDeriv*abs(maxxr.x[0]-minxx)
    dx0l=epsDeriv*abs(maxxl.x[0]-minxx)

    dy0=epsDeriv*abs(maxy.x[0])
    dz0=epsDeriv*abs(maxz.x[0])

    return ((minxx, minx.fun),
            (maxxr.x[0], -maxxr.fun),
            (maxxl.x[0], -maxxl.fun),
            (maxy.x[0], -maxy.fun),
            (maxz.x[0], -maxz.fun),
            derivatives(lambda x : Bnorm(x,0,0), minxx,  dx0r),
            derivatives(lambda x : Bnorm(x,0,0), minxx, -dx0l),
            derivatives(lambda y : Bnorm(minxx,y,0), 0, dy0),
            derivatives(lambda z : Bnorm(minxx,0,z), 0, dz0)
           )

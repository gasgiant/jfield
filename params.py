import numpy as _np
from collections import namedtuple

def _init_intfield_params():
    coeffs = _np.loadtxt('JRM09_coefficients.txt')
    k = coeffs.shape[0] // 2 
    g = coeffs[:k,:]
    h = coeffs[k:,:]
    a = 71492
    global internal_field
    internal_field = InternalFieldParams(g, h, a)

def _get_schmidt_norm_coeffs(K):
    from scipy.special import factorial
    coeffs = _np.ones((K, K + 1))
    for n in range(1, K + 1):
        for m in range(n + 1):
            if m == 0:
                coeffs[n-1,m] = 1
            else:
                coeffs[n-1,m] =  (-1) ** m * _np.sqrt( 2 * factorial(n - m) / factorial(n + m))
    return coeffs

def _calcualteAdditionalIntfieldParams():
        dipole_tilt = _np.arccos(internal_field.g[0,0] / 
            _np.sqrt(internal_field.g[0,0] ** 2 + internal_field.g[0,1] ** 2 + internal_field.h[0,1] ** 2))
        lat = dipole_tilt
        longt = _np.arctan(internal_field.h[0,1] / internal_field.g[0,1]) + _np.pi
        Zrot = _np.array([[_np.cos(longt), -_np.sin(longt), 0], [_np.sin(longt), _np.cos(longt), 0], [0, 0, 1]])
        Yrot = _np.array([[_np.cos(lat), 0, _np.sin(lat)], [0, 1, 0], [-_np.sin(lat), 0, _np.cos(lat)]])
        M2SIII_mat = _np.dot(Zrot, Yrot)
        SIII2M_mat = _np.transpose(M2SIII_mat)
        return (dipole_tilt, M2SIII_mat, SIII2M_mat, )


InternalFieldParams = namedtuple('InternalFieldParams' , 'g h a')
PCDParams = namedtuple('PCDParams' , 'I0 D Rin Rout R1 R2')
CurvatureParams = namedtuple('CurvatureParams' , 'omega_j nu0 rho0 x0')

PCD = PCDParams(23.6, 2.5, 7.1, 95, 27.3, 50)
curvature = CurvatureParams(0.63307163, 37.4, 33.2, 33.5)
internal_field = InternalFieldParams(0,0,0)

_init_intfield_params()
dipole_tilt, M2SIII_mat, SIII2M_mat = _calcualteAdditionalIntfieldParams()
schmidt_norm_coeffs = _get_schmidt_norm_coeffs(25)

RJ = 71492

def set_internal_field_params(g=None, h=None, a=None):
    '''Set parameters of the internal field of the planet. 

    For more info see e.g. Connerney et. al. 2018 (https://doi.org/10.1002/2018GL077312)

    Parameters
    ----------
    g : ndarray, optional 
        Schmidt coefficients (nT) of spherical harmonics for cos.
    h : ndarray, optional 
        Schmidt coefficients (nT) of spherical harmonics for sin.
    h : float, optional 
        Planetary radius (km) used for coefficients

    '''
    global internal_field
    if g is None:
        g = internal_field.g
    if h is None:
        h = internal_field.h
    if a is None:
        a = internal_field.a
    internal_field = InternalFieldParams(g,h,a)
    global dipole_tilt
    global M2SIII_mat  
    global SIII2M_mat
    dipole_tilt, M2SIII_mat, SIII2M_mat = _calcualteAdditionalIntfieldParams()


def set_curvature_params(omega_j=None, nu0=None, rho0=None, x0=None):
    '''Set parameters of the magnetodisk hinge. 

    For more info see Khurana (1997) https://doi.org/10.1029/97JA00563 

    Parameters
    ----------
    omega_j : float, optional 
        Angular velocity of the planet (rad / h)
    nu_0 : float, optional 
        Asymptotic value of the wave velocity (RJ / h)
    pho_0 : float, optional 
        The effective distance (RJ) beyond which the propagation delay is significant
    x_0 : float, optional 
        Characteristic distance (RJ)

    '''
    global curvature
    if omega_j is None:
        omega_j = curvature.omega_j
    if nu0 is None:
        nu0 = curvature.nu0
    if rho0 is None:
        rho0 = curvature.rho0
    if x0 is None:
        x0 = curvature.x0
    curvature = CurvatureParams(omega_j, nu0, rho0, x0)


def set_pcd_params(I0=None, D=None, Rin=None, Rout=None, R1=None, R2=None):
    '''Set parameters of the PCD model. 

    Parameters
    ----------
    I0 : float, optional 
        Current density constant (1e6 * A / Jupiter radii).
    D : float, optional 
        Half thickness of the current sheet (Jupiter radii). 
    Rin : float, optional 
        Inner radius of the current disk (Jupiter radii).
    Rout : float, optional 
        Outer radius of the current disk (Jupiter radii).
    R1 : float, optional 
        Distance of the 1/rho to 1/rho^2 law change (Jupiter radii).
    R2 : float, optional 
        Distance of the 1/rho^2 back to 1/rho law change (Jupiter radii).

    '''
    global PCD
    if I0 is None:
        I0 = PCD.I0
    if D is None:
        D = PCD.D
    if Rin is None:
        Rin = PCD.Rin
    if Rout is None:
        Rout = PCD.Rout
    if R1 is None:
        R1 = PCD.R1
    if R2 is None:
        R2 = PCD.R2
    PCD = PCDParams(I0, D, Rin, Rout, R1, R2)
    












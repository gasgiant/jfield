import numpy as _np

import internal_field as internal_field_cython
import current_disk as current_disk_cython

import params_holder as params
import transform

_ph = params


def internal_field(r, theta, phi):
    '''Calculates internal magnetic field of the Jupiter.

    Parameters
    ----------
    r : 1-D array
        Spherical coordinates r (Jupiter radii) of the sample points in right-handed System III (1965)
    theta : 1-D array
        Spherical coordinates theta (rad) of the sample points in right-handed System III (1965)
    phi : 1-D array
        Spherical coordinates phi (rad) of the sample points in right-handed System III (1965)

    Returns
    -------
    ndarray
        Magnetic field in sample points in nT in spherical right-handed System III (1965)

    '''
    n = _ph.internal_field.g.shape[0]
    return internal_field_cython.internal_field_1darray(
        r * (params.RJ / _ph.internal_field.a), theta, phi, _ph.internal_field.g, _ph.internal_field.h, _ph.schmidt_norm_coeffs[:n, :n + 1])


def PCD_field(rho, phi, z, x):
    '''Calculates magnetic field of the Jupiter's current disk according to the PCD model.

    Note: 
        Magnetic equatorial cylindrical coordinate system: rho and phi in the magnetic equator, z parallel to the dipole axis.

    Parameters
    ----------
    rho : 1-D array
        Cylindrical coordinates rho (Jupiter radii) of the sample points in magnetic equatorial coordinate system.
    phi : 1-D array
        Cylindrical coordinates phi (rad) of the sample points in magnetic equatorial coordinate system.
    z : 1-D array
        Cylindrical coordinates z (Jupiter radii) of the sample points in magnetic equatorial coordinate system.
    x : 1-D array
        Coordinates x (Jupiter radii) of the sample points along the Jupiter-Sun axis.

    Returns
    -------
    ndarray
        Magnetic field in sample points in nT in cylindrical magnetic equatorial coordinate system

    '''
    res = current_disk_cython.PCD_field_1darray(
        rho, z, _ph.PCD.I0, _ph.PCD.D, _ph.PCD.Rin, _ph.PCD.Rout, _ph.PCD.R1, _ph.PCD.R2, _z_cs(rho, phi, x), _dzcs_drho(rho, phi, x))
    return _np.array([res[0], _np.zeros(res.shape[1]), res[1]])


def SIII2M(vec):
    '''Converts vectors from System III (1965) to magnetic equatorial system
        
    Note: vectors should be in cartesian coordinates
    '''
    return _np.dot(_ph.SIII2M_mat, vec)


def M2SIII(vec):
    '''Converts vectors from magnetic equatorial system to System III (1965)
    
    Note: vectors should be in cartesian coordinates
    '''
    return _np.dot(_ph.M2SIII_mat, vec)


def _dzcs_drho(rho, phi, x):
    tan = _np.tan(_ph.dipole_tilt)
    return (tan * (_np.cosh(x / _ph.curvature.x0)) ** (-2) * _np.cos(phi - _delta(rho))
            - rho * tan * _ph.curvature.x0 / x * _np.tanh(x / _ph.curvature.x0) * _np.sin(phi - _delta(
                rho)) * _ph.curvature.omega_j / _ph.curvature.nu0 * _np.tanh(rho / _ph.curvature.rho0)
            - tan * _np.cos(phi - _delta(rho)))


def _z_cs(rho, phi, x):
    return rho * _np.tan(_ph.dipole_tilt) * (_ph.curvature.x0 / x * _np.tanh(x / _ph.curvature.x0) * _np.cos(phi - _delta(rho))
                                             - _np.cos(phi - _np.pi))


def _delta(rho):
    return _np.pi - _ph.curvature.omega_j * _ph.curvature.rho0 / _ph.curvature.nu0 * _np.log(_np.cosh(rho / _ph.curvature.rho0))


def lines(starts, ds, N, B_func):
    positions = _np.zeros((3, N, starts.shape[1]))
    positions[:, 0, :] = starts
    end_ind = _np.zeros(starts.shape[1], dtype=_np.int)
    end_mask = _np.arange(starts.shape[1], dtype=_np.int)
    for i in range(1, N):
        B = B_func(positions[:, i-1, end_mask])
        positions[:, i, end_mask] = positions[:, i-1, end_mask] + \
            ds[end_mask] * B / _np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)
        mag = _np.sqrt(positions[0, i, :] ** 2 +
                       positions[1, i, :] ** 2 + positions[2, i, :] ** 2)
        end_ind[end_mask] = i
        end_mask = _np.where(mag > 1)[0]
        if (end_mask.size == 0):
            break

    return (positions, end_ind)


def _totalB(M_cart):
    sun_long = 0
    SIII_cart = M2SIII(M_cart)
    SIII_sph = transform.coords_cart2sph(SIII_cart)
    M_cyl = transform.coords_cart2cyl(M_cart)
    x_jso = M_cyl[0] * _np.tan(_ph.dipole_tilt) * _np.cos(M_cyl[2] - sun_long)
    B_int_M_cart = SIII2M(transform.vectors_sph2cart(
        internal_field(SIII_sph[0], SIII_sph[1], SIII_sph[2]), SIII_sph))
    B_md_M_cart = transform.vectors_cyl2cart(
        PCD_field(M_cyl[0], M_cyl[1], M_cyl[2], x_jso), M_cyl)
    return B_int_M_cart + B_md_M_cart

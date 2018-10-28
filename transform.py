import numpy as _np

# Transformations of coordinates

def coords_cart2cyl(cart):
    ''' Convert cartesian coordinates to cylindrical'''
    x = cart[0]
    y = cart[1]
    z = cart[2]
    scalars = False
    if not isinstance(x, _np.ndarray):
        if isinstance(x, list):
            x = _np.array(x)
            y = _np.array(y)
            z = _np.array(z)
        else:
            x = _np.array([x])
            y = _np.array([y])
            z = _np.array([z])
            scalars = True          
    rho = _np.sqrt(x ** 2 + y ** 2)
    x_neg = _np.where(x < 0)
    x_pos = _np.where(x >= 0)
    phi = _np.zeros(rho.shape)
    phi[x_pos] = _np.arctan(y[x_pos] / x[x_pos])
    phi[x_neg] = _np.pi + _np.arctan(y[x_neg] / x[x_neg])
    if (scalars):
        return _np.array([rho, phi, z])[:,0]
    else:
        return _np.array([rho, phi, z])


def coords_cart2sph(cart):
    ''' Convert cartesian coordinates to spherical'''
    x = cart[0]
    y = cart[1]
    z = cart[2]
    scalars = False
    if not isinstance(x, _np.ndarray):
        if isinstance(x, list):
            x = _np.array(x)
            y = _np.array(y)
            z = _np.array(z)
        else:
            x = _np.array([x])
            y = _np.array([y])
            z = _np.array([z])
            scalars = True 
    r = _np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = _np.arccos(z / r)
    x_neg = _np.where(x < 0)
    x_pos = _np.where(x >= 0)
    phi = _np.zeros(r.shape)
    phi[x_pos] = _np.arctan(y[x_pos] / x[x_pos])
    phi[x_neg] = _np.pi + _np.arctan(y[x_neg] / x[x_neg])
    if (scalars):
        return _np.array([r, theta, phi])[:,0]
    else:
        return _np.array([r, theta, phi])


def coords_cyl2cart(cyl):
    ''' Convert cylindrical coordinates to cartesian'''
    rho = cyl[0]
    phi = cyl[1]
    z = cyl[2]

    x = rho * _np.cos(phi)
    y = rho * _np.sin(phi)
    return _np.array([x, y, z])


def coords_cyl2sph(cyl):
    ''' Convert cylindrical coordinates to spherical'''
    rho = cyl[0]
    phi = cyl[1]
    z = cyl[2]

    r = _np.sqrt(rho ** 2 + z ** 2)
    theta = _np.arccos(z / r)
    return _np.transpose(_np.array([r, theta, phi]))


def coords_sph2cart(sph):
    ''' Convert spherical coordinates to cartesian'''
    r = sph[0]
    theta = sph[1]
    phi = sph[2]

    x = r * _np.sin(theta) * _np.cos(phi)
    y = r * _np.sin(theta) * _np.sin(phi)
    z = r * _np.cos(theta)
    return _np.transpose(_np.array([x, y, z]))


def coords_sph2cyl(sph):
    ''' Convert spherical coordinates to cylindrical '''
    r = sph[0]
    theta = sph[1]
    phi = sph[2]

    rho = r * _np.sin(theta)
    z = r * _np.cos(theta)
    return _np.transpose(_np.array([rho, phi, z]))


# Transformations of vectors fields

def vectors_cart2cyl(F, cart):
    ''' Convert vector field from cartesian coordinates to cylindrical'''
    Fx = F[0]
    Fy = F[1]
    Fz = F[2]
    x = cart[0]
    y = cart[1]

    rho = _np.sqrt(x ** 2 + y ** 2)
    Frh = (Fx * x + Fy * y) / rho
    Fp = (-y * Fx + x * Fy) / rho
    return _np.array([Frh, Fp, Fz])


def vectors_cart2sph(F, cart):
    ''' Convert vector field from cartesian coordinates to spherical'''
    Fx = F[0]
    Fy = F[1]
    Fz = F[2]
    x = cart[0]
    y = cart[1]
    z = cart[2]

    r = _np.sqrt(x ** 2 + y ** 2 + z ** 2)
    rho = _np.sqrt(x ** 2 + y ** 2)
    Fr = (Fx * x + Fy * y + Fz * z) / r
    Ft = ((Fx * x + Fy * y) * z - (x * x + y * y) * Fz) / r / rho
    Fp = (-y * Fx + x * Fy) / rho
    return _np.array([Fr, Ft, Fp])


def vectors_cyl2cart(F, cyl):
    ''' Convert vector field from cylindrical coordinates to cartesian'''
    Frh = F[0] 
    Fp = F[1]
    Fz = F[2]
    phi = cyl[1]
    
    sin_ph = _np.sin(phi)
    cos_ph = _np.cos(phi)
    Fx = cos_ph * Frh - sin_ph * Fp
    Fy = sin_ph * Frh + cos_ph * Fp
    return _np.array([Fx, Fy, Fz])


def vectors_cyl2sph(F, cyl):
    ''' Convert vector field from cylindrical coordinates to spherical'''
    Frh = F[0] 
    Fp = F[1]
    Fz = F[2]
    rho = cyl[0]
    z = cyl[2]

    r = _np.sqrt(rho ** 2 + z ** 2)
    Fr = (rho * Frh + z * Fz) / r
    Ft = (z * Frh - rho * Fz) / r
    return _np.array([Fr, Ft, Fp])


def vectors_sph2cart(F, sph):
    ''' Convert vector field from spherical coordinates to cartesian'''
    Fr = F[0] 
    Ft = F[1]
    Fp = F[2]
    theta = sph[1]
    phi = sph[2]

    sin_th = _np.sin(theta)
    cos_th = _np.cos(theta)
    sin_ph = _np.sin(phi)
    cos_ph = _np.cos(phi)
    Fx = sin_th * cos_ph * Fr + cos_th * cos_ph * Ft - sin_ph * Fp
    Fy = sin_th * sin_ph * Fr + cos_th * sin_ph * Ft + cos_ph * Fp
    Fz = cos_th * Fr - sin_th * Ft
    return _np.array([Fx, Fy, Fz])


def vectors_sph2cyl(F, sph):
    ''' Convert vector field from spherical coordinates to cylindrical '''
    Fr = F[0] 
    Ft = F[1]
    Fp = F[2]
    theta = sph[1]

    sin_th = _np.sin(theta)
    cos_th = _np.cos(theta)
    Frh = sin_th * Fr + cos_th * Ft
    Fz = cos_th * Fr - sin_th * Ft
    return _np.array([Frh, Fp, Fz])







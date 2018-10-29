import numpy as np
import scipy.integrate as integrate
from scipy.special import jn_zeros, jv
cimport scipy.special.cython_special as sp
cimport cython

cdef extern from "math.h":
    int floor(double x)
    double fabs(double x)
    double copysign(double x1, double x2)
    double exp(double x)
    double cosh(double x)
    double sinh(double x)


cpdef _get_clambda2_interpolation():
    ''' Calculates Clambda2 interpolation values '''

    N = 400
    lambd = np.linspace(1e-9, 40, N)
    n_zeros = 100
    kk = 16

    yint = np.zeros(N)
    cdef int i

    for i in range(N):
        bessel_zeros = jn_zeros(1, n_zeros - 1)
        mask = np.where(lambd[i] < bessel_zeros)
        args = np.append(lambd[i], bessel_zeros[mask])
        n1 = args.shape[0]
        lerp = np.tile(np.linspace(0, 1, kk, False), [1, n1 - 1])
        repeated1 = np.repeat(args[0:n1 - 1], kk)
        repeated2 = np.repeat(args[1:n1], kk)
        x = repeated1 + (repeated2 - repeated1) * lerp
        y = jv(1, x) / x
        yint[i] = integrate.simps(y, x)
    return (yint, lambd)

# initializing
clambda2_y, clambda2_x = _get_clambda2_interpolation()
j0_zeros = jn_zeros(0, 50)
j1_zeros = jn_zeros(1, 50)

cpdef PCD_field_1darray(double[:] rho, double[:] z, double I0, double D, double Rin, double Rout, double R1, double R2, double[:] zcs, double[:] dzcs_drho):
    
    cdef Py_ssize_t N = rho.shape[0]
    field = np.zeros((2, N))
    cdef double[:,:] field_memview = field
    cdef double Brho_temp, Bz_temp

    cdef int i 

    for i in range(N):
        field_memview[0, i], field_memview[1, i] = PCD_field(rho[i], z[i], I0, D, Rin, Rout, R1, R2, zcs[i], dzcs_drho[i])
    return field


cpdef PCD_field(double rho, double z, double I0, double D, double Rin, double Rout, double R1, double R2, double zcs, double dzcs_drho):

    cdef double const, Brho, Bz, Brho_temp, Bz_temp
    cdef double mu0 = 4 * 3.1415927 * 1e-7 / 71323 / 1000 * 1e9
    
    const = mu0 * I0 * 1e6
    Brho = 0
    Bz = 0
    Brho_temp, Bz_temp = semi_infinite_disk(rho, z, Rin, D, zcs, dzcs_drho, 0, 1)
    Brho += Brho_temp
    Bz += Bz_temp
    Brho_temp, Bz_temp = semi_infinite_disk(rho, z, R1, D, zcs, dzcs_drho, 0, 1)
    Brho -= Brho_temp
    Bz -= Bz_temp

    Brho_temp, Bz_temp = semi_infinite_disk(rho, z, R1, D, zcs, dzcs_drho, 1, 1)
    Brho += R1 * Brho_temp
    Bz += R1 * Bz_temp
    Brho_temp, Bz_temp = semi_infinite_disk(rho, z, R2, D, zcs, dzcs_drho, 1, 1)
    Brho -= R1 * Brho_temp
    Bz -= R1 * Bz_temp

    Brho_temp, Bz_temp = semi_infinite_disk(rho, z, R2, D, zcs, dzcs_drho, 0, 1)
    Brho += R1 / R2 * Brho_temp
    Bz += R1 / R2 * Bz_temp
    Brho_temp, Bz_temp = semi_infinite_disk(rho, z, Rout, D, zcs, dzcs_drho, 0, 1)
    Brho -= R1 / R2 * Brho_temp
    Bz -= R1 / R2 * Bz_temp

    Brho = const * Brho
    Bz = const * Bz

    return (Brho, Bz)


cpdef semi_infinite_disk(double rho, double z, 
                            double a, double D, double z_cs, double dzcs_drho, int mod, int bend):

    # calculates the field of the semi-infinite current disk at the given point
    # if mod == 1 uses I(rho) proportional to 1 / rho^2, otherwise 1 / rho
    # if bend == 1 takes bend of the current sheet into account
    # if bended, it is nessesary to provide z_cs and dzcs_drho values

    cdef int n_zeros = 25
    cdef int k = 10 # must be even!
    cdef double[:] zeros_rho = np.zeros(n_zeros)
    cdef double[:] zeros_z = np.zeros(n_zeros)
    cdef double[:] val_rho = np.zeros(k + 1)
    cdef double[:] val_z = np.zeros(k + 1)
    cdef double lambd_r, lambd_z, cl_r, cl_z, hr, hz, z_new, k_d
    cdef int i, j

    cdef double[:] j0_zeros_m = j0_zeros
    cdef double[:] j1_zeros_m = j1_zeros
    cdef double[:] clambda2_x_m = clambda2_x
    cdef double[:] clambda2_y_m = clambda2_y

    cdef double Brho, Bz

    get_zeros(zeros_rho, zeros_z, n_zeros, a, rho, j0_zeros_m, j1_zeros_m)
    Brho = 0
    Bz = 0
    for i in range(n_zeros-1):
        hr = (zeros_rho[i + 1] - zeros_rho[i]) / k
        hz = (zeros_z[i + 1] - zeros_z[i]) / k
        for j in range(k + 1):
            lambd_r = hr * j + zeros_rho[i]
            lambd_z = hz * j + zeros_z[i]
            if mod == 1:
                cl_r = clambda2(lambd_r, a, clambda2_x_m, clambda2_y_m)
                cl_z = clambda2(lambd_z, a, clambda2_x_m, clambda2_y_m)
            else:
                cl_r = clambda1(lambd_r, a)
                cl_z = clambda1(lambd_z, a)
            if bend == 1:
                z_new = z - z_cs
                val_rho[j] = b_rho_integrand(lambd_r, rho, z_new, D, cl_r)
                val_z[j] = b_z_integrand(lambd_z, rho, z_new, D, cl_z)
            else:
                val_rho[j] = b_rho_integrand(lambd_r, rho, z, D, cl_r)
                val_z[j] = b_z_integrand(lambd_z, rho, z, D, cl_z)
        Brho = Brho + simpson(k + 1, val_rho, hr)
        Bz = Bz + simpson(k + 1, val_z, hz)
    if bend == 1:
        Bz = Bz + dzcs_drho * Brho
    return (Brho, Bz)



cpdef get_zeros(double[:] zeros_rho, double[:] zeros_z, int n_zeros, double a, double rho, double[:] j0_zeros, double[:] j1_zeros):
    ''' Returns first n zeros for Brho and Bz integrands. 
    
    Integrals of the periodic functions are divided into a series of integrals over intervals between zeros of integrands.
    '''
    cdef double[:] bessel_zeros_rho = np.zeros(2 * n_zeros + 1)
    cdef double[:] bessel_zeros_z = np.zeros(2 * n_zeros + 1)

    cdef int i

    bessel_zeros_rho[0] = 1e-9
    bessel_zeros_z[0] = 1e-9

    for i in range(1, n_zeros + 1):
        bessel_zeros_rho[i] = j0_zeros[i-1] / a
        bessel_zeros_z[i] = bessel_zeros_rho[i]
    for i in range(n_zeros + 1, 2 * n_zeros + 1):
        bessel_zeros_rho[i] = j1_zeros[i - n_zeros - 1] / rho
        bessel_zeros_z[i] = j0_zeros[i - n_zeros - 1] / rho
    quicksort(bessel_zeros_rho, 0, n_zeros * 2)
    quicksort(bessel_zeros_z, 0, n_zeros * 2)

    for i in range(n_zeros):
        zeros_rho[i] = bessel_zeros_rho[i]
        zeros_z[i] = bessel_zeros_z[i]



cdef double b_rho_integrand(double l, double rho, double z, double D, double clmbd):
    ''' Calculates integrand for Brho for semi-infinite current disk'''
    if (fabs(z) <= D):
        return clmbd * sp.j1(l * rho) * sinh(l * z) * exp(-l * D)
    else:
        return clmbd * copysign(1.0,z) * sp.j1(l * rho) * sinh(l * D) * exp(-l * fabs(z))

cdef double b_z_integrand(double l, double rho, double z, double D, double clmbd):
    ''' Calculates integrand for Bz for semi-infinite current disk'''
    if (fabs(z) <= D):
        return clmbd * sp.j0(l * rho) * (1.0 - cosh(l * z) * exp(-l * D))
    else:
        return clmbd * sp.j0(l * rho) * sinh(l * D) * exp(-l * fabs(z)) 


cdef double clambda2(double x, double a, double[:] clambda2_x, double[:] clambda2_y):
    ''' C(lambda) for I proportional 1 / rho ^ 2. '''
    if x > clambda2_x[-1]:
        return sp.j0(x * a) / x / a
    else:
        return interpolate(x * a, clambda2_y, clambda2_x)

cdef double clambda1(double x, double a):
    ''' C(lambda) for I proportional 1 / rho. See Connerney et al. (1981) https://doi.org/10.1029/JA086iA10p08370 '''
    return sp.j0(x * a) / x


cpdef double interpolate(double x, double[:] data_y, double[:] data_x):
    ''' Interpolates a set of data with a piecewise linear function. '''
    cdef int ind
    cdef double x_range
    cdef Py_ssize_t size = data_y.shape[0]
    if x <= data_x[0]:
        return data_y[0]
    if x >= data_x[size - 1]:
        return data_y[size - 1]
    
    x_range = data_x[size - 1] - data_x[0]
    ind = floor((x - data_x[0]) / x_range * (size - 1))
    return (data_y[ind + 1] - data_y[ind]) * (x - data_x[ind]) / (data_x[ind + 1] - data_x[ind]) + data_y[ind]
    

cpdef double simpson(int size, double[:] values, double h):
    ''' Calculates integral by composite Simpson's rule. '''
    cdef double s = 0
    cdef int i
    s = 0
    for i in range(1, size, 2):
        s = s + values[i - 1] + 4 * values[i] + values[i + 1]
    return s * h / 3   


cpdef quicksort(double[:] s_arr, int first, int last):
    ''' Sorts an array of doubles by quicksort algorithm. '''
    cdef int left = first
    cdef int right = last
    cdef double middle = s_arr[(left + right) // 2]
    cdef double tmp
    while True:
        while (s_arr[left] < middle): 
            left+=1
        while (s_arr[right] > middle):
            right-=1
        if left <= right:
            tmp = s_arr[left];
            s_arr[left] = s_arr[right];
            s_arr[right] = tmp;
            left+=1;
            right-=1;
        if left > right: break
    if first < right:
        quicksort(s_arr, first, right)
    if left < last:
        quicksort(s_arr, left, last)
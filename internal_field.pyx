import numpy as np
cimport scipy.special.cython_special as sp
cimport cython

cdef extern from "math.h":
     double sin(double arg)
     double cos(double arg)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False) # Deactivate negative indexing
@cython.cdivision(True) 

cpdef internal_field(double r, double theta, double phi, double[:,:] g, double[:,:] h, double[:,:] sch):
    
    cdef Py_ssize_t max_degree = g.shape[0]
    cdef double[3] field

    cdef int n, m
    cdef double b_r, b_t, b_ph
    cdef double legendre, legendre_d1, start_mult_r, start_mult_t, start_mult_ph, 
    cdef double cos_th, one_minus
    cdef double[100] cos_mphi, sin_mphi, prev_legendres

    for m in range(max_degree + 1):
        cos_mphi[m] = cos(m * phi)
        sin_mphi[m] = sin(m * phi)
        prev_legendres[m] = 0
    
    prev_legendres[0] = 1
   
    b_r = 0
    b_t = 0
    b_ph = 0

    start_mult_r = 1 / r / r
    start_mult_t = sin(theta) / r / r
    start_mult_ph = - 1 / sin(theta) / r / r

    cos_th = cos(theta)
    one_minus = 1 / (1 - cos_th * cos_th)

    for n in range(1, max_degree + 1):
        start_mult_r = start_mult_r / r
        start_mult_t = start_mult_t / r
        start_mult_ph = start_mult_ph / r
        for m in range(n + 1):
                legendre =  sp.lpmv(m, n, cos_th)
                legendre_d1 = ((n + m) * prev_legendres[m] - n * cos_th * legendre) * one_minus
                prev_legendres[m] = legendre
                legendre *= sch[n - 1, m]
                legendre_d1 *= sch[n - 1, m]

                b_r = b_r + start_mult_r * (n + 1) * legendre * (g[n - 1,m] * cos_mphi[m] + h[n - 1,m] * sin_mphi[m])
                b_t = b_t + start_mult_t * legendre_d1 * (g[n - 1,m] * cos_mphi[m] + h[n - 1,m] * sin_mphi[m])
                b_ph = b_ph + start_mult_ph * legendre * m * (-g[n - 1,m] * sin_mphi[m] + h[n - 1,m] * cos_mphi[m])

    field[0] = b_r
    field[1] = b_t
    field[2] = b_ph

    return field

cpdef internal_field_1darray(double[:] r, double[:] theta, double[:] phi, double[:,:] g, double[:,:] h, double[:,:] sch):

    cdef Py_ssize_t N = r.shape[0]
    cdef Py_ssize_t max_degree = g.shape[0]

    field = np.zeros((3,N))
    cdef double[:,:] field_memview = field
    cdef double[3] field_temp 

    cdef int i

    for i in range(N):
        field_temp = internal_field(r[i], theta[i], phi[i], g, h, sch)
        field_memview[0,i] = field_temp[0]
        field_memview[1,i] = field_temp[1]
        field_memview[2,i] = field_temp[2]
    return field
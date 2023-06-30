"""
The Williamson 6 test case, Rossby-Haurwitz wave

This uses an icosahedral mesh of the sphere, and runs a series of resolutions to find convergence.
"""

from re import L
from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, pi, sqrt, min_value, errornorm, norm, cos, sin,
                       acos, grad, curl, div, conditional)
import sys
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

day = 24.*60.*60.
# setup resolution and timestepping parameters for convergence test
dts = [ 500., 250. ]
tmax = 1*day
ndumps = 1

dt_true = 50. 

# setup shallow water parameters
a = 6371220.
H = 5960.
parameters = ShallowWaterParameters(H=H)
kvals_Mvals={8:4, 6:3, 4:2, 2:1}
kvals = [8, 6, 4, 2]

cols=['b','g','r','c']

ref_level= 5
degree = 3
mesh = IcosahedralSphereMesh(radius=a,
                             refinement_level=ref_level, degree=2)

x = SpatialCoordinate(mesh)

# Domain
domain = Domain(mesh, dt_true, 'BDM', degree)

# Equations
lat, lon = latlon_coords(mesh)
Omega = parameters.Omega
fexpr = 2*Omega * x[2] / a
eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, u_transport_option='vector_advection_form')

# I/O
dirname = "williamson_6_ref%s_dt%s_k%s_deg%s" % (ref_level, dt_true, 1, degree)
dumpfreq = int(tmax / (ndumps*dt_true))
output = OutputParameters(dirname=dirname,
                        dumpfreq=dumpfreq,
                        checkpoint_method = 'dumbcheckpoint',
                        log_level='INFO')
io = IO(domain, output)
scheme = BDF2(domain)
stepper = Timestepper(eqns, scheme, io)

# ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #
u0 = stepper.fields('u')
D0 = stepper.fields('D')

Vu = domain.spaces("HDiv")
K = Constant(7.847e-6)
w = K
R = 4.
h0 = 8000.
g = parameters.g

# Intilising the velocity field
CG2 = FunctionSpace(mesh, 'CG', 2)
psi = Function(CG2)
psiexpr = -a**2 * w * sin(lat) + a**2 * K * cos(lat)**R * sin(lat) * cos(R*lon)
psi.interpolate(psiexpr)
uexpr = domain.perp(grad(psi))

# Initilising the depth field
A = (w / 2) * (2 * Omega + w) * cos(lat)**2 + 0.25 * K**2 * cos(lat)**(2 * R) * ((R + 1) * cos(lat)**2 + (2 * R**2 - R - 2) - 2 * R**2 * cos(lat)**(-2))
B_frac = (2 * (Omega + w) * K) / ((R + 1) * (R + 2))
B = B_frac * cos(lat)**R * ((R**2 + 2 * R + 2) - (R + 1)**2 * cos(lat)**2)
C = (1 / 4) * K**2 * cos(lat)**(2 * R) * ((R + 1)*cos(lat)**2 - (R + 2))
Dexpr = h0 * g + a**2 * (A + B*cos(lon*R) + C * cos(2 * R * lon))

# Finalizing fields and setting reference profiles
u0.project(uexpr)
D0.interpolate(Dexpr / g)
# Dbar is a background field for diagnostics
Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)

u = stepper.fields('u')
D = stepper.fields('D')

utrue_data = u.dat.data[:]
Dtrue_data = D.dat.data[:]

print("dt", "k", "error_norm_D", "norm_D", "error_norm_u", "norm_u")

for dt in dts:

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #
    mesh = IcosahedralSphereMesh(radius=a,
                                  refinement_level=ref_level, degree=2)

    x = SpatialCoordinate(mesh)
    # Domain
    domain = Domain(mesh, dt, 'BDM', degree)

    # Equations
    lat, lon = latlon_coords(mesh)
    Omega = parameters.Omega
    fexpr = 2*Omega * x[2] / a
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, u_transport_option='vector_advection_form')
    for kval, Mval in kvals_Mvals.items():

        # I/O
        dirname = "williamson_6_ref%s_dt%s_k%s_deg%s" % (ref_level, dt, kval, degree)
        dumpfreq = int(tmax / (ndumps*dt))
        output = OutputParameters(dirname=dirname,
                                dumpfreq=dumpfreq,
                                checkpoint_method = 'dumbcheckpoint',
                                log_level='INFO')
        io = IO(domain, output)

        # Time stepper
        k = kval
        M = Mval

        scheme = IMEX_SDC(domain, M, k)

        stepper = Timestepper(eqns, scheme, io)

        # ------------------------------------------------------------------------ #
        # Initial conditions
        # ------------------------------------------------------------------------ #
        u0 = stepper.fields('u')
        D0 = stepper.fields('D')

        Vu = domain.spaces("HDiv")
        K = Constant(7.847e-6)
        w = K
        R = 4.
        h0 = 8000.
        g = parameters.g

        # Intilising the velocity field
        CG2 = FunctionSpace(mesh, 'CG', 2)
        psi = Function(CG2)
        psiexpr = -a**2 * w * sin(lat) + a**2 * K * cos(lat)**R * sin(lat) * cos(R*lon)
        psi.interpolate(psiexpr)
        uexpr = domain.perp(grad(psi))

        # Initilising the depth field
        A = (w / 2) * (2 * Omega + w) * cos(lat)**2 + 0.25 * K**2 * cos(lat)**(2 * R) * ((R + 1) * cos(lat)**2 + (2 * R**2 - R - 2) - 2 * R**2 * cos(lat)**(-2))
        B_frac = (2 * (Omega + w) * K) / ((R + 1) * (R + 2))
        B = B_frac * cos(lat)**R * ((R**2 + 2 * R + 2) - (R + 1)**2 * cos(lat)**2)
        C = (1 / 4) * K**2 * cos(lat)**(2 * R) * ((R + 1)*cos(lat)**2 - (R + 2))
        Dexpr = h0 * g + a**2 * (A + B*cos(lon*R) + C * cos(2 * R * lon))

        # Finalizing fields and setting reference profiles
        u0.project(uexpr)
        D0.interpolate(Dexpr / g)
        # Dbar is a background field for diagnostics
        Dbar = Function(D0.function_space()).assign(H)
        stepper.set_reference_profiles([('D', Dbar)])

        # ------------------------------------------------------------------------ #
        # Run
        # ------------------------------------------------------------------------ #

        stepper.run(t=0, tmax=tmax)

        u = stepper.fields('u')
        D = stepper.fields('D')

        usol = Function(u.function_space())
        Dsol = Function(D.function_space())

        usol.dat.data[:] = utrue_data
        Dsol.dat.data[:] = Dtrue_data

        error_norm_D = errornorm(Dsol, stepper.fields("D"), mesh=mesh)
        norm_D = norm(Dsol, mesh=mesh)

        error_norm_u = errornorm(usol, stepper.fields("u"), mesh=mesh)
        norm_u = norm(usol, mesh=mesh)

        print(dt, kval, error_norm_D, norm_D, error_norm_u, norm_u)
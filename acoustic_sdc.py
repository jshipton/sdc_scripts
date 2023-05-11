from firedrake import (PeriodicIntervalMesh, FunctionSpace, MixedFunctionSpace,
                       TestFunctions, Function, dx, Constant, split,
                       SpatialCoordinate, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, File, exp, cos, assemble)
from gusto import *
#from gusto.fml.form_manipulation_labelling import Label, drop, all_terms
#from gusto.labels import time_derivative, subject, replace_subject, implicit, explicit, transport, advecting_velocity
import numpy as np
import scipy
from scipy.special import legendre


class AcousticEquation(PrognosticEquation):

    field_names = ["u", "p"]

    def __init__(self, domain):

        Vu = FunctionSpace(mesh, 'Lagrange', 1)
        Vp = FunctionSpace(mesh, 'CG', 1)
        W = MixedFunctionSpace((Vu, Vp))

        super().__init__(domain, W, "u_p")

        w, phi = TestFunctions(W)
        X = Function(W)
        u, p = split(X)

        c_s = Constant(1)                   # speed of sound
        u_mean = Constant(0.05)                  # mean flow
        ubar = Function(Vu).assign(u_mean)

        mass_form = time_derivative(subject(w * u * dx + phi * p * dx, X))
        fast_form = subject(c_s * (w * p.dx(0) + phi * u.dx(0)) * dx, X)
        slow_form = transport(subject(ubar * (w * u.dx(0) + phi * p.dx(0)) * dx, X))
        nonlinear = False
        if nonlinear:
            self.residual = mass_form + fast_form + transporting_velocity(slow_form, ubar)
        else:
            self.residual = mass_form + fast_form + slow_form



a = 0                           # time of start
b = 3                           # time of end
n = 512                         # number of spatial nodes
n_steps = 154.                  # number of time steps
dt = (b-a)/(n_steps)            # delta t

mesh = PeriodicIntervalMesh(n, 1)

output = OutputParameters(dirname="new_acoustic")

domain = Domain(mesh, dt, "CG", 1)

eqn = AcousticEquation(domain)

io = IO(domain, output)

M = 3
maxk = 2
scheme = IMEX_SDC(domain, M, maxk)
timestepper = Timestepper(eqn, scheme, io)

p0 = timestepper.fields("p")

x1 = 0.25
x0 = 0.75
sigma = 0.1

def p_0(x, sigma=sigma):
    return exp(-x**2/sigma**2)
                 
def p_1(x, p0=p_0, sigma=sigma, k=7.2*np.pi):
    return p0(x)*cos(k*x/sigma)

def p_init(x, p0=p_0, p1=p_1, x0=x0, x1=x1, coeff=1.):
    return p_0(x-x0) + coeff*p_1(x-x1)

x = SpatialCoordinate(mesh)[0]
p0.interpolate(p_init(x))

timestepper.run(a, b)

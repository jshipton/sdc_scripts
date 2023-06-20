from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate, as_vector,
                       FunctionSpace, pi, cos, sin, sqrt, acos)

# radius of sphere
R = 6371220.

dirname = "sdc_advection"
mesh = IcosahedralSphereMesh(radius=R, refinement_level=3, degree=1)
x = SpatialCoordinate(mesh)
#global_normal = x
#mesh.init_cell_orientations(x)

output = OutputParameters(dirname=dirname)

dt = 300
degree = 1
domain = Domain(mesh, dt, family="BDM", degree=degree)

io = IO(domain, output)

Vf = domain.spaces("DG")

eqn = AdvectionEquation(domain, Vf, field_name="tracer")
# eqn.residual = advecting_velocity.remove(eqn.residual)
# for t in eqn.residual:
#     print(t.labels.keys())
M = 3
maxk = 2
scheme = IMEX_SDC(domain, M, maxk)
#scheme = SSPRK3(domain)
timestepper = PrescribedTransport(eqn, scheme, io)

# interpolate initial conditions
u0 = timestepper.fields("u")
day = 24.*60.*60.
u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
u0.project(uexpr)

q0 = timestepper.fields("tracer")
theta, lamda = latlon_coords(mesh)
R0 = R/3.
R0sq = R0**2
lamda_c = 3*pi/2.
theta_c = 0.
r = R*acos(cos(theta)*cos(lamda-lamda_c))
qbar = 1000
qexpr = qbar/2 * (1 + cos(pi*r/R0))
q0.interpolate(conditional(r < R0, qexpr, 0))

tmax = 12*day
timestepper.run(0, tmax)

import pyomo.environ as pyo
m1 = pyo.ConcreteModel()
m1.time = pyo.Set(initialize=[0, 1, 2])
m1.units = pyo.Set(initialize=["A", "B"])
m1.var1 = pyo.Var(m1.time, initialize=1.0)

@m1.Block(m1.units, m1.time)
def block(m, i, t):
    m.var2 = pyo.Var(initialize=0.0)

# Fix initial condition using a slice
m1.block[:, 0.0].var2.fix(2.0)

m1.var2_A_ref = pyo.Reference(m1.block["A", :].var2)
m1.var2_B_ref = pyo.Reference(m1.block["B", :].var2)

# Set all variable values to their initial conditions
for var in m1.component_objects(pyo.Var, descend_into=False):
    # We know that these variables are indexed only by time
    var[:] = var[0.0]

m2 = pyo.ConcreteModel()
m2.time = pyo.Set(initialize=[0, 2, 4])
m2.units = pyo.Set(initialize=["A", "B"])

@m2.Block(m2.units, m2.time)
def block(m, i, t):
    m.var2 = pyo.Var(initialize=0.0)

# The "referent" attribute recovers the slice from the reference
m1_cuid = pyo.ComponentUID(m1.var2_A_ref.referent)
m2_cuid = pyo.ComponentUID(m2.block["A", :].var2)
print(m1_cuid == m2_cuid)

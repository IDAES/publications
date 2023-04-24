import pyomo.environ as pyo
m = pyo.ConcreteModel()
m.time = pyo.Set(
    initialize=[0, 1, 2]
)
m.temperature = pyo.Var(m.time)
@m.Block(m.time)
def block(m, t):
    m.heat_cap = pyo.Var()
m.heat_cap_ref = pyo.Reference(
    m.block[:].heat_cap
)
for var in m.component_objects(
    pyo.Var, descend_into=False
):
    var[0].fix()

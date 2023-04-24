import pyomo.environ as pyo
m = pyo.ConcreteModel()
m.time = pyo.Set(
    initialize=[0, 1, 2]
)
m.units = pyo.Set(
    initialize=["A", "B"]
)
@m.Block(m.units, m.time)
def block(m, i, t):
    m.var = pyo.Var(initialize=1.0)
m.block[:, 0].var.fix()

import pyomo.environ as pyo
m1 = pyo.ConcreteModel()
m1.time = pyo.Set(
    initialize=[0, 1, 2]
)
m1.unit = pyo.Set(
    initialize=["A", "B"]
)
m1.var = pyo.Var(m1.unit, m1.time)

m2 = pyo.ConcreteModel()
m2.time = pyo.Set(
    initialize=[0, 1, 2]
)
m2.unit = pyo.Set(
    initialize=["A", "B"]
)
m2.var = pyo.Var(m2.unit, m2.time)

cuid1 = pyo.ComponentUID(
    m1.var["A", :]
)
cuid2 = pyo.ComponentUID(
    m2.var["B", :]
)
print(cuid1)
print(cuid2)
print(cuid1 == cuid2)

## Serialization module

This module contains code for serializing variable values in a structured
model, i.e. the values of variables indexed by particular sets are stored
in nd-arrays (nested lists).
Components are grouped according to indexing sets along which their data
are stored.
Names and values for indexing sets are stored as well.
The key for each component (or indexing set) is the string representation
of that component's (or its referent's) ComponentUID.

This is intended to be a fairly general representation of index set-
structured variable values from Pyomo models.
One thing it does not handle, however, is further subdivision of
variables, e.g. design variables vs. uncertain parameters, or
differential vs. algebraic variables.
Allowing such subdivisions via blocks is a natural extension of this
data structure, as is allowing these blocks to be indexed.

A more challenging extension would be extending this data structure to
encode constraints. Two challenges here are the representation of
the nonlinear constraint body and how to encode a "slice" over a
variable index. The constraint would have to be "the same" across
elements of its inexing set. It remains undetermined how we would
encode constraints which become undefined at certain indices, e.g.
discretization equations. One option is to use the nl format's
polish-prefix notation for constraint expressions. Would need
additional operators for getattr and getitem.

May be desirable to add other aspects of a component's state, e.g.
fixed, active, bounds, domain.
The combination of the above is a structure-aware JSON serialization
of a Pyomo model.
Ideally we could use this data structure to construct an "equivalent" model
in another modeling language, subject to the restrictions of that language
of course. (E.g. cannot encode blocks in GAMS, to my knowledge.)

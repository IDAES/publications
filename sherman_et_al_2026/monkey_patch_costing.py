# Derived from idaes/models_extra/power_generation/costing/power_plant_capcost.py

#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files IDAES_COPYRIGHT.md and 
# IDAES_LICENSE.md for full copyright and license information.
#################################################################################

from pyomo.environ import (
    Param,
    Var,
    units as pyunits,
)

import idaes.logger as idaeslog
from idaes.models_extra.power_generation.costing.power_plant_capcost import QGESSCostingData

_log = idaeslog.getLogger(__name__)

def monkey_patch_costing():
    QGESSCostingData.get_fixed_OM_costs = get_fixed_OM_costs

def get_fixed_OM_costs(
    b,
    net_power=None,
    nameplate_capacity=650,
    capacity_factor=0.85,
    labor_rate=38.50,
    labor_burden=30,
    operators_per_shift=6,
    tech=1,
    fixed_TPC=None,
    CE_index_year="2018",
):
    """
    Creates constraints for the following fixed O&M costs in $MM/yr:
        1. Annual operating labor
        2. Maintenance labor
        3. Admin and support labor
        4. Property taxes and insurance
        5. Other fixed costs
        6. Total fixed O&M cost
        7. Maintenance materials (actually a variable cost,
            but scales off TPC)

    These costs apply to the project as a whole and are scaled based on the
    total TPC.

    Args:
        b: costing block to add fixed cost variables and constraints to
        net_power: actual plant output in MW, only required if calculating
            variable costs
        nameplate_capacity: rated plant output in MW
        capacity_factor: multiplicative factor for normal operating
            capacity
        labor_rate: hourly rate of plant operators in project dollar year
        labor_burden: a percentage multiplier used to estimate non-salary
            labor expenses
        operators_per_shift: average number of operators per shift
        tech: int 1-7 representing the categories in get_PP_costing, used
            to determine maintenance costs
        fixed_TPC: The TPC in $MM that will be used to determine fixed O&M,
            costs. If the value is None, the function will try to use the
            TPC calculated from the individual units.
        CE_index_year: year for cost basis, e.g. "2018" to use 2018 dollars

    Returns:
        None
    """

    try:
        CE_index_units = getattr(pyunits, "MUSD_" + CE_index_year)
    except AttributeError:
        raise AttributeError(
            "CE_index_year %s is not a valid currency base option. "
            "Valid CE index options include CE500, CE394 and years from "
            "1990 to 2020." % (CE_index_year)
        )

    # create and fix total_TPC if it does not exist yet
    if not hasattr(b, "total_TPC"):
        b.total_TPC = Var(
            initialize=100,
            bounds=(0, 1e4),
            doc="total TPC in $MM",
            units=CE_index_units,
        )
        if fixed_TPC is None:
            b.total_TPC.fix(100 * CE_index_units)
            _log.warning(
                "b.costing.total_TPC does not exist and a value "
                "for fixed_TPC was not specified, total_TPC will be "
                "fixed to 100 MM$"
            )
        else:
            @b.Constraint()
            def total_TPC_constraint(c):
                return c.total_TPC == fixed_TPC
    else:
        if fixed_TPC is not None:
            _log.warning(
                "b.costing.total_TPC already exists, the value "
                "passed for fixed_TPC will be ignored."
            )

    # make params
    b.labor_rate = Param(
        initialize=labor_rate,
        mutable=True,
        units=getattr(pyunits, "USD_" + CE_index_year) / pyunits.hr,
    )
    b.labor_burden = Param(
        initialize=labor_burden, mutable=True, units=pyunits.dimensionless
    )
    b.operators_per_shift = Param(
        initialize=operators_per_shift, mutable=True, units=pyunits.dimensionless
    )
    b.nameplate_capacity = Param(
        initialize=nameplate_capacity, mutable=True, units=pyunits.MW
    )

    maintenance_percentages = {
        1: [0.4, 0.016],
        2: [0.4, 0.016],
        3: [0.35, 0.03],
        4: [0.35, 0.03],
        5: [0.35, 0.03],
        6: [0.4, 0.019],
        7: [0.4, 0.016],
    }

    b.maintenance_labor_TPC_split = Param(
        initialize=maintenance_percentages[tech][0], mutable=True
    )
    b.maintenance_labor_percent = Param(
        initialize=maintenance_percentages[tech][1], mutable=True
    )
    b.maintenance_material_TPC_split = Param(
        initialize=(1 - maintenance_percentages[tech][0]), mutable=True
    )
    b.maintenance_material_percent = Param(
        initialize=maintenance_percentages[tech][1], mutable=True
    )

    # make vars
    b.annual_operating_labor_cost = Var(
        initialize=1,
        bounds=(0, 1e4),
        doc="annual labor cost in $MM/yr",
        units=CE_index_units,
    )
    b.maintenance_labor_cost = Var(
        initialize=1,
        bounds=(0, 1e4),
        doc="maintenance labor cost in $MM/yr",
        units=CE_index_units,
    )
    b.admin_and_support_labor_cost = Var(
        initialize=1,
        bounds=(0, 1e4),
        doc="admin and support labor cost in $MM/yr",
        units=CE_index_units,
    )
    b.property_taxes_and_insurance = Var(
        initialize=1,
        bounds=(0, 1e4),
        doc="property taxes and insurance cost in $MM/yr",
        units=CE_index_units,
    )
    b.total_fixed_OM_cost = Var(
        initialize=4,
        bounds=(0, 1e4),
        doc="total fixed O&M costs in $MM/yr",
        units=CE_index_units,
    )

    # variable for user to assign other fixed costs to,
    # fixed to 0 by default
    b.other_fixed_costs = Var(
        initialize=0,
        bounds=(0, 1e4),
        doc="other fixed costs in $MM/yr",
        units=CE_index_units,
    )
    b.other_fixed_costs.fix(0)

    # maintenance material cost is technically a variable cost, but it
    # makes more sense to include with the fixed costs because it uses TPC
    b.maintenance_material_cost = Var(
        initialize=2e-7,
        bounds=(0, 1e4),
        doc="cost of maintenance materials in $MM/year",
        units=CE_index_units / pyunits.year,
    )

    # create constraints
    TPC = b.total_TPC  # quick reference to total_TPC

    # calculated from labor rate, labor burden, and operators per shift
    @b.Constraint()
    def annual_labor_cost_rule(c):
        return c.annual_operating_labor_cost == pyunits.convert(
            (
                c.operators_per_shift
                * c.labor_rate
                * (1 + c.labor_burden / 100)
                * 8760
                * pyunits.hr
            ),
            CE_index_units,
        )

    # technology specific percentage of TPC
    @b.Constraint()
    def maintenance_labor_cost_rule(c):
        return c.maintenance_labor_cost == (
            TPC * c.maintenance_labor_TPC_split * c.maintenance_labor_percent
        )

    # 25% of the sum of annual operating labor and maintenance labor
    @b.Constraint()
    def admin_and_support_labor_cost_rule(c):
        return c.admin_and_support_labor_cost == (
            0.25 * (c.annual_operating_labor_cost + c.maintenance_labor_cost)
        )

    # 2% of TPC
    @b.Constraint()
    def taxes_and_insurance_cost_rule(c):
        return c.property_taxes_and_insurance == 0.02 * TPC

    # sum of fixed O&M costs
    @b.Constraint()
    def total_fixed_OM_cost_rule(c):
        return c.total_fixed_OM_cost == (
            c.annual_operating_labor_cost
            + c.maintenance_labor_cost
            + c.admin_and_support_labor_cost
            + c.property_taxes_and_insurance
            + c.other_fixed_costs
        )

    # technology specific percentage of TPC
    @b.Constraint()
    def maintenance_material_cost_rule(c):
        if net_power is not None:
            return c.maintenance_material_cost == (
                TPC
                * c.maintenance_material_TPC_split
                * c.maintenance_material_percent
                / (capacity_factor * pyunits.year)
                * net_power[0]
                / c.nameplate_capacity
            )
        else:
            return c.maintenance_material_cost == (
                TPC
                * c.maintenance_material_TPC_split
                * c.maintenance_material_percent
                / (capacity_factor * pyunits.year)
                * (capacity_factor)
            )
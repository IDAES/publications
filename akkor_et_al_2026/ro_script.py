import pyomo.environ as pyo
import pyomo.contrib.pyros as pyros

from PZ_AFS_flowsheet import flowheet_model
from optimize_nominal_case import *
import logging

# Set-up solvers

pyros_solver = pyo.SolverFactory("pyros")

local_solver = pyo.SolverFactory("gams")
local_solver.options = {"solver": "conopt3"}

l1 = pyo.SolverFactory("gams")
l1.options = {"solver": "ipopt"}

l2 = pyo.SolverFactory("gams")
l2.options = {"solver": "knitro"}

l3 = pyo.SolverFactory("gams")
l3.options = {"solver": "conopt4"}

l4 = pyo.SolverFactory("ipopt")


global_solver = pyo.SolverFactory("gams")
global_solver.options = {"solver": "baron"}

# Construct model
m = flowheet_model("ngcc", "commercial", False, True)
m.fs = optimize_plant(m.fs, local_solver, 40)

# Remove bounds to reduce separation problems
speciation_components = [
    "HCO3-",
    "H3O+",
    "CO2",
    "H2O",
    "CO3-2",
    "OH-",
    "PZ",
    "PZH+",
    "PZCOO-",
    "H+PZCOO-",
    "PZ(COO-)2",
]

m.fs.F_bottom.setlb(None)
m.fs.F_hot.setlb(None)
m.fs.F_cold.setlb(None)
m.fs.F_warm.setlb(None)
m.fs.F_l_prime.setlb(None)
m.fs.x_prime.setlb(None)
m.fs.x_prime.setub(None)

m.fs.afs.flash_tank.K_flash.setlb(None)
m.fs.afs.flash_tank.feed_properties[0].flow_mol.setlb(None)
m.fs.afs.flash_tank.liquid_properties[0].flow_mol.setlb(None)
m.fs.afs.flash_tank.vapor_properties[0].flow_mol.setlb(None)
m.fs.afs.flash_tank.vapor_properties[0].pressure.setlb(None)
m.fs.afs.flash_tank.vapor_properties[0].pressure.setub(None)

m.fs.afs.flash_tank.feed_properties[0].temperature.setlb(None)
m.fs.afs.flash_tank.liquid_properties[0].temperature.setlb(None)
m.fs.afs.flash_tank.vapor_properties[0].temperature.setlb(None)
m.fs.afs.flash_tank.feed_properties[0].temperature.setub(None)
m.fs.afs.flash_tank.liquid_properties[0].temperature.setub(None)
m.fs.afs.flash_tank.vapor_properties[0].temperature.setub(None)

for j in ["CO2", "H2O", "PZ"]:
    m.fs.afs.flash_tank.feed_properties[0].mole_frac_comp[j].setlb(None)
    m.fs.afs.flash_tank.vapor_properties[0].mole_frac_comp[j].setlb(None)
    m.fs.afs.flash_tank.liquid_properties[0].mole_frac_comp[j].setlb(None)
    m.fs.afs.flash_tank.liquid_properties[0].P_sat[j].setlb(None)

m.fs.absorber.cross_sectional_area.setlb(None)
m.fs.afs.stripper.cross_sectional_area.setlb(None)

for i in range(0, 41):
    m.fs.absorber.effective_vapor_velocity[i].setlb(None)
    m.fs.absorber.effective_liquid_velocity[i].setlb(None)
    m.fs.afs.stripper.effective_vapor_velocity[i].setlb(None)
    m.fs.afs.stripper.effective_liquid_velocity[i].setlb(None)

    m.fs.absorber.heat_transfer_coefficient[i].setlb(None)
    m.fs.afs.stripper.heat_transfer_coefficient[i].setlb(None)

    m.fs.absorber.vapor_velocity[i].setlb(None)
    m.fs.absorber.liquid_velocity[i].setlb(None)
    m.fs.afs.stripper.vapor_velocity[i].setlb(None)
    m.fs.afs.stripper.liquid_velocity[i].setlb(None)

    m.fs.absorber.liquid_properties[0, i].reaction_rate.setlb(None)
    m.fs.afs.stripper.liquid_properties[0, i].reaction_rate.setlb(None)
    m.fs.absorber.liquid_properties[0, i].henry.setlb(None)
    m.fs.afs.stripper.liquid_properties[0, i].henry.setlb(None)
    m.fs.absorber.liquid_properties[0, i].diffusivity_cst.setlb(None)
    m.fs.afs.stripper.liquid_properties[0, i].diffusivity_cst.setlb(None)
    m.fs.absorber.liquid_properties[0, i].viscosity_water.setlb(None)
    m.fs.afs.stripper.liquid_properties[0, i].viscosity_water.setlb(None)
    m.fs.absorber.liquid_properties[0, i].visc_d.setlb(None)
    m.fs.afs.stripper.liquid_properties[0, i].visc_d.setlb(None)
    m.fs.absorber.vapor_properties[0, i].conc_mol.setlb(None)
    m.fs.afs.stripper.vapor_properties[0, i].conc_mol.setlb(None)

    m.fs.absorber.liquid_properties[0, i].flow_mol.setlb(None)
    m.fs.afs.stripper.liquid_properties[0, i].flow_mol.setlb(None)
    m.fs.afs.stripper.vapor_properties[0, i].flow_mol.setlb(None)

    m.fs.absorber.liquid_properties[0, i].temperature.setlb(None)
    m.fs.absorber.liquid_properties[0, i].temperature.setub(None)
    m.fs.afs.stripper.liquid_properties[0, i].temperature.setlb(None)
    m.fs.afs.stripper.liquid_properties[0, i].temperature.setub(None)
    m.fs.absorber.vapor_properties[0, i].temperature.setlb(None)
    m.fs.absorber.vapor_properties[0, i].temperature.setub(None)
    m.fs.afs.stripper.vapor_properties[0, i].temperature.setlb(None)
    m.fs.afs.stripper.vapor_properties[0, i].temperature.setub(None)

    m.fs.absorber.vapor_properties[0, i].pressure.setlb(None)
    m.fs.absorber.vapor_properties[0, i].pressure.setub(None)
    m.fs.afs.stripper.vapor_properties[0, i].pressure.setlb(None)
    m.fs.afs.stripper.vapor_properties[0, i].pressure.setub(None)

    m.fs.absorber.vapor_properties[0, i].cp_vol.setlb(None)

    for k in range(1, 8):
        m.fs.absorber.liquid_properties[0, i].log_k_eq[k].setub(None)
        m.fs.afs.stripper.liquid_properties[0, i].log_k_eq[k].setub(None)

    for j in ["CO2", "H2O", "PZ"]:
        m.fs.absorber.material_transfer_coefficient_gas[j, i].setlb(None)
        m.fs.afs.stripper.material_transfer_coefficient_gas[j, i].setlb(None)
        m.fs.absorber.material_transfer_coefficient_tot[j, i].setlb(None)
        m.fs.afs.stripper.material_transfer_coefficient_tot[j, i].setlb(None)

        m.fs.absorber.liquid_properties[0, i].equilibrium_pressure[j].setlb(None)
        m.fs.afs.stripper.liquid_properties[0, i].equilibrium_pressure[j].setlb(None)
        m.fs.afs.stripper.liquid_properties[0, i].flow_mol_comp[j].setlb(None)
        m.fs.afs.stripper.liquid_properties[0, i].mole_frac_comp[j].setlb(None)

        m.fs.afs.stripper.vapor_properties[0, i].pressure_comp[j].setlb(None)
        m.fs.absorber.vapor_properties[0, i].diffus_comp[j].setlb(None)
        m.fs.afs.stripper.vapor_properties[0, i].diffus_comp[j].setlb(None)
        m.fs.afs.stripper.vapor_properties[0, i].conc_mol_comp[j].setlb(None)
        m.fs.afs.stripper.vapor_properties[0, i].mole_frac_comp[j].setlb(None)
        m.fs.afs.stripper.vapor_properties[0, i].flow_mol_comp[j].setlb(None)

    for j in ["CO2", "H2O", "N2"]:
        m.fs.absorber.vapor_properties[0, i].cp_mol_comp[j].setlb(None)
        m.fs.absorber.vapor_properties[0, i].flow_mol_comp[j].setlb(None)

    for j in ["CO2", "H2O"]:
        m.fs.afs.stripper.vapor_properties[0, i].cp_mol_comp[j].setlb(None)

    for j in ["CO2", "H2O", "PZ", "N2"]:
        m.fs.absorber.liquid_properties[0, i].mole_frac_comp[j].setlb(None)
        m.fs.absorber.liquid_properties[0, i].flow_mol_comp[j].setlb(None)
        m.fs.absorber.vapor_properties[0, i].pressure_comp[j].setlb(None)
        m.fs.absorber.vapor_properties[0, i].conc_mol_comp[j].setlb(None)
        m.fs.absorber.vapor_properties[0, i].mole_frac_comp[j].setlb(None)

    for j in speciation_components:
        m.fs.absorber.liquid_properties[0, i].concentration_spec[j].setlb(None)
        m.fs.afs.stripper.liquid_properties[0, i].concentration_spec[j].setlb(None)
        m.fs.absorber.liquid_properties[0, i].ln_concentration_spec[j].setlb(None)
        m.fs.afs.stripper.liquid_properties[0, i].ln_concentration_spec[j].setlb(None)

results = local_solver.solve(m.fs)
print("remove bounds: " + results.solver.termination_condition)

for i in range(0, 41):
    m.fs.afs.stripper.vapor_properties[0, i].cp_vol.setlb(None)
    m.fs.absorber.vapor_properties[0, i].visc_d.setlb(None)
    m.fs.afs.stripper.vapor_properties[0, i].visc_d.setlb(None)
    for j in ["CO2", "H2O"]:
        m.fs.afs.stripper.vapor_properties[0, i].viscosity_comp[j].setlb(None)
    for j in ["CO2", "H2O", "N2"]:
        m.fs.absorber.vapor_properties[0, i].viscosity_comp[j].setlb(None)
    m.fs.absorber.vapor_properties[0, i].flow_mol_comp["PZ"].setlb(None)

results = local_solver.solve(m.fs)
print("remove bounds: " + results.solver.termination_condition)


# ==============================================================
# Uncertain parameters

kref = m.fs.absorber.liquid_properties.params.reference_rate
kref_nominal = 53.7
kref_sd = 1.5 

r3 = m.fs.absorber.liquid_properties.params.dens_mass_param_3
r3_nominal = 0.991
r3_sd = 0.002

henry1 = m.fs.absorber.liquid_properties.params.henry_param_1
henry1_nominal = 1.7107 * 10**7
henry2 = m.fs.absorber.liquid_properties.params.henry_param_2
henry2_nominal = -1886.1
henry_cov = [[3.33e+12, -6.1e+07], [-6.1e+07, 1.124e+03]]

diff1 = m.fs.absorber.liquid_properties.params.diffusivity_param_1
diff1_nominal = 0.024 * 10 ** (-4)
diff2 = m.fs.absorber.liquid_properties.params.diffusivity_param_2
diff2_nominal = -2122
diff_cov = [[6.57e-14, -8.63e-06], [-8.63e-06, 1.14e+03]]

Habs = m.fs.afs.stripper.liquid_properties.params.Habs_param_1
Habs_nominal = -11054
Habs_sd = 120

Peq_param1 = (
    m.fs.afs.flash_tank.liquid_properties.params.equilibrium_pressure_CO2_param_1
)
Peq_param1_nominal = 35.3
Peq_param1_sd = 0.3

Peq_param2 = (
    m.fs.afs.flash_tank.liquid_properties.params.equilibrium_pressure_CO2_param_2
)
Peq_param2_nominal = -11054
Peq_param2_sd = 120


Peq_param3 = (
    m.fs.afs.flash_tank.liquid_properties.params.equilibrium_pressure_CO2_param_3
)
Peq_param3_nominal = -18.9
Peq_param3_sd = 2.7

Peq_param4 = (
    m.fs.afs.flash_tank.liquid_properties.params.equilibrium_pressure_CO2_param_4
)
Peq_param4_nominal = 4958
Peq_param4_sd = 347

Peq_param5 = (
    m.fs.afs.flash_tank.liquid_properties.params.equilibrium_pressure_CO2_param_5
)
Peq_param5_nominal = 10163
Peq_param5_sd = 1085
# ==============================================================

# Example set
uncertain_parameters = [
     henry1,
     henry2,
]

ellipsoid_set_henry = pyros.EllipsoidalSet(
  center=[henry1_nominal, henry2_nominal],
  shape_matrix=henry_cov,
  scale = None,
  gaussian_conf_lvl = 0.95
)
print(ellipsoid_set_henry.parameter_bounds)


#################################################################################################################
# Other set examples
#bounds = [
#     (kref_nominal - kref_sd, kref_nominal + kref_sd),
#     (r3_nominal - r3_sd, r3_nominal + r3_sd),
#    (Habs_nominal - Habs_sd, Habs_nominal + Habs_sd),
#]
#
#box_uncertainty_set = pyros.BoxSet(bounds=bounds)
#print(box_uncertainty_set.bounds)
#
#ellipsoid_set_diff = pyros.EllipsoidalSet(
#  center=[diff1_nominal, diff2_nominal],
#  shape_matrix=diff_cov,
#  scale=None,
#  gaussian_conf_lvl = 0.99
#)
#print(ellipsoid_set_diff.parameter_bounds)

#axisalligned_uncertainty_set = pyros.AxisAlignedEllipsoidalSet(
#   center=[Peq_param1_nominal, Peq_param2_nominal, Peq_param3_nominal, Peq_param4_nominal, Peq_param5_nominal],
#   half_lengths=[Peq_param1_sd, Peq_param2_sd, Peq_param3_sd, Peq_param4_sd, Peq_param5_sd]
#)
#
#################################################################################################################

first_stage_variables = [
    m.fs.absorber.length,
    m.fs.absorber.diameter,
    m.fs.afs.stripper.length,
    m.fs.afs.stripper.diameter,
    m.fs.A1,
    m.fs.A2,
    m.fs.A3,
    m.fs.A_intercooler,
    m.fs.A_cooler,
    m.fs.A_heater,
    m.fs.flash_tank_volume,
    m.fs.condenser_tank_volume,
    m.fs.pump_1_s,
    m.fs.pump_2_s,
    m.fs.pump_1_P_c,
    m.fs.pump_2_P_c,
]

second_stage_variables = [
    m.fs.cw_cooler,
    m.fs.cw_intercooler,
    m.fs.steam_flowrate,
    m.fs.M_w,
    m.fs.M_pz,
    m.fs.bypass1,
    m.fs.bypass2,
]

# logger (optional)
pyros_logger = logging.getLogger('pyomo.contrib.pyros')
pyros_logger.setLevel(logging.DEBUG)

sep_dict = {'fs.pump_1_P_c_design_con': 1}  # constraint(s) to be prioritized per case

results = pyros_solver.solve(
    model=m,
    first_stage_variables=first_stage_variables,
    second_stage_variables=second_stage_variables,
    uncertain_params=uncertain_parameters,
    uncertainty_set=ellipsoid_set_henry,
    local_solver=local_solver,
    backup_local_solvers=[l3, l1, l2, l4],
    global_solver=global_solver,
    objective_focus=pyros.ObjectiveType.nominal,
    separation_priority_order=sep_dict,           
    bypass_global_separation=True,
    solve_master_globally=False,
    decision_rule_order=0,
)

# Query results

time = results.time

iterations = results.iterations

termination_condition = results.pyros_termination_condition

objective = results.final_objective_value

# Print results

single_stage_final_objective = round(objective, -1)

print(f"Final objective value: {single_stage_final_objective}")

print(f"PyROS termination condition: {termination_condition}")

print("Design Variables")
print(" Absorber packed length (m): " + str(round(m.fs.absorber.length.value, 2)))
print(" Absorber diameter (m): " + str(round(m.fs.absorber.diameter.value, 2)))
print(
    " AFS length (stripper packed length + flash sump) (m): "
    + str(
        round(
            m.fs.afs.stripper.length.value
            + m.fs.L_flash_tank.value * 0.0254,
            2,
        )
    )
)
print(" AFS diameter (m): " + str(round(m.fs.afs.stripper.diameter.value, 2)))
print(" Condenser tank volume (m3): " + str(round(m.fs.condenser_tank_volume.value, 1)))
print(" Area of exchanger #1 (m2): " + str(round(m.fs.A1.value, 1)))
print(" Area of exchanger #2 (m2): " + str(round(m.fs.A2.value, 1)))
print(" Area of exchanger #3 (m2): " + str(round(m.fs.A3.value, 1)))
print(" Area of intercooler (m2): " + str(round(m.fs.A_intercooler.value, 1)))
print(" Area of cooler (m2): " + str(round(m.fs.A_cooler.value, 1)))
print(" Area of heater (m2): " + str(round(m.fs.A_heater.value, 1)))
print(" Pump 1 size factor ((gpm)(ft)^0.5): " + str(round(m.fs.pump_1_s.value, 1)))
print(" Pump 1 motor capacity (Hp): " + str(round(m.fs.pump_1_P_c.value, 1)))
print(" Pump 2 size factor ((gpm)(ft)^0.5): " + str(round(m.fs.pump_2_s.value, 1)))
print(" Pump 2 motor capacity (Hp): " + str(round(m.fs.pump_2_P_c.value, 1)))

print("Operational Variables")
print(" Cooling water for lean cooler (kg/s): " + str(round(m.fs.cw_cooler.value, 1)))
print(" Cooling water for intercooler (kg/s): " + str(round(m.fs.cw_intercooler.value, 1)))
print(" Steam for heater (kg/s): " + str(round(m.fs.steam_flowrate.value, 1)))
print(" Solvent flowrate (mol/s): " + str(round(m.fs.F_l_prime.value, 1)))
print("  Make-up of water (mol/s): " + str(round(m.fs.M_w.value, 2)))
print("  Make-up of PZ (mol/s): " + str(round(m.fs.M_pz.value, 2)))
print(" Cold bypass (%): " + str(round(m.fs.bypass1.value * 100, 2)))
print(" Warm bypass (%): " + str(round(m.fs.bypass2.value * 100, 2)))

print("Costs")
print(' TAC (single train) ', round(m.fs.total_cost.value, 1))
print('  Annualized capex ', round(value(m.fs.TASC.expr), 1))
print('  Variable opex ', round(m.fs.variable_OMC.value, 1))
print('  Fixed opex ', round(value(m.fs.fixed_OMC.expr), 1))
print(' CC: ',
    round(
        m.fs.total_cost.value
        / m.fs.afs.stripper.vapor_properties[0, 40].flow_mol_comp["CO2"].value
        / 8000
        / 3600
        / 44
        * (10**6),
        3,
    )
)
print(
    " Capture: ",
    round(
        100
        * (m.fs.afs.stripper.vapor_properties[0, 40].flow_mol_comp["CO2"].value)
        / (m.fs.absorber.vapor_properties[0, 0].flow_mol_comp["CO2"].value),
        2,
    ),
)
print(' Cost of target capture: ', round(m.fs.total_cost.value/(584.25 * 8000 * 3600 * 44 / (10**6)), 3))

# Check bounds
positive_vars = [
    m.fs.F_bottom,
    m.fs.F_hot,
    m.fs.F_cold,
    m.fs.F_warm,
    m.fs.F_l_prime,
    m.fs.afs.flash_tank.K_flash['CO2'],
    m.fs.afs.flash_tank.K_flash['PZ'],
    m.fs.afs.flash_tank.K_flash['H2O'],
    m.fs.afs.flash_tank.feed_properties[0].flow_mol,
    m.fs.afs.flash_tank.liquid_properties[0].flow_mol,
    m.fs.afs.flash_tank.vapor_properties[0].flow_mol,
]

for i in range(0, 41):
    positive_vars.append(m.fs.absorber.effective_vapor_velocity[i])
    positive_vars.append(m.fs.absorber.effective_liquid_velocity[i])
    positive_vars.append(m.fs.afs.stripper.effective_vapor_velocity[i])
    positive_vars.append(m.fs.afs.stripper.effective_liquid_velocity[i])
    positive_vars.append(m.fs.absorber.heat_transfer_coefficient[i])
    positive_vars.append(m.fs.afs.stripper.heat_transfer_coefficient[i])
    positive_vars.append(m.fs.absorber.vapor_velocity[i])
    positive_vars.append(m.fs.absorber.liquid_velocity[i])
    positive_vars.append(m.fs.afs.stripper.vapor_velocity[i])
    positive_vars.append(m.fs.afs.stripper.liquid_velocity[i])
    positive_vars.append(m.fs.absorber.liquid_properties[0, i].reaction_rate)
    positive_vars.append(m.fs.afs.stripper.liquid_properties[0, i].reaction_rate)
    positive_vars.append(m.fs.absorber.liquid_properties[0, i].henry)
    positive_vars.append(m.fs.afs.stripper.liquid_properties[0, i].henry)
    positive_vars.append(m.fs.absorber.liquid_properties[0, i].diffusivity_cst)
    positive_vars.append(m.fs.afs.stripper.liquid_properties[0, i].diffusivity_cst)
    positive_vars.append(m.fs.absorber.liquid_properties[0, i].viscosity_water)
    positive_vars.append(m.fs.afs.stripper.liquid_properties[0, i].viscosity_water)
    positive_vars.append(m.fs.absorber.liquid_properties[0, i].visc_d)
    positive_vars.append(m.fs.afs.stripper.liquid_properties[0, i].visc_d)
    positive_vars.append(m.fs.absorber.vapor_properties[0, i].conc_mol)
    positive_vars.append(m.fs.afs.stripper.vapor_properties[0, i].conc_mol)
    positive_vars.append(m.fs.absorber.vapor_properties[0, i].visc_d)
    positive_vars.append(m.fs.afs.stripper.vapor_properties[0, i].visc_d)

    for j in ["CO2", "H2O", "PZ"]:

        positive_vars.append(m.fs.absorber.material_transfer_coefficient_gas[j, i])
        positive_vars.append(m.fs.afs.stripper.material_transfer_coefficient_gas[j, i])
        positive_vars.append(m.fs.absorber.material_transfer_coefficient_tot[j, i])
        positive_vars.append(m.fs.afs.stripper.material_transfer_coefficient_tot[j, i])
        positive_vars.append(m.fs.absorber.liquid_properties[0, i].equilibrium_pressure[j])
        positive_vars.append(m.fs.afs.stripper.liquid_properties[0, i].equilibrium_pressure[j])
        positive_vars.append(m.fs.afs.stripper.liquid_properties[0, i].flow_mol_comp[j])
        positive_vars.append(m.fs.afs.stripper.liquid_properties[0, i].mole_frac_comp[j])

        positive_vars.append(m.fs.afs.stripper.vapor_properties[0, i].pressure_comp[j])
        positive_vars.append(m.fs.absorber.vapor_properties[0, i].diffus_comp[j])
        positive_vars.append(m.fs.afs.stripper.vapor_properties[0, i].diffus_comp[j])
        positive_vars.append(m.fs.afs.stripper.vapor_properties[0, i].conc_mol_comp[j])
        positive_vars.append(m.fs.afs.stripper.vapor_properties[0, i].mole_frac_comp[j])
        positive_vars.append(m.fs.afs.stripper.vapor_properties[0, i].flow_mol_comp[j])
    for j in ["CO2", "H2O"]:
        positive_vars.append(m.fs.afs.stripper.vapor_properties[0, i].viscosity_comp[j])
    for j in ["CO2", "H2O", "N2"]:
        positive_vars.append(m.fs.absorber.vapor_properties[0, i].viscosity_comp[j])
    positive_vars.append(m.fs.absorber.vapor_properties[0, i].flow_mol_comp["PZ"])
    for j in speciation_components:
        positive_vars.append(m.fs.absorber.liquid_properties[0, i].concentration_spec[j])
        positive_vars.append(m.fs.afs.stripper.liquid_properties[0, i].concentration_spec[j])

    for j in ["CO2", "H2O", "N2"]:
        positive_vars.append(m.fs.absorber.vapor_properties[0, i].cp_mol_comp[j])
        positive_vars.append(m.fs.absorber.vapor_properties[0, i].flow_mol_comp[j])

    for j in ["CO2", "H2O"]:
        positive_vars.append(m.fs.afs.stripper.vapor_properties[0, i].cp_mol_comp[j])

    for j in ["CO2", "H2O", "PZ", "N2"]:
        positive_vars.append(m.fs.absorber.liquid_properties[0, i].flow_mol_comp[j])
        positive_vars.append(m.fs.absorber.vapor_properties[0, i].pressure_comp[j])
        positive_vars.append(m.fs.absorber.vapor_properties[0, i].conc_mol_comp[j])

for v in positive_vars:
    if v.value != None:
        if v.value < 0:
            print(v.name + ' less than 0')

mole_frac_vars = [
    m.fs.afs.flash_tank.feed_properties[0].mole_frac_comp['CO2'],
    m.fs.afs.flash_tank.vapor_properties[0].mole_frac_comp['CO2'],
    m.fs.afs.flash_tank.liquid_properties[0].mole_frac_comp['CO2'],
    m.fs.afs.flash_tank.feed_properties[0].mole_frac_comp['H2O'],
    m.fs.afs.flash_tank.vapor_properties[0].mole_frac_comp['H2O'],
    m.fs.afs.flash_tank.liquid_properties[0].mole_frac_comp['H2O'],
    m.fs.afs.flash_tank.feed_properties[0].mole_frac_comp['PZ'],
    m.fs.afs.flash_tank.vapor_properties[0].mole_frac_comp['PZ'],
    m.fs.afs.flash_tank.liquid_properties[0].mole_frac_comp['PZ'],
]
for i in range(0, 41):
    for j in ["CO2", "H2O", "PZ", "N2"]:
        mole_frac_vars.append(m.fs.absorber.liquid_properties[0, i].mole_frac_comp[j])
        mole_frac_vars.append(m.fs.absorber.vapor_properties[0, i].mole_frac_comp[j])

for v in mole_frac_vars:
    if v.value > 1:
        print(v.name + ' greater than 1')
    elif v.value < 0:
        print(v.name + ' less than 0')

temperature_vars = [
    m.fs.afs.flash_tank.feed_properties[0].temperature,
    m.fs.afs.flash_tank.liquid_properties[0].temperature,
    m.fs.afs.flash_tank.vapor_properties[0].temperature
]
for i in range(0, 41):
    temperature_vars.append(m.fs.absorber.liquid_properties[0, i].temperature)
    temperature_vars.append(m.fs.absorber.liquid_properties[0, i].temperature)
    temperature_vars.append(m.fs.afs.stripper.liquid_properties[0, i].temperature)
    temperature_vars.append(m.fs.afs.stripper.liquid_properties[0, i].temperature)
    temperature_vars.append(m.fs.absorber.vapor_properties[0, i].temperature)
    temperature_vars.append(m.fs.absorber.vapor_properties[0, i].temperature)
    temperature_vars.append(m.fs.afs.stripper.vapor_properties[0, i].temperature)
    temperature_vars.append(m.fs.afs.stripper.vapor_properties[0, i].temperature)

for v in temperature_vars:
    if v.value > (156+273):
        print(v.name + ' larger than 156C')
    elif v.value < (20+273):
        print(v.name + ' less than 20C')

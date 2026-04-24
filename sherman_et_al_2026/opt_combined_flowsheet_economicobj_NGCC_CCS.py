# Python imports
import logging
import sys
import time

# Pyomo imports
import pyomo.environ as pyo
from pyomo.util.calc_var_value import calculate_variable_from_constraint

from idaes.core.solvers.get_solver import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale
import idaes.core.util.model_serializer as ms
from idaes.core.util.constants import Constants

# External IDAES imports (workspace)
sys.path.insert(0, 'flowsheets')
from mea_properties import (
    switch_liquid_to_parmest_params
)

from combined_flowsheet_withcosting_NGCC_CCS import (MEACombinedFlowsheet, 
                                            MEACombinedFlowsheetData)

from monkey_patch_costing import monkey_patch_costing
monkey_patch_costing()

def deterministic_optimization_problem(fs, target_CO2_capture):
    
    # Column height to diameter ratio
    fs.absorber_section.absorber.HDratio_lower_bound = pyo.Constraint(
        expr=1.2*fs.absorber_section.absorber.diameter_column <= 
        fs.absorber_section.absorber.length_column)
    fs.absorber_section.absorber.HDratio_upper_bound = pyo.Constraint(
        expr=30*fs.absorber_section.absorber.diameter_column >= 
        fs.absorber_section.absorber.length_column)
    
    fs.stripper_section.stripper.HDratio_lower_bound = pyo.Constraint(
        expr=1.2*fs.stripper_section.stripper.diameter_column <= 
        fs.stripper_section.stripper.length_column)
    fs.stripper_section.stripper.HDratio_upper_bound = pyo.Constraint(
        expr=30*fs.stripper_section.stripper.diameter_column >= 
        fs.stripper_section.stripper.length_column)
    
    # Flooding fraction constraints
    @fs.absorber_section.absorber.Constraint(
        fs.time, 
        fs.absorber_section.absorber.vapor_phase.length_domain)
    def LB_flood_ratio(b, t, x):
        if x == b.vapor_phase.length_domain.first():
            return pyo.Constraint.Skip
        return b.flood_fraction[t, x] >= 0.5
    
    @fs.absorber_section.absorber.Constraint(
        fs.time, 
        fs.absorber_section.absorber.vapor_phase.length_domain)
    def UB_flood_ratio(b, t, x):
        if x == b.vapor_phase.length_domain.first():
            return pyo.Constraint.Skip
        return b.flood_fraction[t, x] <= 0.8

    @fs.stripper_section.stripper.Constraint(
        fs.time, 
        fs.stripper_section.stripper.vapor_phase.length_domain)
    def LB_flood_ratio(b ,t ,x):
        if x == b.vapor_phase.length_domain.first():
            return pyo.Constraint.Skip
        return b.flood_fraction[t, x] >= 0.5

    @fs.stripper_section.stripper.Constraint(
        fs.time, 
        fs.stripper_section.stripper.vapor_phase.length_domain)
    def UB_flood_ratio(b ,t ,x):
        if x == b.vapor_phase.length_domain.first():
            return pyo.Constraint.Skip
        return b.flood_fraction[t, x] <= 0.8
    
    fs.target_CO2_capture = pyo.Param(
        mutable=True,
        initialize=target_CO2_capture
    )

    # CO2 capture rate constraint
    fs.CO2_lower_bound = pyo.Constraint(
        expr=fs.absorber_section.absorber.co2_capture[0] >= fs.target_CO2_capture)
    iscale.constraint_scaling_transform(
      fs.CO2_lower_bound, 1e-2
    )
    
    fs.stripper_section.reboiler.bottoms.temperature.unfix()
    fs.stripper_section.condenser.reflux.temperature.unfix()

    # Unfix variables (DOFs)
    design_variables = [
            fs.absorber_section.absorber.length_column,
            fs.absorber_section.absorber.diameter_column,
            fs.stripper_section.stripper.length_column,
            fs.stripper_section.stripper.diameter_column,
            fs.absorber_section.lean_rich_heat_exchanger.area,
            fs.stripper_section.condenser.area,
            fs.stripper_section.reboiler.area,
    ]
    control_variables = [
        fs.mea_recirculation_rate,
        fs.makeup.flow_mol,
        fs.stripper_section.reboiler.steam_flow_mol,
        fs.stripper_section.condenser.cooling_flow_mol,
    ]
    fixed_state_variables = [
        fs.h2o_mea_ratio,
        fs.stripper_section.reboiler.condensate_temperature_outlet,
        fs.stripper_section.condenser.cooling_water_temperature_outlet,
    ]
    
    for var in design_variables + control_variables:
        var.unfix()
    
    fs.h2o_mea_ratio.fix() # Fixed to 7.91 to maintain a 30 wt% MEA solution
    
    # Condenser outlet (distillate) temperature should be less than 40 C   
    # When including fs.stripper_section.condenser.vapor_phase.properties_out[0].mole_frac_comp["H2O"]
    # in the objective function, this constraint should be deactivated    
    @fs.stripper_section.Constraint(fs.time)
    def stripper_condenser_outlet_temperature(b ,t):
        return b.condenser.vapor_outlet.temperature[t] <= (40 + 273.15)
    
    # Numerator of Levelized Cost of Capture (equivalent to a levelized cost)
    fs.obj = pyo.Objective(expr = fs.costing.cost_of_capture 
                            * pyo.units.convert(fs.costing_setup.CO2_capture_rate[0],
                                                to_units=pyo.units.tonne / pyo.units.hour) 
                            * 1e-4, 
                            sense = pyo.minimize) # LCOC, $/tonne of CO2 captured

    return design_variables, control_variables, fixed_state_variables


def plot_results(fs):
    import matplotlib.pyplot as plt    
    
    # Temperature profile(s)
    Tliq = []
    Tvap = []
    for x in fs.vapor_phase.length_domain:
        Tvap.append(pyo.value(fs.vapor_phase.properties[0,x].temperature - 273.15))
    for x in fs.liquid_phase.length_domain:
        Tliq.append(pyo.value(fs.liquid_phase.properties[0,x].temperature - 273.15))
                                      
    plt.figure()
    plt.plot(fs.liquid_phase.length_domain, Tliq, 
             label='$T_{liquid}$', 
             linestyle='-',
             color='tab:blue')
    plt.plot(fs.vapor_phase.length_domain, Tvap, 
             label='$T_{vapor}$', 
             linestyle='--',
             color='tab:orange')
    plt.legend(loc='best',ncol=1)
    plt.grid()
    plt.title(f'{fs}'), 
    plt.xlabel("Normalized bed height [-]")
    plt.ylabel("Temperature [°C]") 
    
    # CO2 loading profile
    loading_CO2 = []
    for x in fs.liquid_phase.length_domain:
        loading_CO2.append(pyo.value(fs.liquid_phase.properties[0,x].mole_frac_comp["CO2"] /
                                     fs.liquid_phase.properties[0,x].mole_frac_comp["MEA"]))
                                      
    plt.figure()
    plt.plot(fs.liquid_phase.length_domain, loading_CO2, 
             label='$CO_2$', 
             linestyle='-',
             color='xkcd:grass')
    plt.legend(loc='best',ncol=1)
    plt.grid()
    plt.title(f'{fs}'), 
    plt.xlabel("Normalized bed height [-]")
    plt.ylabel("Loading [mol CO2/ mol MEA]") 
    
    # Vapor phase partial pressure of CO2
    PpCO2_vap = []
    PpCO2_liq = []
    for x in fs.vapor_phase.length_domain:
        PpCO2_vap.append(pyo.value(fs.vapor_phase.properties[0,x].pressure *
                               fs.vapor_phase.properties[0,x].mole_frac_comp['CO2'] *
                               1e-3))
    for x in fs.liquid_phase.length_domain:
        PpCO2_liq.append(pyo.value(fs.liquid_phase.properties[0,x].henry['Liq','CO2'] *
                               fs.liquid_phase.properties[0,x].conc_mol_phase_comp_true['Liq','CO2'] *
                               1e-3))
        
    plt.figure()
    plt.plot(fs.vapor_phase.length_domain, PpCO2_vap, 
             label='$P_{CO2}$', 
             linestyle='-',
             color='tab:gray')
    plt.plot(fs.liquid_phase.length_domain, PpCO2_liq, 
             label='$P^{ *}_{CO2}$', 
             linestyle='--',
             color='tab:gray')
    plt.legend(loc='best',ncol=1)
    plt.grid()
    plt.title(f'{fs}'), 
    plt.xlabel("Normalized bed height [-]")
    plt.ylabel("CO2 partial pressure [kPa]") 
    
    # Flood fraction profile
    flooding = []
    LB = []
    UB = []
    for x in fs.vapor_phase.length_domain:
        flooding.append(pyo.value(fs.flood_fraction[0,x]))
        LB.append(0.5)
        UB.append(0.8)
    
    plt.figure()
    plt.plot(fs.vapor_phase.length_domain, flooding, 
              linestyle='None',
              marker='.',
              color='k')
    plt.plot(fs.vapor_phase.length_domain, LB, 
              label='lower limit', 
              linestyle=':',
              color='tab:red')
    plt.plot(fs.vapor_phase.length_domain, UB, 
              label='upper limit', 
              linestyle='--',
              color='tab:red')
    plt.legend(loc='best',ncol=1)
    plt.grid()
    plt.title(f'{fs}'), 
    plt.xlabel("Normalized bed height [-]")
    plt.ylabel("Flooding fraction [-]") 
    
    plt.show()


if __name__ == "__main__":
    logging.getLogger('pyomo.repn.plugins.nl_writer').setLevel(logging.ERROR)
    _log = idaeslog.getLogger("mea_flowsheet")
    # Create finite element set for absorber and stripper
    nfe = 40
    grid = [i / nfe for i in range(nfe + 1)]

    # Build flowsheet
    m = pyo.ConcreteModel()
    m.fs = MEACombinedFlowsheet(
        time_set=[0],
        absorber_finite_element_set=grid,
        stripper_finite_element_set=grid,
    )

    switch_liquid_to_parmest_params(m.fs.stripper_section.liquid_properties, ions=True)
    switch_liquid_to_parmest_params(m.fs.stripper_section.liquid_properties_no_ions, ions=False)
    switch_liquid_to_parmest_params(m.fs.absorber_section.liquid_properties, ions=True)
    switch_liquid_to_parmest_params(m.fs.absorber_section.liquid_properties_no_ions, ions=False)

    ts_init = time.time()
    # Set Initial guesses for inlets to sub-flowsheets
    m.fs.absorber_section.makeup_mixer.h2o_makeup.flow_mol.fix(1300)  # mol/sec
    iscale.calculate_scaling_factors(m.fs)
    m.fs.initialize_build(
        outlvl=idaeslog.DEBUG,
        optarg={
            'nlp_scaling_method': 'user-scaling',
            'linear_solver': 'ma57',
            'OF_ma57_automatic_scaling': 'yes',
            'max_iter': 300,
            'tol': 1e-8,
            'halt_on_ampl_error': 'no',
        },
    )
    
    print("\n")
    print("----------------------------------------------------------")
    print('Total initialization time: ', pyo.value(time.time() - ts_init), " s")
    print("----------------------------------------------------------")
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ SIMULATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Solve flowsheet, to a better initial point for optimization
    # Also add condenser and reboiler area calculation constraints
    
    _log.info("Increasing column diameters")
    
    m.fs.absorber_section.absorber.diameter_column.fix(14)
    m.fs.stripper_section.stripper.diameter_column.fix(6.5) 
    
    optarg={
        'bound_push' : 1e-4,
        'nlp_scaling_method': 'user-scaling',
        'linear_solver': 'ma57',
        'OF_ma57_automatic_scaling': 'yes',
        'max_iter': 1000,
        'constr_viol_tol': 1e-8,
        'halt_on_ampl_error': 'no',
        "mu_init": 10**-2,
    }
    solver_obj = get_solver("ipopt", optarg)

    results = solver_obj.solve(m, tee=True)
    pyo.assert_optimal_termination(results)
    
    _log.info("Adding heat exchanger areas")
    
    m.fs.stripper_section.add_condenser_reboiler_performance_equations()
    
    assert degrees_of_freedom(m) == 0
    results = solver_obj.solve(m, tee=True)
    pyo.assert_optimal_termination(results)
    
    print("\n-------- Absorber Simulation Results --------")
    m.fs.absorber_section.print_column_design_parameters()
    # # Plot column profiles
    
    print("\n-------- Stripper Simulation Results --------")
    m.fs.stripper_section.print_column_design_parameters()
    # # Plot column profiles
    print()
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ ADD COSTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    MEACombinedFlowsheetData.add_costing(m.fs)  # call costing module
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ OPTIMIZATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ts_opt = time.time()
    
    print("\n")
    _log.info("Build and solve optimization model")
    
    target_CO2_capture = 99
    
    design_variables, control_variables, fixed_state_variables = deterministic_optimization_problem(m.fs, target_CO2_capture)
    for var in design_variables:
        assert not var.fixed
    for var in control_variables:
        assert not var[0].fixed
    for var in fixed_state_variables:
        assert var[0].fixed
    
    assert len(design_variables + control_variables) - len(fixed_state_variables) == degrees_of_freedom(m)
    # Solve deterministic optimization model
    for t in m.fs.time:
        m.fs.absorber_section.lean_rich_heat_exchanger.delta_temperature_in[t].setlb(0)
        m.fs.absorber_section.lean_rich_heat_exchanger.delta_temperature_out[t].setlb(0)
    
    # Some Vars that are uninvolved in the NLP are uninitialized
    # However, this causes problems for the scaled model back propagation
    # so we want to initialize them with dummy values
    
    for var in m.component_data_objects(pyo.Var):
        if var.value is None:
            var.set_value(0)

    m_scaled = pyo.TransformationFactory('core.scale_model').create_using(m, rename=False)
    optarg.pop("nlp_scaling_method", None) # Scaled model doesn't need user scaling
    optarg["max_iter"] = 500
    solver_obj = get_solver(
        "ipopt",
        options=optarg
    )

    results = solver_obj.solve(m_scaled, tee=True)
    pyo.assert_optimal_termination(results)

    pyo.TransformationFactory('core.scale_model').propagate_solution(m_scaled, m)

    print("\n-------- Deterministic optimization Results --------") 
    print("\n***Absorber column***")
    m.fs.absorber_section.print_column_design_parameters()
    
    print("\n***Stripper column***")
    m.fs.stripper_section.print_column_design_parameters()
    
    print("\nFlowsheet, other")
    print("\nMEA recirculation rate: ", pyo.value(m.fs.mea_recirculation_rate[0]), "mol/s")
    print("Lean-rich HX area: ", pyo.value(m.fs.absorber_section.lean_rich_heat_exchanger.area), "m2")
    print("Lean-rich HX heat duty: ", pyo.value(m.fs.absorber_section.lean_rich_heat_exchanger.heat_duty[0]*1e-6), "MW")
    print("Condenser area: ", pyo.value(m.fs.stripper_section.condenser.area), "m2")
    print("Condenser heat duty: ", pyo.value(m.fs.stripper_section.condenser.heat_duty[0]*1e-6), "MW")
    print("Condenser cooling agent molar flow rate: ", pyo.value(m.fs.stripper_section.condenser.cooling_flow_mol[0]*1e-3), "kmol/s")
    print("Condenser vapor H2O mole fraction: ", pyo.value(m.fs.stripper_section.condenser.vapor_phase.properties_out[0].mole_frac_comp["H2O"]))
    print("Reboiler area: ", pyo.value(m.fs.stripper_section.reboiler.area), "m2")
    print("Reboiler heat duty: ", pyo.value(m.fs.stripper_section.reboiler.heat_duty[0]*1e-6), "MW")
    print("Reboiler steam consumption rate: ", pyo.value(m.fs.stripper_section.reboiler.steam_flow_mol[0]*1e-3), "kmol/s")
    print("Reboiler bottoms temperature: ", pyo.value(m.fs.stripper_section.reboiler.bottoms.temperature[0]), "K")
    print("\nObjective value: ", pyo.value(m.fs.obj))    
    print("Levelized Cost of Capture: ", pyo.value(m.fs.costing.cost_of_capture), "$/tonne CO2 captured")
    
    print("\n")
    print("----------------------------------------------------------")
    print('Total optimization time: ', pyo.value(time.time() - ts_opt), " s")
    print("----------------------------------------------------------")

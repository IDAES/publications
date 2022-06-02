import pyomo.environ as pyo
from idaes.core import FlowsheetBlock
from idaes.generic_models.properties import iapws95
import flowsheets.steam_cycle_subfs as stc
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.dyn_utils import copy_values_at_time, copy_non_time_indexed_values
import idaes.logger as idaeslog
import idaes.core.plugins
import idaes.core.util.scaling as iscale

def _new_solve(self, model, **kwargs):
    self.options["nlp_scaling_method"] = "user-scaling"
    self.options["linear_solver"] = "ma27"
    self.options["tol"] = 1e-6
    self.options['ma27_pivtol'] = 0.01
    self.options['ma27_pivtolmax'] = 0.6
    if kwargs["tee"]:
        print("THIS IPOPT SOLVER HAS BEEN MONKEY PATCHED FOR SCALING")
    #iscale.constraint_autoscale_large_jac(model, min_scale=1e-6)
    res = self._old_solve(model, **kwargs)
    return res

def monkey_patch_ipopt():
    from pyomo.solvers.plugins.solvers.IPOPT import IPOPT
    IPOPT._old_solve = IPOPT.solve
    IPOPT.solve = _new_solve

def undo_patch_ipopt():
    IPOPT.solve = IPOPT._old_solve

monkey_patch_ipopt()

def set_scaling_factors(m):
    """ Set scaling factors for variables and expressions. These are used for
    variable scaling and used by the framework to scale constraints.

    Args:
        m: plant model to set scaling factors for.

    Returns:
        None
    """

    # Set steam cycle scale factors
    fs = m.fs_main.fs_stc

    iscale.set_scaling_factor(fs.condenser.side_1.heat, 1e-9)
    iscale.set_scaling_factor(fs.condenser.side_2.heat, 1e-9)

    iscale.set_scaling_factor(fs.aux_condenser.side_1.heat, 1e-7)
    iscale.set_scaling_factor(fs.aux_condenser.side_2.heat, 1e-7)

    iscale.set_scaling_factor(fs.hotwell_tank.control_volume.energy_holdup, 1e-9)
    iscale.set_scaling_factor(fs.hotwell_tank.control_volume.material_holdup, 1e-6)

    iscale.set_scaling_factor(fs.fwh1.condense.side_1.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh1.condense.side_1.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh1.condense.side_2.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh1.condense.side_2.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh1.condense.side_1.heat, 1e-7)
    iscale.set_scaling_factor(fs.fwh1.condense.side_2.heat, 1e-7)

    iscale.set_scaling_factor(fs.fwh2.condense.side_1.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh2.condense.side_1.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh2.condense.side_2.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh2.condense.side_2.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh2.condense.side_1.heat, 1e-7)
    iscale.set_scaling_factor(fs.fwh2.condense.side_2.heat, 1e-7)

    iscale.set_scaling_factor(fs.fwh3.condense.side_1.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh3.condense.side_1.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh3.condense.side_2.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh3.condense.side_2.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh3.condense.side_1.heat, 1e-7)
    iscale.set_scaling_factor(fs.fwh3.condense.side_2.heat, 1e-7)

    iscale.set_scaling_factor(fs.da_tank.control_volume.energy_holdup, 1e-10)
    iscale.set_scaling_factor(fs.da_tank.control_volume.material_holdup, 1e-6)

    iscale.set_scaling_factor(fs.fwh5.condense.side_1.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh5.condense.side_1.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh5.condense.side_2.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh5.condense.side_2.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh5.condense.side_1.heat, 1e-7)
    iscale.set_scaling_factor(fs.fwh5.condense.side_2.heat, 1e-7)

    iscale.set_scaling_factor(fs.fwh6.condense.side_1.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh6.condense.side_1.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh6.condense.side_2.material_holdup, 1e-4)
    iscale.set_scaling_factor(fs.fwh6.condense.side_2.energy_holdup, 1e-8)
    iscale.set_scaling_factor(fs.fwh6.condense.side_1.heat, 1e-7)
    iscale.set_scaling_factor(fs.fwh6.condense.side_2.heat, 1e-7)

    # Calculate calculated scaling factors
    iscale.calculate_scaling_factors(m)

def main():
    m_ss = get_model(dynamic=False)
    #m_dyn = get_model(dynamic=True)
    #copy_non_time_indexed_values(m_dyn.fs_main, m_ss.fs_main, copy_fixed=True, outlvl=idaeslog.WARNING)
    #for t in m_dyn.fs_main.config.time:
    #    copy_values_at_time(m_dyn.fs_main, m_ss.fs_main, t, 0.0, copy_fixed=True, outlvl=idaeslog.WARNING)
    t0 = 0
    # estimate integral error for the PI controller
    '''
    m_dyn.fs_main.fs_stc.fwh2_ctrl.mv_ref.value = m_dyn.fs_main.fs_stc.fwh2_valve.valve_opening[t0].value
    m_dyn.fs_main.fs_stc.fwh3_ctrl.mv_ref.value = m_dyn.fs_main.fs_stc.fwh3_valve.valve_opening[t0].value
    m_dyn.fs_main.fs_stc.fwh5_ctrl.mv_ref.value = m_dyn.fs_main.fs_stc.fwh5_valve.valve_opening[t0].value
    m_dyn.fs_main.fs_stc.fwh6_ctrl.mv_ref.value = m_dyn.fs_main.fs_stc.fwh6_valve.valve_opening[t0].value
    m_dyn.fs_main.fs_stc.makeup_ctrl.mv_ref.value = m_dyn.fs_main.fs_stc.makeup_valve.valve_opening[t0].value
    m_dyn.fs_main.fs_stc.da_ctrl.mv_ref.value = m_dyn.fs_main.fs_stc.cond_valve.valve_opening[t0].value
    m_dyn.fs_main.fs_stc.spray_ctrl.mv_ref.value = m_dyn.fs_main.fs_stc.spray_valve.valve_opening[t0].value
    opening = m_dyn.fs_main.fs_stc.spray_valve.valve_opening[t0].value
    m_dyn.fs_main.fs_stc.spray_ctrl.integral_of_error[:].value = pyo.value(m_dyn.fs_main.fs_stc.spray_ctrl.integral_of_error_ref[t0])
    opening = m_dyn.fs_main.fs_stc.makeup_valve.valve_opening[t0].value
    m_dyn.fs_main.fs_stc.makeup_ctrl.integral_of_error[:].value = pyo.value(m_dyn.fs_main.fs_stc.makeup_ctrl.integral_of_error_ref[t0])
    '''

    m_dyn.fs_main.fs_stc.fwh2.condense.level[0].fix()
    m_dyn.fs_main.fs_stc.fwh3.condense.level[0].fix()
    m_dyn.fs_main.fs_stc.fwh5.condense.level[0].fix()
    m_dyn.fs_main.fs_stc.fwh6.condense.level[0].fix()
    m_dyn.fs_main.fs_stc.hotwell_tank.level[0].fix()
    m_dyn.fs_main.fs_stc.da_tank.level[0].fix()

    m_dyn.fs_main.fs_stc.spray_valve.valve_opening[0].fix()


    solver = pyo.SolverFactory("ipopt")
    solver.options = {
            "tol": 1e-7,
            "linear_solver": "ma27",
            "max_iter": 50,
    }

    dof = degrees_of_freedom(m_dyn.fs_main)
    print('dof of full model', dof)
    # solving dynamic model at steady-state
    print('solving dynamic model at steady-state...')
    results = solver.solve(m_dyn.fs_main, tee=True)
    
    print ('main steam enth=', pyo.value(m_dyn.fs_main.fs_stc.turb.inlet_split.inlet.enth_mol[0]))
    print ('main steam flow_mol=', pyo.value(m_dyn.fs_main.fs_stc.turb.inlet_split.inlet.flow_mol[0]))
    print ('main steam pressure=', pyo.value(m_dyn.fs_main.fs_stc.turb.inlet_split.inlet.pressure[0]))
    print ('throttle_valve Cv=', pyo.value(m_dyn.fs_main.fs_stc.turb.throttle_valve[1].Cv))
    print ('throttle_valve opening=', pyo.value(m_dyn.fs_main.fs_stc.turb.throttle_valve[1].valve_opening[0]))
    print ('HP stage 1 inlet enth_mol=', pyo.value(m_dyn.fs_main.fs_stc.turb.hp_stages[1].inlet.enth_mol[0]))
    print ('HP stage 1 inlet flow_mol=', pyo.value(m_dyn.fs_main.fs_stc.turb.hp_stages[1].inlet.flow_mol[0]))
    print ('HP stage 1 inlet pressure=', pyo.value(m_dyn.fs_main.fs_stc.turb.hp_stages[1].inlet.pressure[0]))
    print ('IP stage 1 inlet enth_mol=', pyo.value(m_dyn.fs_main.fs_stc.turb.ip_stages[1].inlet.enth_mol[0]))
    print ('IP stage 1 inlet flow_mol=', pyo.value(m_dyn.fs_main.fs_stc.turb.ip_stages[1].inlet.flow_mol[0]))
    print ('IP stage 1 inlet pressure=', pyo.value(m_dyn.fs_main.fs_stc.turb.ip_stages[1].inlet.pressure[0]))
    print ('Outlet stage enth_mol=', pyo.value(m_dyn.fs_main.fs_stc.turb.outlet_stage.outlet.enth_mol[0]))
    print ('Outlet stage flow_mol=', pyo.value(m_dyn.fs_main.fs_stc.turb.outlet_stage.outlet.flow_mol[0]))
    print ('Outlet stage pressure=', pyo.value(m_dyn.fs_main.fs_stc.turb.outlet_stage.outlet.pressure[0]))
    print ('Power output of main turbine=', pyo.value(m_dyn.fs_main.fs_stc.power_output[0]))
    print ('Power output of bfp turbine=', pyo.value(m_dyn.fs_main.fs_stc.bfp_turb.control_volume.work[0]))
    print ('FWH6 outlet enth_mol=', pyo.value(m_dyn.fs_main.fs_stc.fwh6.desuperheat.outlet_2.enth_mol[0]))
    print ('FWH6 outlet flow_mol=', pyo.value(m_dyn.fs_main.fs_stc.fwh6.desuperheat.outlet_2.flow_mol[0]))
    print ('FWH6 outlet pressure=', pyo.value(m_dyn.fs_main.fs_stc.fwh6.desuperheat.outlet_2.pressure[0]))
    print ('water makeup flow=', pyo.value(m_dyn.fs_main.fs_stc.condenser_hotwell.makeup.flow_mol[0]))
    print ('spray flow=', pyo.value(m_dyn.fs_main.fs_stc.spray_valve.outlet.flow_mol[0]))
    print('Cv fwh2=', m_dyn.fs_main.fs_stc.fwh2_valve.Cv.value)
    print('Cv fwh3=', m_dyn.fs_main.fs_stc.fwh3_valve.Cv.value)
    print('Cv fwh5=', m_dyn.fs_main.fs_stc.fwh5_valve.Cv.value)
    print('Cv fwh6=', m_dyn.fs_main.fs_stc.fwh6_valve.Cv.value)
    print('Cv cond_valve=', m_dyn.fs_main.fs_stc.cond_valve.Cv.value)
    print('Cv makeup_valve=', m_dyn.fs_main.fs_stc.makeup_valve.Cv.value)
    print('Cv hotwell_rejection_valve=', m_dyn.fs_main.fs_stc.hotwell_rejection_valve.Cv.value)
    print('Cv da_rejection_valve=', m_dyn.fs_main.fs_stc.da_rejection_valve.Cv.value)
    print('Cv spray_valve=', m_dyn.fs_main.fs_stc.spray_valve.Cv.value)
    print('valve opening fwh2=', m_dyn.fs_main.fs_stc.fwh2_valve.valve_opening[0].value)
    print('valve opening fwh3=', m_dyn.fs_main.fs_stc.fwh3_valve.valve_opening[0].value)
    print('valve opening fwh5=', m_dyn.fs_main.fs_stc.fwh5_valve.valve_opening[0].value)
    print('valve opening fwh6=', m_dyn.fs_main.fs_stc.fwh6_valve.valve_opening[0].value)
    print('valve opening cond_valve=', m_dyn.fs_main.fs_stc.cond_valve.valve_opening[0].value)
    print('valve opening makeup_valve=', m_dyn.fs_main.fs_stc.makeup_valve.valve_opening[0].value)
    print('valve opening hotwell_rejection_valve=', m_dyn.fs_main.fs_stc.hotwell_rejection_valve.valve_opening[0].value)
    print('valve opening da_rejection_valve=', m_dyn.fs_main.fs_stc.da_rejection_valve.valve_opening[0].value)
    print('valve opening spray=', m_dyn.fs_main.fs_stc.spray_valve.valve_opening[0].value)
    print('fwh2 level=', m_dyn.fs_main.fs_stc.fwh2.condense.level[0].value)
    print('fwh3 level=', m_dyn.fs_main.fs_stc.fwh3.condense.level[0].value)
    print('fwh5 level=', m_dyn.fs_main.fs_stc.fwh5.condense.level[0].value)
    print('fwh6 level=', m_dyn.fs_main.fs_stc.fwh6.condense.level[0].value)
    print('hotwell tank level=', m_dyn.fs_main.fs_stc.hotwell_tank.level[0].value)
    print('da tank level=', m_dyn.fs_main.fs_stc.da_tank.level[0].value)
    print('hotwell rejection flow=',  m_dyn.fs_main.fs_stc.hotwell_rejection_valve.outlet.flow_mol[0].value)
    print('da rejection flow=',  m_dyn.fs_main.fs_stc.da_rejection_valve.outlet.flow_mol[0].value)
    print('makeup flow=',  m_dyn.fs_main.fs_stc.makeup_valve.outlet.flow_mol[0].value)
    print('spray flow=', m_dyn.fs_main.fs_stc.spray_valve.outlet.flow_mol[0].value)
    print('cond split fraction 1=',  m_dyn.fs_main.fs_stc.cond_split.split_fraction[0,"outlet_1"].value)
    print('cond split fraction 2=',  m_dyn.fs_main.fs_stc.cond_split.split_fraction[0,"outlet_2"].value)
    print('cond split fraction 3=',  m_dyn.fs_main.fs_stc.cond_split.split_fraction[0,"outlet_3"].value) 
    print('split_attemp fraction=',  m_dyn.fs_main.fs_stc.split_attemp.split_fraction[0,"Spray"].value)

    # impose step change for dynamic model
    '''
    m_dyn.fs_main.fs_stc.fwh2.condense.level[0].fix()
    m_dyn.fs_main.fs_stc.fwh3.condense.level[0].fix()
    m_dyn.fs_main.fs_stc.fwh5.condense.level[0].fix()
    m_dyn.fs_main.fs_stc.fwh6.condense.level[0].fix()
    m_dyn.fs_main.fs_stc.hotwell_tank.level[0].fix()
    m_dyn.fs_main.fs_stc.da_tank.level[0].fix()
    m_dyn.fs_main.fs_stc.fwh2_valve.valve_opening[0].unfix()
    m_dyn.fs_main.fs_stc.fwh3_valve.valve_opening[0].unfix()
    m_dyn.fs_main.fs_stc.fwh5_valve.valve_opening[0].unfix()
    m_dyn.fs_main.fs_stc.fwh6_valve.valve_opening[0].unfix()
    m_dyn.fs_main.fs_stc.makeup_valve.valve_opening[0].unfix()
    m_dyn.fs_main.fs_stc.cond_valve.valve_opening[0].unfix()
    '''
    for t in m_dyn.fs_main.config.time:
        if t >= 10:
            m_dyn.fs_main.fs_stc.turb.ip_stages[1].inlet.pressure[t].fix(m_dyn.fs_main.fs_stc.turb.ip_stages[1].inlet.pressure[0].value*1.05)
        else:
            m_dyn.fs_main.fs_stc.turb.ip_stages[1].inlet.pressure[t].fix(m_dyn.fs_main.fs_stc.turb.ip_stages[1].inlet.pressure[0].value)

    dof = degrees_of_freedom(m_dyn.fs_main)
    print('dof of full model', dof)
    # solving dynamic model
    print('solving dynamic model...')
    results = solver.solve(m_dyn.fs_main, tee=True)
    print ('Power output of main turbine=', pyo.value(m_dyn.fs_main.fs_stc.power_output[100]))
    print ('Power output of bfp turbine=', pyo.value(m_dyn.fs_main.fs_stc.bfp_turb.control_volume.work[100]))
    print ('main steam enth=', pyo.value(m_dyn.fs_main.fs_stc.turb.inlet_split.inlet.enth_mol[100]))
    print ('main steam flow_mol=', pyo.value(m_dyn.fs_main.fs_stc.turb.inlet_split.inlet.flow_mol[100]))
    print ('main steam pressure=', pyo.value(m_dyn.fs_main.fs_stc.turb.inlet_split.inlet.pressure[100]))
    print ('fw pressure=', pyo.value(m_dyn.fs_main.fs_stc.bfp.outlet.pressure[100]))

    return m_dyn
      
def get_model(dynamic=True):
    m = pyo.ConcreteModel()
    m.dynamic = dynamic
    m.init_dyn = True
    if m.dynamic:
        m.fs_main = FlowsheetBlock(default={"dynamic": True, "time_set": [0, 10, 50]})
    else:
        m.fs_main = FlowsheetBlock(default={"dynamic": False})
    # Add property packages to flowsheet library
    m.fs_main.prop_water = iapws95.Iapws95ParameterBlock()
    m.fs_main.fs_stc = FlowsheetBlock()
    m = stc.add_unit_models(m)
    if m.dynamic:
        m.discretizer = pyo.TransformationFactory('dae.finite_difference')
        m.discretizer.apply_to(m,
                           nfe=3,
                           wrt=m.fs_main.time,
                           scheme="BACKWARD")
    m = stc.set_arcs_and_constraints(m)
    m = stc.set_inputs(m)
    set_scaling_factors(m)
    m = stc.initialize(m)
    return m

if __name__ == "__main__":
    m = main()
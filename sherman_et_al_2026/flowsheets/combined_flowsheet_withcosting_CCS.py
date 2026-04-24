"""
Costing for flowsheet consisting of absorber and stripper sections for 
Monoethanolamine solvent carbon capture system, one train
"""

# Python imports
import logging
import sys
import os

# Pyomo imports
import pyomo.environ as pyo
from pyomo.network import Arc, Port
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.config import ConfigValue, ListOf

# IDAES imports
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.core.util as iutil
from idaes.core import FlowsheetBlockData, declare_process_block_class

from idaes.models.unit_models import Translator
from idaes.models.unit_models.stream_scaler import StreamScaler

from util import fix_vars, restore_fixedness

from idaes.core.solvers.get_solver import get_solver
import idaes.logger as idaeslog
from idaes.core.util.initialization import propagate_state
import idaes.core.util.scaling as iscale

from flowsheets.mea_stripper_reformulated import MEAStripperFlowsheet
from flowsheets.mea_absorber_reformulated import MEAAbsorberFlowsheet
from flowsheets.mea_properties import state_bounds_default

from flowsheets.MEA_CCS_system_cost_module import get_CO2_absorption_cost


@declare_process_block_class("MEACombinedFlowsheet")
class MEACombinedFlowsheetData(FlowsheetBlockData):
    CONFIG = FlowsheetBlockData.CONFIG()
    CONFIG.declare(
        "stripper_finite_element_set",
        ConfigValue(
            domain=ListOf(float),
            description="List containing coordinates finite element faces "
                        "of stripper column. Coordinates must start with zero, be "
                        "strictly increasing, and end with one",
        ),
    )
    CONFIG.declare(
        "absorber_finite_element_set",
        ConfigValue(
            domain=ListOf(float),
            description="List containing coordinates finite element faces "
                        "of absorber column. Coordinates must start with zero, be "
                        "strictly increasing, and end with one",
        ),
    )
    CONFIG.declare(
        "property_package_bounds",
        state_bounds_default()
    )
    def build(self):
        super().build()
        self._add_units()
        self._add_arcs()
        self._add_performance_math()
        self._set_design_inputs()
        self.design_variables = ComponentSet([
            self.absorber_section.absorber.length_column,
            self.absorber_section.absorber.diameter_column,
            self.stripper_section.stripper.length_column,
            self.stripper_section.stripper.diameter_column,
            self.absorber_section.lean_rich_heat_exchanger.area,
        ])
        self.control_variables = ComponentSet([
            self.mea_recirculation_rate,
            self.h2o_mea_ratio,
            self.stripper_section.condenser.heat_duty,
            self.stripper_section.reboiler.heat_duty,
        ])

    def _add_units(self):
        self.stripper_section = MEAStripperFlowsheet(
            column_finite_element_set=self.config.stripper_finite_element_set,
            property_package_bounds=self.config.property_package_bounds,
        )
        self.absorber_section = MEAAbsorberFlowsheet(
            column_finite_element_set=self.config.absorber_finite_element_set,
            property_package_bounds=self.config.property_package_bounds,
        )
        self.flue_gas_feed_scaler = StreamScaler(
            property_package=self.absorber_section.vapor_properties
        )
        self.clean_gas_scaler = StreamScaler(
            property_package=self.absorber_section.vapor_properties
        )
        self.makeup_scaler = StreamScaler(
            property_package=self.absorber_section.liquid_properties_no_ions
        )
        self.distillate_scaler = StreamScaler(
            property_package=self.stripper_section.vapor_properties
        )

        # Use Translator block in order to manage specification of MEA flowrate in total recycle loop
        # Use the liquid property package without ions to reduce number of equations generated
        self.lean_solvent_translator = Translator(
            inlet_property_package=self.absorber_section.liquid_properties_no_ions,
            outlet_property_package=self.absorber_section.liquid_properties_no_ions,
            outlet_state_defined=False
        )

    def _add_arcs(self):
        # Define Arcs (streams)
        self.lean_solvent_stream01 = Arc(
            source=self.stripper_section.lean_solvent,
            destination=self.lean_solvent_translator.inlet
        )
        self.lean_solvent_stream02 = Arc(
            source=self.lean_solvent_translator.outlet,
            destination=self.absorber_section.lean_solvent
        )
        self.rich_solvent_stream = Arc(
            source=self.absorber_section.rich_solvent,
            destination=self.stripper_section.rich_solvent
        )
        self.flue_gas_feed_scaler_stream = Arc(
            source=self.flue_gas_feed_scaler.outlet,
            destination=self.absorber_section.flue_gas_feed
        )
        self.clean_gas_scaler_stream = Arc(
            source=self.absorber_section.clean_gas,
            destination=self.clean_gas_scaler.inlet
        )
        self.makeup_scaler_stream = Arc(
            source=self.makeup_scaler.outlet,
            destination=self.absorber_section.makeup,
        )
        self.distillate_scaler_stream = Arc(
            source=self.stripper_section.distillate,
            destination=self.distillate_scaler.inlet
        )

        # Transform Arcs
        pyo.TransformationFactory("network.expand_arcs").apply_to(self)

        # Add sub-flowsheet level ports for easy connectivity
        self.flue_gas_feed = Port(extends=self.flue_gas_feed_scaler.inlet)
        self.clean_gas = Port(extends=self.clean_gas_scaler.outlet)
        self.makeup = Port(extends=self.makeup_scaler.inlet)
        self.distillate = Port(extends=self.distillate_scaler.outlet)

    def _add_performance_math(self):
        self.h2o_mea_ratio = pyo.Var(self.time, initialize=7.91)

        @self.Constraint(self.time)
        def h2o_mea_ratio_eqn(b, t):
            return (
                    b.h2o_mea_ratio[t]
                    * b.absorber_section.makeup_mixer.outlet.mole_frac_comp[t, "MEA"]
                    == b.absorber_section.makeup_mixer.outlet.mole_frac_comp[t, "H2O"]
            )

        self.lean_loading = pyo.Var(self.time)

        @self.Constraint(self.time)
        def lean_loading_eqn(fs, t):
            return (
                    fs.lean_loading[t]
                    * fs.absorber_section.absorber.liquid_inlet.mole_frac_comp[t, "MEA"]
                    == fs.absorber_section.absorber.liquid_inlet.mole_frac_comp[t, "CO2"]
            )

        self.rich_loading = pyo.Var(self.time)

        @self.Constraint(self.time)
        def rich_loading_eqn(fs, t):
            return (
                    fs.rich_loading[t]
                    * fs.absorber_section.absorber.liquid_outlet.mole_frac_comp[t, "MEA"]
                    == fs.absorber_section.absorber.liquid_outlet.mole_frac_comp[t, "CO2"]
            )

        @self.Constraint(self.time)
        def stripper_reflux_pressure(b, t):
            return (
                    b.stripper_section.reflux_mixer.rich_solvent.pressure[t]
                    == b.stripper_section.reflux_mixer.reflux.pressure[t]
            )
        
        # specific reboiler duty
        self.SRD = pyo.Var(
            self.time, initialize=1, units=pyo.units.MJ/pyo.units.kg
            )
        
        @self.Constraint(self.time)
        def calculate_SRD(fs, t):
            return (
                fs.SRD[t] 
                * pyo.units.convert(
                    fs.absorber_section.flue_gas_flow_mass_CO2[t] -
                    fs.absorber_section.stack_gas_flow_mass_CO2[t], 
                    to_units=pyo.units.kg/pyo.units.s) 
                == pyo.units.convert(fs.stripper_section.reboiler.heat_duty[t],
                                     to_units=pyo.units.MJ/pyo.units.s)
                )
        
        self.number_trains = pyo.Var(initialize=4)
        self.number_trains.fix()

        @self.Constraint()
        def flue_gas_feed_multiplier_eqn(b):
            return b.flue_gas_feed_scaler.multiplier == 1 / b.number_trains
        
        @self.Constraint()
        def clean_gas_multiplier_eqn(b):
            return b.clean_gas_scaler.multiplier == b.number_trains
        
        @self.Constraint()
        def makeup_multiplier_eqn(b):
            return b.makeup_scaler.multiplier == 1 / b.number_trains
        
        @self.Constraint()
        def distillate_multiplier_eqn(b):
            return b.distillate_scaler.multiplier == b.number_trains
        
        # Specify translator block
        @self.lean_solvent_translator.Constraint(self.time)
        def temperature_eqn(b, t):
            return b.properties_in[t].temperature == b.properties_out[t].temperature
        
        @self.lean_solvent_translator.Constraint(self.time)
        def pressure_eqn(b, t):
            return b.properties_in[t].pressure == b.properties_out[t].pressure
        
        @self.lean_solvent_translator.Constraint(self.time, self.lean_solvent_translator.properties_in.component_list)
        def flow_mol_comp_eqn(b, t, j):
            return b.properties_in[t].flow_mol_comp[j] == b.properties_out[t].flow_mol_comp[j]
        # Need to deactivate this constraint in order to specify amount of MEA in recycle loop
        self.lean_solvent_translator.flow_mol_comp_eqn[:,"MEA"].deactivate()

        self.mea_recirculation_rate = pyo.Var(self.time, initialize=800, units=pyo.units.mol/pyo.units.s)
        @self.lean_solvent_translator.Constraint(self.time)
        def mea_recirculation_eqn(b, t):
            return b.properties_out[t].flow_mol_comp["MEA"] == self.mea_recirculation_rate[t]
        self.mea_recirculation_rate.fix()

    def _set_design_inputs(self):
        self.flue_gas_feed.flow_mol.fix(37116.4)  # mol/sec
        self.flue_gas_feed.temperature.fix(313.15)  # K
        self.flue_gas_feed.pressure.fix(105000)  # Pa
        self.flue_gas_feed.mole_frac_comp[0, "CO2"].fix(0.04226)
        self.flue_gas_feed.mole_frac_comp[0, "H2O"].fix(0.05480)
        self.flue_gas_feed.mole_frac_comp[0, "N2"].fix(0.76942 + 0.00920) # Added term is to make up for Argon, which is neglected
        self.flue_gas_feed.mole_frac_comp[0, "O2"].fix(0.12430)
        
        self.makeup.flow_mol.fix(7500)  # mol/sec
        self.makeup.temperature.fix(313.15)  # K
        self.makeup.pressure.fix(183700)  # Pa
        # Pressure determined by pressure equality with lean solvent stream
        self.makeup.mole_frac_comp[0, "CO2"].fix(1e-12)
        self.makeup.mole_frac_comp[0, "H2O"].fix(1.0)
        self.makeup.mole_frac_comp[0, "MEA"].fix(1e-12)
    
    def calculate_scaling_factors(self):
        for t in self.time:
            iscale.constraint_scaling_transform(self.stripper_reflux_pressure[t], 1e-5)

            sT = iscale.get_scaling_factor(self.lean_solvent_translator.properties_in[t].temperature)
            iscale.constraint_scaling_transform(self.lean_solvent_translator.temperature_eqn[t], sT)
            sP = iscale.get_scaling_factor(self.lean_solvent_translator.properties_in[t].pressure)
            iscale.constraint_scaling_transform(self.lean_solvent_translator.pressure_eqn[t], sP)
            for j in self.lean_solvent_translator.properties_in.component_list:
                sF = iscale.get_scaling_factor(self.lean_solvent_translator.properties_in[t].flow_mol_comp[j])
                iscale.constraint_scaling_transform(self.lean_solvent_translator.flow_mol_comp_eqn[t, j], sF)
            sF = iscale.set_and_get_scaling_factor(self.mea_recirculation_rate[t], 1/750)
            iscale.constraint_scaling_transform(self.lean_solvent_translator.mea_recirculation_eqn[t], sF)

    def strip_statevar_bounds(self):
        def strip_var_bounds(var):
            for j in var.keys():
                var.setlb(None)
                var.setub(None)
        self.absorber_section.strip_statevar_bounds_for_absorber_initialization()
        self.stripper_section.strip_statevar_bounds_for_stripper_initialization()
        # Strip variable bounds
        for t in self.time:
            strip_var_bounds(self.makeup_scaler.inlet_block[t].temperature)
            strip_var_bounds(self.makeup_scaler.inlet_block[t].pressure)
            strip_var_bounds(self.makeup_scaler.inlet_block[t].flow_mol)
            strip_var_bounds(self.makeup_scaler.inlet_block[t].mole_frac_comp)
            strip_var_bounds(self.makeup_scaler.inlet_block[t].mole_frac_phase_comp)


    def add_costing(self):
        """
        Method to add costing equations using IDAES Power Plant Costing 
        Framework. Relevant process variables are collected from various 
        process results, and passed to an external economic module where 
        costing blocks are generated as childs of `self`. Note that `self` 
        should be a flowsheet object, which is defined in the script 
        `run_combined_flowsheet`.
        
        Results are exported to CSV unless set to `False`. New results will
        automatically overwrite (delete) old files by default, if `False` the 
        new files will not be generated if old files are found.
        """

        self.costing_setup = pyo.Block()
        
        # wash section components
        # don't need to scale heights
        self.costing_setup.wash_section_packing_height = pyo.Var(
            initialize=6.0, units=pyo.units.m
            )
        self.costing_setup.wash_section_packing_height.fix()
        
        # solvent initial fill
        self.costing_setup.solvent_fill_init = pyo.Var(
            self.time, initialize=1, units=pyo.units.kg)

        @self.costing_setup.Constraint(self.time)
        def calculate_solvent_fill_init(b, t):
            return (
                b.solvent_fill_init[t] ==
                pyo.units.convert(
                    b.parent_block().mea_recirculation_rate[t]
                    * b.parent_block().absorber_section.liquid_properties.MEA.mw,
                    to_units=pyo.units.kg/pyo.units.hr
                    ) *
                b.parent_block().number_trains.value * 5 * pyo.units.hr
                )
        
        # emissions components
        self.costing_setup.CO2_capture_rate = pyo.Var(
            self.time, initialize=1, units=pyo.units.lb/pyo.units.hr
            )
        
        @self.costing_setup.Constraint(self.time)
        def calculate_CO2_capture_rate(b, t):
            return (
                b.CO2_capture_rate[t] == pyo.units.convert(
                    b.parent_block().number_trains.value *
                    (
                        b.parent_block().absorber_section.flue_gas_flow_mass_CO2[t] -
                        b.parent_block().absorber_section.stack_gas_flow_mass_CO2[t]
                        ),
                    to_units=pyo.units.lb/pyo.units.hr
                    )
                )

        get_CO2_absorption_cost(self)

        # post-solve with costing
        print("\nSolve with costing setup variables and constraints...\n")
        optarg={
            'nlp_scaling_method': 'user-scaling',
            'linear_solver': 'ma57',
            'OF_ma57_automatic_scaling': 'yes',
            'max_iter': 300,
            'tol': 1e-8,
            'halt_on_ampl_error': 'no',
        }
        solver_obj = get_solver("ipopt", optarg)
        
        results = solver_obj.solve(self, tee=True)
        pyo.assert_optimal_termination(results)


    def print_costing_results_to_csv(self, directory, file_name):
        CE_index_year = "2018"
        CE_index_units = getattr(pyo.units, "MUSD_" + CE_index_year)
        
        # CCS Cost Summary
        CCS_cost_summary_component_name = [
            "Absorber Column Volume with Heads",
            "Absorber Packing Volume",
            "Lean Rich Heat Exchanger",
            "Rich Solvent Pump (based on rich solvent rated flow)",
            "Lean Solvent Pump (based on lean solvent rated flow)",
            "Stripper Column Volume with Heads",
            "Stripper Packing Volume",
            "Stripper Condenser Area",
            "Stripper Reboiler Area",
            "Stripper Reboiler Condensate Pot (based on reboiler heat duty)",
            "Stripper Reflux Drum (based on CO2 capture rate)",
            "Stripper Reflux Pump (based on CO2 capture rate)",
            "Solvent Filtration (based on lean solvent flowrate per train)",
            "Solvent Storage Tank (based on lean solvent flowrate per train)",
            "Solvent Initial Fill",
            "Utility Cost, Rich Solvent Pump",
            "Utility Cost, Lean Solvent Pump",
            "Utility Cost, Steam (Reboiler)",
            "Utility Cost, Cooling Water (Condenser)",
            "Utility Cost, Water Make-up",
            "Total Equipment Cost, CCS TEC",
            "Total Plant Cost, CCS TPC",
            "Annualized Investment Cost, CCS AIC",
            "Annual Operating Cost, CCS AOC",
            "Total Annualized Cost, CCS TAC",
            "Total Annualized Cost per Tonne of CO2 Captured",
        ]
        CCS_cost_summary_component_value = [
            # Absorber column volume with heads
            self.costing.absorber_volume_column_withheads,
            self.costing.absorber_packing_volume,
            self.absorber_section.lean_rich_heat_exchanger.area,
            # Rich solvent rated flow
            1.1*pyo.units.convert(
                self.absorber_section.rich_solvent_flow_vol[0], 
                to_units=pyo.units.gal/pyo.units.min),
            # Lean solvent rated flow
            1.1*pyo.units.convert(
                self.stripper_section.lean_solvent_flow_vol[0], 
                to_units=pyo.units.gal/pyo.units.min), 
            # Stripper column volume with heads
            self.costing.stripper_volume_column_withheads,
            self.costing.stripper_packing_volume,
            # Stripper condenser area
            pyo.units.convert(self.stripper_section.condenser.area, 
                              to_units=pyo.units.m**2),
            # Stripper reboiler area
            self.stripper_section.reboiler.area, 
            # Reboiler condensate pot based on reboiler heat duty
            pyo.units.convert(self.stripper_section.reboiler.heat_duty[0], 
                              to_units=pyo.units.MW),
            # Stripper reflux drum based on CO2 capture rate, one train
            pyo.units.convert(self.costing_setup.CO2_capture_rate[0], 
                                    to_units=pyo.units.kg/pyo.units.hr) / 
                  self.number_trains.value,
            # Stripper reflux pumo based on CO2 capture rate, one train
            pyo.units.convert(self.costing_setup.CO2_capture_rate[0], 
                                    to_units=pyo.units.kg/pyo.units.hr) / 
                  self.number_trains.value,
            # Solvent filtration based on lean solvent flowrate per train
            pyo.units.convert(self.stripper_section.lean_solvent_flow_vol[0], 
                              to_units=pyo.units.gal/pyo.units.min),
            #Solvent storage tank based on lean solvent flowrate per train
            pyo.units.convert(self.stripper_section.lean_solvent_flow_vol[0], 
                              to_units=pyo.units.gal/pyo.units.min),
            # Solvent, initial fill
            self.costing_setup.solvent_fill_init[0],
            # Utility costs
            self.costing.UC_rich_solvent_pump,
            self.costing.UC_lean_solvent_pump,
            self.costing.UC_reboiler,
            self.costing.UC_condenser,
            self.costing.UC_H2O_makeup,
            # Economic metrics
            pyo.units.convert(self.costing.CCS_TEC,
                              to_units=CE_index_units),
            pyo.units.convert(self.costing.CCS_TPC,
                              to_units=CE_index_units),
            pyo.units.convert(self.costing.CCS_AIC,
                              to_units=CE_index_units/pyo.units.year),            
            pyo.units.convert(self.costing.CCS_AOC,
                              to_units=CE_index_units/pyo.units.year),
            pyo.units.convert(self.costing.CCS_TAC,
                              to_units=CE_index_units/pyo.units.year),
            pyo.units.convert(self.costing.CCS_TAC_perCO2captured,
                              to_units=CE_index_units/pyo.units.tonne),
            ]
            
        CCS_cost_summary_component_cost = [
            self.costing.absorber_column_cost,
            self.costing.absorber_packing_cost,
            self.costing.lean_rich_hex_cost,
            self.costing.rich_solvent_pump_cost,
            self.costing.lean_solvent_pump_cost,
            self.costing.stripper_column_cost,
            self.costing.stripper_packing_cost,
            self.costing.stripper_condenser_cost,
            self.costing.stripper_reboiler_cost,
            self.costing.reboiler_condensate_pot_cost,
            self.costing.stripper_reflux_drum_cost,
            self.costing.stripper_reflux_pump_cost,
            self.costing.solvent_filtration_cost,
            self.costing.solvent_storage_tank_cost,
            self.costing.RemovalSystem_Equip_Adjust,
            # Utility costs
            self.costing.UC_rich_solvent_pump,
            self.costing.UC_lean_solvent_pump,
            self.costing.UC_reboiler,
            self.costing.UC_condenser,
            self.costing.UC_H2O_makeup,
            # Economic metrics
            self.costing.CCS_TEC,
            self.costing.CCS_TPC,
            self.costing.CCS_AIC,
            self.costing.CCS_AOC,
            self.costing.CCS_TAC,
            self.costing.CCS_TAC_perCO2captured,
            ]
            
        CCS_cost_summary_values = [pyo.value(obj) for obj in CCS_cost_summary_component_value]
        CCS_cost_summary_values_units = [pyo.units.get_units(obj) for obj in CCS_cost_summary_component_value]
        
        CCS_cost_summary_costs = [pyo.value(obj) for obj in CCS_cost_summary_component_cost]
        CCS_cost_summary_costs_units = [pyo.units.get_units(obj) for obj in CCS_cost_summary_component_cost]
        
        my_results = {'Component':CCS_cost_summary_component_name,
                'Value': CCS_cost_summary_values,
                'Value unit': CCS_cost_summary_values_units,
                'Cost': CCS_cost_summary_costs,
                'Cost unit': CCS_cost_summary_costs_units,
                }
        
        import pandas as pd
        df_out_my_results = pd.DataFrame(my_results)
        
        # Check if results folder exists (create if it does not exist), save results as .csv
        # Directory name    
        import os
        values_and_costs = os.path.join(directory, file_name)
        
        try:
            os.mkdir(directory)
            df_out_my_results.to_csv(values_and_costs) 
        except:
            df_out_my_results.to_csv(values_and_costs) 
            
            
    def initialize_build(
            self,
            outlvl=idaeslog.NOTSET,
            solver="ipopt",
            optarg=None,
            rich_solvent_guess=None,
            lean_solvent_guess=None,
            stripper_boilup_guess=None,
            load_from=None,
            save_to="combined_flowsheet_init.json.gz",
    ):
        if load_from is not None:
            if os.path.exists(load_from):
                # here suffix=False avoids loading scaling factors
                iutil.from_json(
                    self, fname=load_from, wts=iutil.StoreSpec(suffix=False)
                )
                return

        solver_obj = get_solver(solver, optarg)
        init_log = idaeslog.getInitLogger(self.name, outlvl)
        solve_log = idaeslog.getSolveLogger(self.name, outlvl)

        def safe_solve(blk):
            assert degrees_of_freedom(blk) == 0
            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                results = solver_obj.solve(blk, tee=slc.tee)
            pyo.assert_optimal_termination(results)

        # Set absorber section design and operating conditions
        # Flue gas inlet
        flue_gas_feed_flags = fix_vars(self.flue_gas_feed.vars.values())
        # Makeup H2O
        # Flowrate will be calculated by H2O:MEA ratio
        makeup_flags = fix_vars(self.makeup.vars.values())

        self.flue_gas_feed_scaler.multiplier.value = 1 / self.number_trains.value
        self.clean_gas_scaler.multiplier.value = self.number_trains.value
        self.makeup_scaler.multiplier.value = 1 / self.number_trains.value
        self.distillate_scaler.multiplier.value = self.number_trains.value

        self.flue_gas_feed_scaler.initialize_build(
            outlvl=outlvl, 
            solver=solver,
            optarg=optarg,
        )
        propagate_state(self.flue_gas_feed_scaler_stream, overwrite_fixed=True)
        self.makeup_scaler.initialize_build(
            outlvl=outlvl, 
            solver=solver,
            optarg=optarg,
        )
        propagate_state(self.makeup_scaler_stream, overwrite_fixed=True)

        if lean_solvent_guess is None:
            lean_pump_inlet = self.absorber_section.lean_solvent_pump.inlet
            lean_solvent_guess = ComponentMap()
            lean_solvent_guess[lean_pump_inlet.flow_mol] = 23000/self.number_trains.value  # mol/sec 52000/self.number_trains.value  # mol/sec #
            lean_solvent_guess[lean_pump_inlet.temperature] = 396  # K
            lean_solvent_guess[lean_pump_inlet.pressure] = 183700  # Pa
            lean_solvent_guess[lean_pump_inlet.mole_frac_comp[:, "CO2"]] = 0.015178
            lean_solvent_guess[lean_pump_inlet.mole_frac_comp[:, "H2O"]] = 0.840049
            lean_solvent_guess[lean_pump_inlet.mole_frac_comp[:, "MEA"]] = 0.144773

        for comp, guess in lean_solvent_guess.items():
            comp.fix(guess)

        # Initialize sub-flowsheets
        self.absorber_section.initialize_build(
            outlvl=outlvl,
            solver=solver,
            optarg=optarg
        )
        propagate_state(self.clean_gas_scaler_stream)
        self.clean_gas_scaler.initialize_build(
            outlvl=outlvl, 
            solver=solver,
            optarg=optarg,
        )

        if rich_solvent_guess is None:
            rich_solvent = self.stripper_section.reflux_mixer.rich_solvent
            rich_solvent_guess = ComponentMap()
            rich_solvent_guess[rich_solvent.flow_mol] = 28000/self.number_trains.value  # mol/sec 60500/self.number_trains.value  # mol/sec #
            rich_solvent_guess[rich_solvent.temperature] = 366.7  # K
            rich_solvent_guess[rich_solvent.pressure] = 183700  # Pa
            rich_solvent_guess[rich_solvent.mole_frac_comp[:, "CO2"]] = 0.058298
            rich_solvent_guess[rich_solvent.mole_frac_comp[:, "H2O"]] = 0.823349
            rich_solvent_guess[rich_solvent.mole_frac_comp[:, "MEA"]] =  0.118353

        for comp, guess in rich_solvent_guess.items():
            comp.fix(guess)

        self.stripper_reflux_pressure.deactivate()
        self.lean_solvent_stream02_expanded.deactivate()
        self.rich_solvent_stream_expanded.deactivate()

        self.stripper_section.initialize_build(
            outlvl=outlvl,
            solver=solver, 
            optarg=optarg,
            boilup_guess=stripper_boilup_guess
        )
        propagate_state(self.distillate_scaler_stream)
        self.distillate_scaler.initialize_build(
            outlvl=outlvl, 
            solver=solver,
            optarg=optarg,
        )
        propagate_state(self.lean_solvent_stream01)
        self.lean_solvent_translator.initialize_build(
            outlvl=outlvl,
            solver=solver,
            optarg=optarg,
        )

        init_log.info("Switching specification of reboiler and makeup")
        self.absorber_section.flue_gas_feed.unfix()
        self.absorber_section.makeup.unfix()
        self.stripper_section.reboiler.heat_duty.unfix()
        self.stripper_section.reboiler.bottoms.temperature.fix()

        self.makeup.flow_mol.unfix()
        self.h2o_mea_ratio.fix(7.91)

        safe_solve(self)

        init_log.info("Switching specification of condenser")
        self.stripper_section.condenser.heat_duty.unfix()
        self.stripper_section.condenser.reflux.temperature.fix()

        safe_solve(self)
        
        init_log.info("Coupling rich solvent stream")
        self.rich_solvent_stream_expanded.activate()

        self.stripper_section.rich_solvent.unfix()

        safe_solve(self)

        init_log.info("Coupling lean solvent stream")
        self.lean_solvent_stream02_expanded.activate()
        propagate_state(self.lean_solvent_stream02)

        self.absorber_section.lean_solvent.unfix()
        safe_solve(self)

        init_log.info("Final Steps")

        self.absorber_section.rich_solvent_pump.deltaP.unfix()
        self.stripper_reflux_pressure.activate()

        self.absorber_section.lean_solvent_pump.deltaP.fix(0)
        safe_solve(self)

        if save_to is not None:
            iutil.to_json(self, fname=save_to)
            init_log.info_low(f"Initialization saved to {save_to}")

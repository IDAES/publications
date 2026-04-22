"""
Flowsheet combining absorber and stripper sections for Monoethanolamine solvent carbon capture system
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

from util import fix_vars

from idaes.core.solvers.get_solver import get_solver
import idaes.logger as idaeslog
from idaes.core.util.initialization import propagate_state
import idaes.core.util.scaling as iscale

from flowsheets.mea_stripper_reformulated import MEAStripperFlowsheet
from flowsheets.mea_absorber_reformulated import MEAAbsorberFlowsheet
from flowsheets.mea_properties import state_bounds_default

from flowsheets.MEA_NGCC_CCS_integrated_system_cost_module import get_ngcc_solvent_cost
from flowsheets.combined_flowsheet_withcosting_CCS import MEACombinedFlowsheetData as MEACombinedFlowsheetDataCCS

@declare_process_block_class("MEACombinedFlowsheet")
class MEACombinedFlowsheetData(MEACombinedFlowsheetDataCCS):
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

        # temporarily fix unfixed process model variables
        vars_temporarily_fixed = []
        for var in self.parent_block().component_data_objects(pyo.Var, descend_into=True):
            if not var.is_fixed():
                vars_temporarily_fixed.append(var)
                var.fix()

        # define hard-coded and scaled variables that don't exist in the model
        # some of these initialize from existing model variables
        # some of these scale by the number of trains when initializing
        # some of these scale with hard-coded reference values (expressions)
        self.costing_setup = pyo.Block()

        # reference total flue gas flow for all trains - value taken from 
        # Exhibit 5-22. Case B31B.90 stream table, NGCC with capture (Rev 4a)
        self.costing_setup.ref_fg_feed_flow = pyo.Param(
            initialize=3927398, mutable=True,
            units=pyo.units.kg/pyo.units.hr
            )
        self.costing_setup.fg_feed_ratio = pyo.Var(
            self.time, initialize=1, units=pyo.units.dimensionless
            )

        # calculate mass flow of flue gas feed stream
        self.costing_setup.flue_gas_feed_flow_mass = pyo.Var(
            self.time, initialize=1e4, units=pyo.units.kg/pyo.units.hr
            )

        @self.costing_setup.Constraint(self.time)
        def calculate_fg_flow_mass(b, t):
            return (
                b.flue_gas_feed_flow_mass[t] == pyo.units.convert(
                    sum(
                        b.parent_block().flue_gas_feed.flow_mol[t] *  # mol/s
                        b.parent_block().flue_gas_feed.mole_frac_comp[t, comp] *  # mol/mol
                        getattr(
                            b.parent_block().absorber_section.vapor_properties,
                            comp
                            ).mw  # kg/mol
                        for comp in ["CO2", "H2O", "N2", "O2"]
                        ),
                    to_units=pyo.units.kg/pyo.units.hr
                    )
                )

        @self.costing_setup.Constraint(self.time)
        def calculate_fg_feed_ratio(b, t):
            return (b.fg_feed_ratio[t] ==
                    (
                        b.flue_gas_feed_flow_mass[t] /
                        b.ref_fg_feed_flow
                    )
                    )

        # reference total stripper condenser vapor outlet for one train
        self.costing_setup.ref_cond_vap_flow = pyo.Param(
            initialize=114294.388265244, mutable=True,
            units=pyo.units.kg/pyo.units.hr
            )
        self.costing_setup.cond_vap_ratio = pyo.Var(
            self.time, initialize=1, units=pyo.units.dimensionless
            )

        # calculate mass flow of condenser vapor outlet stream
        self.costing_setup.cond_vap_flow_mass = pyo.Var(
            self.time, initialize=1e4, units=pyo.units.kg/pyo.units.hr
            )

        @self.costing_setup.Constraint(self.time)
        def calculate_cond_vap_flow_mass(b, t):
            return (
                b.cond_vap_flow_mass[t] == pyo.units.convert(
                    sum(
                        b.parent_block().stripper_section.condenser.vapor_outlet.flow_mol[t] *  # mol/s
                        b.parent_block().stripper_section.condenser.vapor_outlet.mole_frac_comp[t, comp] *  # mol/mol
                        getattr(
                            b.parent_block().stripper_section.vapor_properties,
                            comp
                            ).mw  # kg/mol
                        for comp in ["CO2", "H2O"]
                        ),
                    to_units=pyo.units.kg/pyo.units.hr
                    )
                )

        @self.costing_setup.Constraint(self.time)
        def calculate_cond_vap_ratio(b, t):
            return (b.cond_vap_ratio[t] ==
                    (
                        b.cond_vap_flow_mass[t] /
                        b.ref_cond_vap_flow
                        )
                    )

        # NGCC components
        self.costing_setup.feedwater_flowrate = pyo.Var(
            self.time, initialize=486242.527057,
            units=pyo.units.kg/pyo.units.hr
            )

        @self.costing_setup.Constraint(self.time)
        def calculate_feedwater_flowrate(b, t):
            return (b.feedwater_flowrate[t] ==
                    (
                        486242.527057 * pyo.units.kg/pyo.units.hr *
                        b.fg_feed_ratio[t]
                        )
                    )

        self.costing_setup.fuelgas_flowrate = pyo.Var(
            self.time, initialize=93272.0,
            units=pyo.units.kg/pyo.units.hr
            )

        @self.costing_setup.Constraint(self.time)
        def calculate_fuelgas_flowrate(b, t):
            return (b.fuelgas_flowrate[t] ==
                    (
                        93272.0 * pyo.units.kg/pyo.units.hr *
                        b.fg_feed_ratio[t]
                        )
                    )

        self.costing_setup.fluegas_flowrate = pyo.Var(
            self.time, initialize=1282.72285,
            units=pyo.units.m**3/pyo.units.s
            )

        @self.costing_setup.Constraint(self.time)
        def calculate_fluegas_flowrate(b, t):
            return (b.fluegas_flowrate[t] ==
                    (
                        1282.72285 * pyo.units.m**3/pyo.units.s *
                        b.fg_feed_ratio[t]
                        )
                    )

        # turbine components
        self.costing_setup.flue_gas_blower_load = pyo.Var(
            self.time, initialize=8037475.28, units=pyo.units.W
            )

        @self.costing_setup.Constraint(self.time)
        def calculate_flue_gas_blower_load(b, t):
            return (b.flue_gas_blower_load[t] ==
                    (
                        8037475.28 * pyo.units.W *
                        b.fg_feed_ratio[t]
                        )
                    )

        # Fraction of steam extraction - surrogate model function of stripper 
        # reboiler duty
        self.costing_setup.IP_LP_crossover_steam_fraction = pyo.Var(
            self.time, initialize=0.662806512, units=pyo.units.dimensionless
            )
        
        @self.costing_setup.Constraint(self.time)
        def calculate_fraction_steam_extraction(b, t): 
            return (b.IP_LP_crossover_steam_fraction[t] * 379.61  ==
                b.parent_block().stripper_section.reboiler.heat_duty[0] * 1e-6
                * b.parent_block().number_trains
                )
        
        self.costing_setup.flue_gas_flowrate_blower_discharge = pyo.Var(
            self.time, initialize=607.06744, units=pyo.units.m**3/pyo.units.s
            )

        @self.costing_setup.Constraint(self.time)
        def calculate_flue_gas_flowrate_blower_discharge(b, t):
            return (b.flue_gas_flowrate_blower_discharge[t] ==
                    (
                        607.06744 * pyo.units.m**3/pyo.units.s *
                        b.fg_feed_ratio[t]
                        )
                    )

        # wash section components
        # don't need to scale heights
        self.costing_setup.wash_section_packing_height = pyo.Var(
            initialize=6.0, units=pyo.units.m
            )
        self.costing_setup.wash_section_packing_height.fix()
        
        # DCC components
        self.costing_setup.dcc_packing_height = pyo.Var(
            initialize=18.0, units=pyo.units.m
            )
        self.costing_setup.dcc_packing_height.fix()
        
        self.costing_setup.dcc_diameter = pyo.Var(
            initialize=16.0, units=pyo.units.m
            )

        # area scales linearly with flow ratio, so diameter scales as sqrt
        @self.costing_setup.Constraint()
        def calculate_dcc_diameter(b):
            return (b.dcc_diameter ==
                    (
                        16.0 * pyo.units.m *
                        pyo.sqrt(b.fg_feed_ratio[0])
                        )
                    )

        self.costing_setup.dcc_duty = pyo.Var(
            self.time, initialize=-81409348.0, units=pyo.units.W
            )

        @self.costing_setup.Constraint(self.time)
        def calculate_dcc_duty(b, t):
            return (b.dcc_duty[t] ==
                    (
                        -81409348.0 * pyo.units.W *
                        b.fg_feed_ratio[t]
                        )
                    )

        # don't need to scale temperatures
        self.costing_setup.dcc_temperature_in = pyo.Var(
            self.time, initialize=54.5472838 + 273.15, units=pyo.units.K
            )
        self.costing_setup.dcc_temperature_in.fix()
        
        self.costing_setup.dcc_temperature_out = pyo.Var(
            self.time, initialize=21.0 + 273.15, units=pyo.units.K
            )
        self.costing_setup.dcc_temperature_out.fix()
        
        self.costing_setup.dcc_pump_load = pyo.Var(
            self.time, initialize=13234.4015, units=pyo.units.W
            )

        @self.costing_setup.Constraint(self.time)
        def calculate_dcc_pump_load(b, t):
            return (b.dcc_pump_load[t] ==
                    (
                        13234.4015 * pyo.units.W *
                        b.fg_feed_ratio[t]
                        )
                    )

        self.costing_setup.dcc_pump_flowrate = pyo.Var(
            self.time, initialize=0.534461996, units=pyo.units.m**3/pyo.units.s
            )

        @self.costing_setup.Constraint(self.time)
        def calculate_dcc_pump_flowrate(b, t):
            return (b.dcc_pump_flowrate[t] ==
                    (
                        0.534461996 * pyo.units.m**3/pyo.units.s *
                        b.fg_feed_ratio[t]
                        )
                    )

        # Compressor components
        self.costing_setup.CO2_compressor_auxiliary_load = pyo.Var(
            self.time, initialize=self.number_trains.value * 11020626.6,
            units=pyo.units.W
            )

        @self.costing_setup.Constraint(self.time)
        def calculate_CO2_compressor_auxiliary_load(b, t):
            return (b.CO2_compressor_auxiliary_load[t] ==
                    (
                        b.parent_block().number_trains.value *
                        11020626.6 * pyo.units.W *
                        b.cond_vap_ratio[t]
                        )
                    )

        self.costing_setup.CO2_compressor_intercooling_duty = pyo.Var(
            self.time, initialize=self.number_trains.value * -16058955.0,
            units=pyo.units.W
            )

        @self.costing_setup.Constraint(self.time)
        def calculate_CO2_compressor_intercooling_duty(b, t):
            return (b.CO2_compressor_intercooling_duty[t] ==
                    (
                        b.parent_block().number_trains.value *
                        -16058955.0 * pyo.units.W *
                        b.cond_vap_ratio[t]
                        )
                    )

        # define variables/expressions that calculate based on other variables
        # NGCC components
        self.costing_setup.stackgas_flowrate = pyo.Var(
            self.time, initialize=1, units=pyo.units.m**3/pyo.units.s
            )

        @self.costing_setup.Constraint(self.time)
        def calculate_stackgas_flowrate(b, t):
            return (
                b.stackgas_flowrate[t] == pyo.units.convert(
                    b.parent_block().number_trains.value *
                    b.parent_block().absorber_section.stack_gas_flow_vol[t],
                    to_units=pyo.units.m**3/pyo.units.s
                    )
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

        self.costing_setup.CO2_emissions_no_cap = pyo.Var(
            self.time, initialize=1, units=pyo.units.lb/pyo.units.hr
            )

        @self.costing_setup.Constraint(self.time)
        def calculate_CO2_emissions_no_cap(b, t):
            return (
                b.CO2_emissions_no_cap[t] == pyo.units.convert(
                    b.parent_block().number_trains.value *
                    b.parent_block().absorber_section.flue_gas_flow_mass_CO2[t],
                    to_units=pyo.units.lb/pyo.units.hr
                    )
                )

        # The 'initial' solvent fill should rather be a 'maximum' solvent fill
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

        # pre-solve setup variables and constraints
        print("\nCosting setup variables and constraints built, solving...\n")
        results = get_solver().solve(self.costing_setup, tee=True)

        # unfix frozen process model variables
        for var in vars_temporarily_fixed:
            var.unfix()

        print("\nCosting setup solved, building main costing blocks...\n")

        get_ngcc_solvent_cost(self)

        # post-solve with costing
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


    def print_costing_results_to_csv(self, directory, file1_name, file2_name,
                                     file3_name):
        
        import pandas as pd
        
        CE_index_year = "2018"
        CE_index_units = getattr(pyo.units, "MUSD_" + CE_index_year)
        
        # CCS Cost Summary
        CCS_cost_summary_header = [
            "DCC Column",
            "DCC Packing",
            "DCC Pump",
            "DCC Cooler",
            "Absorber Column",
            "Absorber Packing",
            "WW Pump",
            "WW Cooler",
            "Rich Solvent Pump",
            "Lean Rich Heat Exchanger",
            "Stripper Column",
            "Stripper Packing",
            "Stripper Condenser",
            "Stripper Reboiler",
            "Stripper Reflux Drum",
            "Lean Solvent Pump",
            "Solvent Filtration",
            "Stripper Reflux Pump",
            "Solvent Sump",
            "Solvent Sump Pump",
            "Solvent Sump Filter",
            "Solvent Sump Pit Pump",
            "Pre Scrubber Pump",
            "Reboiler Condensate Pot",
            "DCC Water Filter",
            "Corrosion Inhibition",
            "Antifoam Feed Package",
            "Solvent Storage Tank",
            "NaOH Makeup",
            "NaOH Storage Tank",
            "Foundations",
            "Interconnecting Piping",
            "Station Service Equipment",
            "Switchgear Motor Control",
            "Conduit Cable Tray",
            "Wire Cable",
            "Main Power Transformers",
            "Electrical Foundations",
            "Control Boards and Panels Rack",
            "Distributed Control System Equipment",
            "Instrument Wiring Tubing",
            "Other Instrumentation Controls Equipment",
            "Other Buildings and Structures",
            "Site Improvements",
            ]
        CCS_cost_summary_objects = [
            self.costing.DCC_column_cost,
            self.costing.DCC_packing_cost,
            self.costing.DCC_pump_cost,
            self.costing.DCC_cooler_cost,
            self.costing.absorber_column_cost,
            self.costing.absorber_packing_cost,
            self.costing.ww_pump_cost,
            self.costing.ww_cooler_cost,
            self.costing.rich_solvent_pump_cost,
            self.costing.lean_rich_hex_cost,
            self.costing.stripper_column_cost,
            self.costing.stripper_packing_cost,
            self.costing.stripper_condenser_cost,
            self.costing.stripper_reboiler_cost,
            self.costing.stripper_reflux_drum_cost,
            self.costing.lean_solvent_pump_cost,
            self.costing.solvent_filtration_cost,
            self.costing.stripper_reflux_pump_cost,
            self.costing.Solvent_Sump_cost,
            self.costing.Solvent_Sump_pump_cost,
            self.costing.Solvent_Sump_Filter_cost,
            self.costing.Solvent_Sump_Pit_Pump_cost,
            self.costing.Pre_Scrubber_Pump_cost,
            self.costing.reboiler_condensate_pot_cost,
            self.costing.DCC_Water_Filter_cost,
            self.costing.Corrosion_Inhib_Package_cost,
            self.costing.Antifoam_Feed_Package_cost,
            self.costing.solvent_storage_tank_cost,
            self.costing.NaOH_Makeup_Pump_cost,
            self.costing.NaOH_Storage_Tank_cost,
            self.costing.Foundations_cost,
            self.costing.Interconnecting_Piping_cost,
            self.costing.Station_Service_Equipment_cost,
            self.costing.Switchgear_Motor_Control_cost,
            self.costing.Conduit_Cable_Tray_cost,
            self.costing.Wire_Cable_cost,
            self.costing.Main_Power_Transformers_cost,
            self.costing.Electrical_Foundations_cost,
            self.costing.Control_Boards_Panels_Racks_cost,
            self.costing.Distributed_Control_System_Equipment_cost,
            self.costing.Instrument_Wiring_Tubing_cost,
            self.costing.Other_Instrumentation_Controls_Equipment_cost,
            self.costing.Other_Buildings_Structures_cost,
            self.costing.Site_Improvements_cost,
            ]
        
        CCS_cost_summary_data = [pyo.value(obj) for obj in CCS_cost_summary_objects]
        CCS_cost_summary_units = [
            pyo.units.get_units(obj) for obj in CCS_cost_summary_objects
        ]
        
        CCS_results = {'Component': CCS_cost_summary_header,
                'Value': CCS_cost_summary_data,
                'Unit': CCS_cost_summary_units,
                }
        
        df_out_CCS_results = pd.DataFrame(CCS_results)
            
        cost_summary_header = [
            "NGCC TPC",
        ]
        cost_summary_obj = [
            self.costing.NGCC_TPC,
        ]
        
        cost_summary_data = [pyo.value(obj) for obj in cost_summary_obj]
        cost_summary_units = [pyo.units.get_units(obj) for obj in cost_summary_obj]
        
        cost_summary_results = {'Component': cost_summary_header,
                'Value': cost_summary_data,
                'Unit': cost_summary_units,
                }
        
        df_out_cost_summary_results = pd.DataFrame(cost_summary_results)
        
        process_summary_header = [
            "Plant Net Power (MWe)",
            "CO2 Emissions (kg/hr)",
            "Plant Gross Power (MWe)",
            "Auxiliary Load (MWe)",
        ]
        
        process_summary_obj = [
            pyo.units.convert(self.costing.plant_net_power, to_units=pyo.units.MW),
            pyo.units.convert(self.costing.CO2_emission_cap, 
                              to_units=pyo.units.kg / pyo.units.hr),
            pyo.units.convert(self.costing.plant_gross_power, to_units=pyo.units.MW),
            pyo.units.convert(self.costing.auxiliary_load, to_units=pyo.units.MW),
        ]
        
        process_summary_data = [pyo.value(obj) for obj in process_summary_obj]
        process_summary_units = [pyo.units.get_units(obj) for obj in process_summary_obj]

        process_summary_results = {'Component': process_summary_header,
                'Value': process_summary_data,
                'Unit': process_summary_units,
                }
        
        df_out_process_summary_results = pd.DataFrame(process_summary_results)
        
        # Check if results folder exists (create if it does not exist), save results as .csv
        # Directory name    
        import os  
        
        CCS_costs = os.path.join(directory, file1_name)
        
        NGCC_CCS_costs = os.path.join(directory, file2_name)
        
        process_summary = os.path.join(directory, file3_name)
        
        try:
            os.mkdir(directory)
            df_out_CCS_results.to_csv(CCS_costs) 
            df_out_cost_summary_results.to_csv(NGCC_CCS_costs) 
            df_out_process_summary_results.to_csv(process_summary) 
        except:
            df_out_CCS_results.to_csv(CCS_costs) 
            df_out_cost_summary_results.to_csv(NGCC_CCS_costs) 
            df_out_process_summary_results.to_csv(process_summary) 
            
        return df_out_CCS_results, df_out_process_summary_results 

##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes".
##############################################################################
"""
0D simple boiler model with mass and energy balance only
assume 100% coal burnout (no unburned carbon in fly ash)

"""
# Import Python libraries
import logging
import math

# Import Pyomo libraries
from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (ControlVolume0DBlock,
                        declare_process_block_class,
                        MaterialBalanceType,
                        EnergyBalanceType,
                        MomentumBalanceType,
                        UnitModelBlockData,
                        useDefault)

from idaes.core.util.config import (is_physical_parameter_block,
                                    is_reaction_parameter_block)
from idaes.core.util.misc import add_object_reference
import idaes.core.util.scaling as iscale


# Additional import for the unit operation
from pyomo.environ import SolverFactory, value, Var, Param, exp, sqrt, log, sin
from pyomo.opt import TerminationCondition


__author__ = "Jinliang Ma"
__version__ = "1.0.0"

# Set up logger
_log = logging.getLogger(__name__)

#----------------------------------------------------------------------------------------------------------
@declare_process_block_class("SimpleBoiler0D")
class SimpleBoiler0DData(UnitModelBlockData):
    '''
    Simple 0D boiler model with mass and energy balance only
    '''

    CONFIG = ConfigBlock()
    CONFIG.declare("dynamic", ConfigValue(
        domain=In([useDefault, True, False]),
        default=useDefault,
        description="Dynamic model flag",
        doc="""Indicates whether this model will be dynamic or not,
**default** = useDefault.
**Valid values:** {
**useDefault** - get flag from parent (default = False),
**True** - set as a dynamic model,
**False** - set as a steady-state model.}"""))
    CONFIG.declare("has_holdup", ConfigValue(
        default=False,
        domain=In([True, False]),
        description="Holdup construction flag",
        doc="""Indicates whether holdup terms should be constructed or not.
Must be True if dynamic = True,
**default** - False.
**Valid values:** {
**True** - construct holdup terms,
**False** - do not construct holdup terms}"""))
    CONFIG.declare("side_1_property_package", ConfigValue(
        default=useDefault,
        domain=is_physical_parameter_block,
        description="Property package to use for control volume",
        doc="""Property parameter object used to define property calculations,
**default** - useDefault.
**Valid values:** {
**useDefault** - use default package from parent model or flowsheet,
**PhysicalParameterObject** - a PhysicalParameterBlock object.}"""))
    CONFIG.declare("side_1_property_package_args", ConfigBlock(
        implicit=True,
        description="Arguments to use for constructing property packages",
        doc="""A ConfigBlock with arguments to be passed to a property block(s)
and used when constructing these,
**default** - None.
**Valid values:** {
see property package for documentation.}"""))
    CONFIG.declare("side_2_property_package", ConfigValue(
        default=useDefault,
        domain=is_physical_parameter_block,
        description="Property package to use for control volume",
        doc="""Property parameter object used to define property calculations,
**default** - useDefault.
**Valid values:** {
**useDefault** - use default package from parent model or flowsheet,
**PhysicalParameterObject** - a PhysicalParameterBlock object.}"""))
    CONFIG.declare("side_2_property_package_args", ConfigBlock(
        implicit=True,
        description="Arguments to use for constructing property packages",
        doc="""A ConfigBlock with arguments to be passed to a property block(s)
and used when constructing these,
**default** - None.
**Valid values:** {
see property package for documentation.}"""))
    CONFIG.declare("material_balance_type", ConfigValue(
        default=MaterialBalanceType.componentPhase,
        domain=In(MaterialBalanceType),
        description="Material balance construction flag",
        doc="""Indicates what type of material balance should be constructed,
**default** - MaterialBalanceType.componentPhase.
**Valid values:** {
**MaterialBalanceType.none** - exclude material balances,
**MaterialBalanceType.componentPhase** - use phase component balances,
**MaterialBalanceType.componentTotal** - use total component balances,
**MaterialBalanceType.elementTotal** - use total element balances,
**MaterialBalanceType.total** - use total material balance.}"""))
    CONFIG.declare("energy_balance_type", ConfigValue(
        default=EnergyBalanceType.enthalpyTotal,
        domain=In(EnergyBalanceType),
        description="Energy balance construction flag",
        doc="""Indicates what type of energy balance should be constructed,
**default** - EnergyBalanceType.enthalpyTotal.
**Valid values:** {
**EnergyBalanceType.none** - exclude energy balances,
**EnergyBalanceType.enthalpyTotal** - single ethalpy balance for material,
**EnergyBalanceType.enthalpyPhase** - ethalpy balances for each phase,
**EnergyBalanceType.energyTotal** - single energy balance for material,
**EnergyBalanceType.energyPhase** - energy balances for each phase.}"""))
    CONFIG.declare("momentum_balance_type", ConfigValue(
        default=MomentumBalanceType.pressureTotal,
        domain=In(MomentumBalanceType),
        description="Momentum balance construction flag",
        doc="""Indicates what type of momentum balance should be constructed,
**default** - MomentumBalanceType.pressureTotal.
**Valid values:** {
**MomentumBalanceType.none** - exclude momentum balances,
**MomentumBalanceType.pressureTotal** - single pressure balance for material,
**MomentumBalanceType.pressurePhase** - pressure balances for each phase,
**MomentumBalanceType.momentumTotal** - single momentum balance for material,
**MomentumBalanceType.momentumPhase** - momentum balances for each phase.}"""))
    CONFIG.declare("has_heat_transfer", ConfigValue(
        default=False,
        domain=In([True, False]),
        description="Heat transfer term construction flag",
        doc="""Indicates whether terms for heat transfer should be constructed,
**default** - False.
**Valid values:** {
**True** - include heat transfer terms,
**False** - exclude heat transfer terms.}"""))
    CONFIG.declare("has_pressure_change", ConfigValue(
        default=False,
        domain=In([True, False]),
        description="Pressure change term construction flag",
        doc="""Indicates whether terms for pressure change should be
constructed,
**default** - False.
**Valid values:** {
**True** - include pressure change terms,
**False** - exclude pressure change terms.}"""))


    def build(self):
        # Call TranslatorData build to setup dynamics
        super(SimpleBoiler0DData, self).build()
        # Build Holdup Block

        self.side_1 = ControlVolume0DBlock(default={
        "dynamic": False, #self.config.dynamic,
        "has_holdup": False, #self.config.has_holdup,
        "property_package": self.config.side_1_property_package,
        "property_package_args": self.config.side_1_property_package_args})

        self.side_2 = ControlVolume0DBlock(default={
        "dynamic": False, #self.config.dynamic,
        "has_holdup": False, #self.config.has_holdup,
        "property_package": self.config.side_2_property_package,
        "property_package_args": self.config.side_2_property_package_args})


        # Add Geometry
        self.side_1.add_geometry()
        self.side_2.add_geometry()

        # Add state block
        self.side_1.add_state_blocks(has_phase_equilibrium=False)
        self.side_2.add_state_blocks(has_phase_equilibrium=False)

        # Add material balance, none for side_2 in this case since fuel is added
        self.side_1.add_material_balances(
            balance_type=self.config.material_balance_type)
        self.side_2.add_material_balances(
            balance_type=MaterialBalanceType.none)

        # add energy balance, none in this case
        self.side_1.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_heat_transfer=self.config.has_heat_transfer)
        self.side_2.add_energy_balances(
            balance_type=EnergyBalanceType.none,
            has_heat_transfer=self.config.has_heat_transfer)

        # add momentum balance, pressure change
        self.side_1.add_momentum_balances(
            balance_type=MomentumBalanceType.pressureTotal,
            has_pressure_change=self.config.has_pressure_change)
        self.side_2.add_momentum_balances(
            balance_type=MomentumBalanceType.pressureTotal,
            has_pressure_change=self.config.has_pressure_change)

        if self.config.has_holdup  is True:
            add_object_reference(self, "volume_side_1", self.side_1.volume)
            add_object_reference(self, "volume_side_2", self.side_2.volume)
            # Total tube side valume
            self.Constraint(doc="total tube side volume")
            def volume_side_1_eqn(b):
                return b.volumne_side_1 == 1
            # Total shell side valume
            self.Constraint(doc="total shell side volume")
            def volume_side_2_eqn(b):
                return b.volumne_side_2 == 1

        self.add_inlet_port(name="side_1_inlet", block=self.side_1)
        self.add_inlet_port(name="side_2_inlet", block=self.side_2)
        self.add_outlet_port(name="side_1_outlet", block=self.side_1)
        self.add_outlet_port(name="side_2_outlet", block=self.side_2)

        # Construct performance equations
        self._make_params()
        self._make_vars()
        self._make_mass_balance()
        self._make_energy_balance()

    def _make_params(self):
        ''' This section is for parameters within this model.'''
        self.am_C = Param(initialize=12.01115)
        self.am_H = Param(initialize=1.00797)
        self.am_O = Param(initialize=15.9994)
        self.am_N = Param(initialize=14.0067)
        self.am_S = Param(initialize=32.06)
        self.am_Ar = Param(initialize=39.95)

    def _make_vars(self):
        ''' This section is for variables within this model.'''
        self.temp_coal = Var(self.flowsheet().time,
                        initialize=300,
                        doc='coal temperature')

        self.flowrate_coal = Var(self.flowsheet().time,
                        initialize=25.0,
                        doc='coal mass flowrate [kg/s]')

        self.mf_C_coal_dry = Var(
                        initialize=0.5,
                        doc='Mass fraction of C on dry basis')

        self.mf_H_coal_dry = Var(
                        initialize=0.5,
                        doc='Mass fraction of H on dry basis')

        self.mf_O_coal_dry = Var(
                        initialize=0.5,
                        doc='Mass fraction of O on dry basis')

        self.mf_N_coal_dry = Var(
                        initialize=0.5,
                        doc='Mass fraction of N on dry basis')

        self.mf_S_coal_dry = Var(
                        initialize=0.5,
                        doc='Mass fraction of S on dry basis')

        self.mf_Ash_coal_dry = Var(
                        initialize=0.5,
                        doc='Mass fraction of Ash on dry basis')

        self.mf_H2O_coal = Var(
                        initialize=0.5,
                        doc='Mass fraction of moisture on as received basis')

        self.hhv_coal_dry = Var(
                        initialize=1e7,
                        doc='HHV of coal on dry basis')

        self.mf_ash_flyash = Var(self.flowsheet().time,
                        initialize=0.99,
                        doc='mass fraction of ash in flyash')

        self.molf_NO_fluegas = Var(self.flowsheet().time,
                        initialize=1e-4,
                        doc='mol fraction of NO in flue gas')

        # fraction of daf on dry basis
        @self.Expression(doc="mf daf dry")
        def mf_daf_dry(b):
            return 1-b.mf_Ash_coal_dry

        @self.Expression(self.flowsheet().time, doc="ash flow rate")
        def flowrate_ash(b, t):
            return b.flowrate_coal[t]*(1-b.mf_H2O_coal)*b.mf_Ash_coal_dry

        @self.Expression(self.flowsheet().time, doc="daf coal flow rate in fuel fed to the boiler")
        def flowrate_daf_fuel(b, t):
            return b.flowrate_coal[t]*(1-b.mf_H2O_coal)*b.mf_daf_dry

        @self.Expression(self.flowsheet().time, doc="daf coal flow rate")
        def flowrate_daf_flyash(b, t):
            return b.flowrate_ash[t]*(1.0/b.mf_ash_flyash[t]-1.0)

        @self.Expression(self.flowsheet().time, doc="daf coal flow rate")
        def flowrate_daf_burned(b, t):
            return b.flowrate_daf_fuel[t] - b.flowrate_daf_flyash[t]

        @self.Expression(doc="daf C")
        def mf_C_daf(b):
            return b.mf_C_coal_dry/b.mf_daf_dry

        @self.Expression(doc="daf H")
        def mf_H_daf(b):
            return b.mf_H_coal_dry/b.mf_daf_dry

        @self.Expression(doc="daf O")
        def mf_O_daf(b):
            return b.mf_O_coal_dry/b.mf_daf_dry

        @self.Expression(doc="daf N")
        def mf_N_daf(b):
            return b.mf_N_coal_dry/b.mf_daf_dry

        @self.Expression(doc="daf S")
        def mf_S_daf(b):
            return b.mf_S_coal_dry/b.mf_daf_dry

        @self.Expression(doc="daf hhv")
        def hhv_daf(b):
            return b.hhv_coal_dry/b.mf_daf_dry

        @self.Expression(doc="dhcoal")
        def dhcoal(b):
            return -b.hhv_daf + 8314.3*298.15/2*(-b.mf_H_daf/2/b.am_H+b.mf_O_daf/b.am_O+b.mf_N_daf/b.am_N)

        @self.Expression(doc="dhc")
        def dhc(b):
            return -94052*4.184*1000/b.am_C*b.mf_C_daf

        @self.Expression(doc="dhh")
        def dhh(b):
            return -68317.4*4.184*1000/b.am_H/2*b.mf_H_daf

        @self.Expression(doc="dhs")
        def dhs(b):
            return -70940*4.184*1000/b.am_S*b.mf_S_daf

        @self.Expression(doc="daf hf")
        def hf_daf(b):
            return b.dhc+b.dhh+b.dhs-b.dhcoal

        @self.Expression(doc="hf_coal")
        def hf_coal(b):
            return b.hf_daf*(1-b.mf_H2O_coal)*b.mf_daf_dry + b.mf_H2O_coal*(-68317.4)*4.184*1000/(b.am_H*2+b.am_O)

        @self.Expression(doc="a_daf")
        def a_daf(b):
            return 1/(b.mf_C_daf/b.am_C+b.mf_H_daf/b.am_H+b.mf_O_daf/b.am_O+b.mf_N_daf/b.am_N+b.mf_S_daf/b.am_S)

        @self.Expression(self.flowsheet().time, doc="gt1")
        def gt1(b, t):
            return 1/(exp(380/b.temp_coal[t])-1)

        @self.Expression(self.flowsheet().time, doc="gt1 for fly ash daf part")
        def gt1_flyash(b, t):
            return 1/(exp(380/b.side_2.properties_out[t].temperature)-1)

        @self.Expression(self.flowsheet().time, doc="gt2")
        def gt2(b, t):
            return 1/(exp(1800/b.temp_coal[t])-1)

        @self.Expression(self.flowsheet().time, doc="gt2 for fly ash daf part")
        def gt2_flyash(b, t):
            return 1/(exp(1800/b.side_2.properties_out[t].temperature)-1)

        @self.Expression(self.flowsheet().time, doc="hs_daf")
        def hs_daf(b, t):
            return 8314.3/b.a_daf*(380*(b.gt1[t]-0.3880471566)+3600*(b.gt2[t]-0.002393883))

        @self.Expression(self.flowsheet().time, doc="hs_daf for fly ash daf part")
        def hs_daf_flyash(b, t):
            return 8314.3/b.a_daf*(380*(b.gt1_flyash[t]-0.3880471566)+3600*(b.gt2_flyash[t]-0.002393883))

        @self.Expression(self.flowsheet().time, doc="hs_coal")
        def hs_coal(b, t):
            return (1-b.mf_H2O_coal)*b.mf_daf_dry*b.hs_daf[t]+b.mf_H2O_coal*4184*(b.temp_coal[t]-298.15)+\
                   (1-b.mf_H2O_coal)*b.mf_Ash_coal_dry*(593*(b.temp_coal[t]-298.15)+0.293*(b.temp_coal[t]**2-298.15**2))

        @self.Expression(self.flowsheet().time, doc="h_coal")
        def h_coal(b, t):
            return b.hs_coal[t]+b.hf_coal

    def _make_mass_balance(self):

        self.molflow_C = Var(self.flowsheet().time,
                        initialize=1,
                        doc='Mole flow of C')

        self.molflow_H = Var(self.flowsheet().time,
                        initialize=1,
                        doc='Mole flow of H')

        self.molflow_O = Var(self.flowsheet().time,
                        initialize=1,
                        doc='Mole flow of O')

        self.molflow_N = Var(self.flowsheet().time,
                        initialize=1,
                        doc='Mole flow of N')

        self.molflow_S = Var(self.flowsheet().time,
                        initialize=1,
                        doc='Mole flow of S')


        @self.Constraint(self.flowsheet().time, doc="C mole flow")
        def molflow_C_eqn(b, t):
            return b.molflow_C[t] == b.flowrate_daf_burned[t]*b.mf_C_daf*1000/b.am_C + \
                   b.side_2_inlet.flow_mol_comp[t,"CO2"]

        @self.Constraint(self.flowsheet().time, doc="H mole flow")
        def molflow_H_eqn(b, t):
            return b.molflow_H[t] == b.flowrate_daf_burned[t]*b.mf_H_daf*1000/b.am_H + \
                   b.side_2_inlet.flow_mol_comp[t,"H2O"]*2 + b.flowrate_coal[t]*b.mf_H2O_coal*1000/ \
                   (b.am_H*2+b.am_O)*2

        @self.Constraint(self.flowsheet().time, doc="O mole flow")
        def molflow_O_eqn(b, t):
            return b.molflow_O[t] == b.flowrate_daf_burned[t]*b.mf_O_daf*1000/b.am_O + \
                   b.side_2_inlet.flow_mol_comp[t,"O2"]*2 + b.side_2_inlet.flow_mol_comp[t,"CO2"]*2 + \
                   b.side_2_inlet.flow_mol_comp[t,"H2O"] + b.side_2_inlet.flow_mol_comp[t,"SO2"]*2 + \
                   b.flowrate_coal[t]*b.mf_H2O_coal*1000/(b.am_H*2+b.am_O)

        @self.Constraint(self.flowsheet().time, doc="N mole flow")
        def molflow_N_eqn(b, t):
            return b.molflow_N[t] == b.flowrate_daf_burned[t]*b.mf_N_daf*1000/b.am_N + \
                   b.side_2_inlet.flow_mol_comp[t,"N2"]*2

        @self.Constraint(self.flowsheet().time, doc="S mole flow")
        def molflow_S_eqn(b, t):
            return b.molflow_S[t] == b.flowrate_daf_burned[t]*b.mf_S_daf*1000/b.am_S + \
                   b.side_2_inlet.flow_mol_comp[t,"SO2"]

        # NO at outlet, calculated based on user specified molf_NO_fluegas[]
        @self.Constraint(self.flowsheet().time, doc="NO at outlet")
        def NO_eqn(b, t):
            return b.side_2_outlet.flow_mol_comp[t,"NO"] == b.molf_NO_fluegas[t]*\
                   sum(b.side_2.properties_out[t].flow_mol_comp[j] for j in b.side_2.properties_out[t].config.parameters.component_list)

        # N2 at outlet
        @self.Constraint(self.flowsheet().time, doc="N2 at outlet")
        def N2_eqn(b, t):
            return b.side_2_outlet.flow_mol_comp[t,"N2"] == b.molflow_N[t]/2

        # SO2 at outlet
        @self.Constraint(self.flowsheet().time, doc="SO2 at outlet")
        def SO2_eqn(b, t):
            return b.side_2_outlet.flow_mol_comp[t,"SO2"] == b.molflow_S[t]

        # H2O at outlet
        @self.Constraint(self.flowsheet().time, doc="H2O at outlet")
        def H2O_eqn(b, t):
            return b.side_2_outlet.flow_mol_comp[t,"H2O"] == b.molflow_H[t]/2

        # CO2 at outlet
        @self.Constraint(self.flowsheet().time, doc="CO2 at outlet")
        def CO2_eqn(b, t):
            return b.side_2_outlet.flow_mol_comp[t,"CO2"] == b.molflow_C[t]

        # O2 at outlet
        @self.Constraint(self.flowsheet().time, doc="O2 at outlet")
        def O2_eqn(b, t):
            return b.side_2_outlet.flow_mol_comp[t,"O2"] == (b.molflow_O[t] - b.molflow_C[t]*2 - \
                   b.molflow_H[t]/2 - b.molflow_S[t]*2)/2

        self.flowrate_tca = Var(self.flowsheet().time,
                        initialize=200,
                        doc='total combustion air mass flow rate')

        # total combustion air mass flow rate
        @self.Constraint(self.flowsheet().time, doc="total combustion air mass flow rate")
        def flowrate_tca_eqn(b, t):
            return b.flowrate_tca[t]*1000 == b.side_2_inlet.flow_mol_comp[t,"O2"]*b.am_O*2+\
                   b.side_2_inlet.flow_mol_comp[t,"N2"]*b.am_N*2+\
                   b.side_2_inlet.flow_mol_comp[t,"CO2"]*(b.am_C+b.am_O*2)+\
                   b.side_2_inlet.flow_mol_comp[t,"SO2"]*(b.am_S+b.am_O*2)+\
                   b.side_2_inlet.flow_mol_comp[t,"H2O"]*(b.am_O+b.am_H*2)

        self.fluegas_o2_pct_dry = Var(self.flowsheet().time,
                        initialize=3,
                        doc='mol percent of O2 on dry basis')

        # mol percent of O2 on dry basis
        @self.Constraint(self.flowsheet().time, doc="mol percent of O2 on dry basis")
        def fluegas_o2_pct_dry_eqn(b, t):
            return b.fluegas_o2_pct_dry[t]*(b.side_2_outlet.flow_mol_comp[t,"O2"]+\
                   b.side_2_outlet.flow_mol_comp[t,"N2"]+b.side_2_outlet.flow_mol_comp[t,"CO2"]+\
                   b.side_2_outlet.flow_mol_comp[t,"SO2"])/100 \
                   == b.side_2_outlet.flow_mol_comp[t,"O2"]


    def _make_energy_balance(self):
        # heat to water wall
        self.heat_ww = Var(self.flowsheet().time,
                        initialize=2e8,
                        doc='heat to water wall')

        # heat to roof superheater
        self.heat_roof = Var(self.flowsheet().time,
                        initialize=5e6,
                        doc='heat to roof superheater')

        # heat to platen superheater
        self.heat_plsh = Var(self.flowsheet().time,
                        initialize=4e7,
                        doc='heat to platen superheater')

        # heat duty to side 1
        @self.Constraint(self.flowsheet().time, doc="heat duty")
        def side_1_heat_eqn(b, t):
            return b.side_1.heat[t] == b.heat_ww[t]

        # enthalpy at side 1 outlet based on iapws95's enth_mol_sat_phase["Vap"] property
        @self.Constraint(self.flowsheet().time, doc="saturated vapor enthalpy")
        def side_1_outlet_enthalpy_eqn(b, t):
            return b.side_1.properties_out[t].enth_mol == b.side_1.properties_out[t].enth_mol_sat_phase["Vap"]

        # enthalpy at side 2 outlet
        @self.Constraint(self.flowsheet().time, doc="temperature at outlet")
        def side_2_temp_eqn(b, t):
            return sum(b.side_2.properties_in[t].flow_mol_comp[j] for j in b.side_2.properties_in[t].config.parameters.component_list)*\
                   b.side_2.properties_in[t].enth_mol + b.h_coal[t]*b.flowrate_coal[t] == \
                   sum(b.side_2.properties_out[t].flow_mol_comp[j] for j in b.side_2.properties_out[t].config.parameters.component_list)*\
                   b.side_2.properties_out[t].enth_mol + b.flowrate_ash[t]*\
                   (593*(b.side_2.properties_out[t].temperature-298.15)+0.293*\
                   (b.side_2.properties_out[t].temperature**2-298.15**2)) +\
                   b.flowrate_daf_flyash[t]*b.hs_daf_flyash[t] + b.heat_ww[t] + b.heat_roof[t] + b.heat_plsh[t]


    def initialize(blk, state_args_1={}, state_args_2={},
                   outlvl=0, solver='ipopt', optarg={'tol': 1e-6}):
        '''
        General Heat Exchanger initialisation routine.

        Keyword Arguments:
            state_args_1 : a dict of arguments to be passed to the property
                           package(s) for side 1 of the heat exchanger to
                           provide an initial state for initialization
                           (see documentation of the specific property package)
                           (default = {}).
            state_args_2 : a dict of arguments to be passed to the property
                           package(s) for side 2 of the heat exchanger to
                           provide an initial state for initialization
                           (see documentation of the specific property package)
                           (default = {}).
            outlvl : sets output level of initialisation routine

                     * 0 = no output (default)
                     * 1 = return solver state for each step in routine
                     * 2 = return solver state for each step in subroutines
                     * 3 = include solver output infomation (tee=True)

            optarg : solver options dictionary object (default={'tol': 1e-6})
            solver : str indicating whcih solver to use during
                     initialization (default = 'ipopt')

        Returns:
            None
        '''
        # Set solver options
        if outlvl > 3:
            stee = True
        else:
            stee = False

        opt = SolverFactory(solver)
        opt.options = optarg

        # ---------------------------------------------------------------------
        # Initialize inlet property blocks
        flags1 = blk.side_1.initialize(outlvl=outlvl-1,
                                       optarg=optarg,
                                       solver=solver,
                                       state_args=state_args_1)

        flags2 = blk.side_2.initialize(outlvl=outlvl-1,
                                       optarg=optarg,
                                       solver=solver,
                                       state_args=state_args_2)
        if outlvl > 0:
            _log.info('{} Initialisation Step 1 Complete.'.format(blk.name))


        # Initialize temperature differentials
        for t in blk.flowsheet().time:
            blk.side_1.properties_out[t].enth_mol.fix(
                    value(blk.side_1.properties_in[t].enth_mol)+1000.0)
            blk.side_2.properties_out[t].temperature.fix(1300)
        # Deactivate Constraints
        blk.side_1_heat_eqn.deactivate()
        blk.side_1_outlet_enthalpy_eqn.deactivate()

        results = opt.solve(blk, tee=stee)
        print ("---------------Step 1 for Simple Boiler Initialization Completed---------------")

        for t in blk.flowsheet().time:
            blk.side_1.properties_out[t].enth_mol.unfix()
            blk.side_2.properties_out[t].temperature.unfix()
        blk.side_1_heat_eqn.activate()
        blk.side_1_outlet_enthalpy_eqn.activate()

        results = opt.solve(blk, tee=stee)

        if outlvl > 0:
            if results.solver.termination_condition == \
                    TerminationCondition.optimal:
                _log.info('{} Initialisation Step 2 Complete.'
                            .format(blk.name))
            else:
                _log.warning('{} Initialisation Step 2 Failed.'
                               .format(blk.name))
        # ---------------------------------------------------------------------
        # Release Inlet state
        blk.side_1.release_state(flags1, outlvl-1)
        blk.side_2.release_state(flags2, outlvl-1)

        if outlvl > 0:
            _log.info('{} Initialisation Complete.'.format(blk.name))
        print ("***************** End of Simple Boiler 0D Model Initialization ***************")

    def calculate_scaling_factors(self):
        pass

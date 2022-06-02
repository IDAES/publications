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
Fireside boiler surrogate model with mass and energy balance enforced
Flue gas compositions of major species O2, N2, CO2, H2O and SO2 are calculated based on elemental mass balance
Flue gas composition of NO is calculated based on surrogate model algebraic function
Unburned organic elements are calculated based on surrogate model algebraic function
Heat duties in waterwall zones and platen SH and roof are calculated based on surrogate model functions
FEGT is calculated based on energy balance

The model has two inlets:
      side_1_inlet: PA with vaporized moisture (after mill), flow rate and composition calculated by model
      side_2_inlet: SA, flow rate and composition calculated by the model

The model has two outlets:
      side_1_outlet: Flue gas (gas only), flow rate and composition calculated by mass balance
      side_2_outlet: Fly ash with unburned organic elements (solid phase only), currently treated as N2 gas at outlet


Note: 1. SR versus raw coal flow rate can be specified by a flowsheet constraint and should be consistent with the surrogate model.
      2. PA to raw coal flow rate ratio is can be specified by a flowsheet constraint
      3. Fraction of coal moisture vaporized in mill should be consistent with the surrogate model (typically 0.6)
      4. Moisture mass fration in raw coal is a user input and should be consistent with the surrogate model
      5. side_1_inlet is PA plus vaporized moisture at PA temperature after mill (typically 150 F). The composition and flow rate
         of side_1_inlet is calculated by constraints in this model
      6. side_2_inlet is SA including OFA with temperature as a user input variable. It can be connected with APH outlet.
      7. side_1_outlet is the flue gas leaving boiler (gas phase only)
      8. side_2_outlet is the fly ash with unburned organic elements (currently treated as N2 gas)
      9. please don't connect the PA inlets with upstream unit operations since PA flow and compostions are calculated in the model
      10. outlet streams can be connected to downstream unit operations. side_2_outlet is typically not connected with any unit models

Surrogate model inputs include
      1-16. slag layer wall temperature of 14 water wall zones, planten SH, and roof
      17. raw coal mass flow rate
      18. moisture mass fraction of raw coal
      19. overall Stoichiometric ratio
      20. lower furnace Stoichiometric ratio
      21. SA temperature

"""
# Import Python libraries
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

from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.core.util.config import (is_physical_parameter_block,
                                    is_reaction_parameter_block)
from idaes.core.util.misc import add_object_reference
import idaes.logger as idaeslog


# Additional import for the unit operation
from pyomo.environ import SolverFactory, value, Var, Param, exp, sqrt, log, sin, RangeSet, Reals, Constraint
from pyomo.opt import TerminationCondition
import idaes.core.util.scaling as iscale


__author__ = "Jinliang Ma"
__version__ = "1.0.0"


#----------------------------------------------------------------------------------------------------------
@declare_process_block_class("BoilerSurrogate")
class BoilerSurrogateData(UnitModelBlockData):
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
        super(BoilerSurrogateData, self).build()
        # Build Holdup Block

        # side_1 is PA for inlet and flue gas for outlet (excluding solid)
        self.side_1 = ControlVolume0DBlock(default={
        "dynamic": self.config.dynamic,
        "has_holdup": self.config.has_holdup,
        "property_package": self.config.side_1_property_package,
        "property_package_args": self.config.side_1_property_package_args})

        # side_2 is SA for inlet and solid of fly ash with unburned organic elements for outlet, treated as N2 gas
        self.side_2 = ControlVolume0DBlock(default={
        "dynamic": self.config.dynamic,
        "has_holdup": self.config.has_holdup,
        "property_package": self.config.side_2_property_package,
        "property_package_args": self.config.side_2_property_package_args})

        # Add Geometry
        if self.config.has_holdup  is True:
            self.side_1.add_geometry()
            self.side_2.add_geometry()

        # Add state block
        self.side_1.add_state_blocks(has_phase_equilibrium=False)
        self.side_2.add_state_blocks(has_phase_equilibrium=False)

        # Add material balance, none since fuel is added without inlet stream (not using framework mass balance)
        self.side_1.add_material_balances(
            balance_type=MaterialBalanceType.none)
        self.side_2.add_material_balances(
            balance_type=MaterialBalanceType.none)

        # add energy balance, none in this case (not using framework energy balance)
        self.side_1.add_energy_balances(
            balance_type=EnergyBalanceType.none,
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
            # PA side
            self.Constraint(doc="side_1 volume")
            def volume_side_1_eqn(b):
                return b.volumne_side_1 == 1
            # SA side
            self.Constraint(doc="side_2 volume")
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
        # atomic mass of elements involved in kg/mol
        self.am_C = Param(initialize=0.01201115)
        self.am_H = Param(initialize=0.00100797)
        self.am_O = Param(initialize=0.0159994)
        self.am_N = Param(initialize=0.0140067)
        self.am_S = Param(initialize=0.03206)
        # air composition in terms of mole fractions of O2, N2, H2O, and CO2 at 25C and relative humidity of 25%
        # please change air compostion accordingly if relative humidity changes
        self.molf_O2_air =  Param(initialize=0.20784)
        self.molf_N2_air =  Param(initialize=0.7839958)
        self.molf_H2O_air = Param(initialize=0.0078267)
        self.molf_CO2_air = Param(initialize=0.0003373)
        self.molf_SO2_air = Param(initialize=0.0000001)
        self.molf_NO_air =  Param(initialize=0.0000001)
        # molecular weight of air at 25% relative humidity, unit is kg/mol
        self.mw_air = Param(initialize=0.0287689)


    def _make_vars(self):
        ''' This section is for variables within this model.'''
        # Number of zones, 14 water wall zones, 1 platen SH zone and 1 roof zone
        self.zones = RangeSet(16)

        self.fheat_ww = Var(self.flowsheet().config.time, initialize=1,
                        doc='correction factor for waterwall heat duty')

        self.fheat_platen = Var(self.flowsheet().config.time, initialize=1,
                        doc='correction factor for platen SH heat duty')

        # wall temperatures of 16 zones
        self.in_WWTemp = Var(self.flowsheet().config.time, self.zones,
                        domain=Reals,
                        initialize=700.0,
                        doc='waterwall zone temp [K]')

        # heat duties of 16 zones
        self.out_heat = Var(self.flowsheet().config.time, self.zones,
                        domain=Reals,
                        initialize=2.2222e7,
                        doc='Zone heat duty [W]')

        # PA or coal temperature, usually fixed around 150 F
        self.temp_coal = Var(self.flowsheet().config.time,
                        initialize=338.7,
                        doc='coal temperature')

        # SA temperature, typically from 550 to 600 F
        self.in_2ndAir_Temp = Var(self.flowsheet().config.time,
                        initialize=560.9,
                        doc='SA temperature')

        # Stoichiometric ratio, used to calculate total combustion air flow rate
        # If SR is a function of load or raw coal flow rate, specify constraint in flowsheet model
        self.SR = Var(self.flowsheet().config.time,
                        initialize=1.15,
                        doc='Overall furnace SR')

        # lower furnace Stoichiometric ratio
        self.SR_lf = Var(self.flowsheet().config.time,
                        initialize=1.15,
                        doc='Overall furnace SR')

        # PA to coal mass flow ratio, typically 2.0 depending on load or mill curve
        self.ratio_PA2coal = Var(self.flowsheet().config.time,
                        initialize=2.0,
                        doc='Primary air to coal ratio')

        # Coal flow rate before mill (raw coal flow without moisture vaporization in mill)
        self.flowrate_coal_raw = Var(self.flowsheet().config.time,
                        initialize=25.0,
                        doc='coal mass flowrate [kg/s]')

        # Coal flow rate to burners after moisture vaporization in mill
        self.flowrate_coal = Var(self.flowsheet().config.time,
                        initialize=20.0,
                        doc='coal mass flowrate [kg/s]')

        # Raw coal moisure content (on as received basis), this could change with time
        self.mf_H2O_coal_raw = Var(self.flowsheet().config.time,
                        initialize=0.15,
                        doc='Raw coal mass fraction of moisture on as received basis')

        # coal moisure content after mill, calculated based on fraction of moistures vaporized in mill
        self.mf_H2O_coal = Var(self.flowsheet().config.time,
                        initialize=0.15,
                        doc='Mass fraction of moisture on as received basis')

        # Fraction of moisture vaporized in mill, set in flowsheet as a function of coal flow rate, default is 0.6
        self.frac_moisture_vaporized = Var(self.flowsheet().config.time,
                        initialize=0.6,
                        doc='fraction of coal moisture vaporized in mill')

        # Vaporized moisture mass flow rate
        @self.Expression(self.flowsheet().config.time, doc="vaporized moisture mass flowrate [kg/s]")
        def flowrate_moist_vaporized(b, t):
            return b.flowrate_coal_raw[t]*b.mf_H2O_coal_raw[t]*b.frac_moisture_vaporized[t]

        # PA mass flow rate before mills
        self.flowrate_PA = Var(self.flowsheet().config.time,
                        initialize=50.0,
                        doc='PA mass flowrate [kg/s]')

        # SA mass flow rate
        self.flowrate_SA = Var(self.flowsheet().config.time,
                        initialize=50.0,
                        doc='SA mass flowrate [kg/s]')

        # Total combustion air mass flow rate
        self.flowrate_TCA = Var(self.flowsheet().config.time,
                        initialize=50.0,
                        doc='total combustion air mass flowrate [kg/s]')

        # Elemental composition of dry coal, assuming not change with time
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

        # HHV of dry coal, assuming not changing with time
        self.hhv_coal_dry = Var(
                        initialize=1e7,
                        doc='HHV of coal on dry basis')

        # Fraction of unburned carbon (actually all organic elements) in fly ash, predicted by surrogate model
        # When generating the surrogate, sum up all elements in fly ash from fireside boiler model outputs
        self.out_ubc_flyash = Var(self.flowsheet().config.time,
                        initialize=0.01,
                        doc='unburned carbon and other organic elements in fly ash')

        # Mass fraction of NOx in flue gas, predicted by surrogate model
        self.out_mf_NO_FG = Var(self.flowsheet().config.time,
                        initialize=1e-4,
                        doc='Mass fraction of NOx in flue gas')

        @self.Expression(self.flowsheet().config.time)
        def out_ppm_NO_FG(b,t):
            return b.out_mf_NO_FG[t]*1e6

        # NOx in lb/MMBTU
        @self.Expression(self.flowsheet().config.time, doc="NOx in lb/MMBTU")
        def nox_lb_mmbtu(b, t):
            return b.side_1_outlet.flow_mol_comp[t,"NO"]*0.046*2.20462/\
                   (b.flowrate_coal_raw[t]*(1-b.mf_H2O_coal_raw[t])*b.hhv_coal_dry/1.054e9)

        # coal flow rate after mill
        @self.Constraint(self.flowsheet().config.time, doc="Coal flow rate to burners after mills")
        def flowrate_coal_eqn(b, t):
            return b.flowrate_coal[t] == b.flowrate_coal_raw[t] - b.flowrate_moist_vaporized[t]

        # moisture mass fraction in coal after mill
        @self.Constraint(self.flowsheet().config.time, doc="moisture mass fraction for coal after mill")
        def mf_H2O_coal_eqn(b, t):
            return b.flowrate_coal_raw[t]*b.mf_H2O_coal_raw[t] == b.flowrate_coal[t]*b.mf_H2O_coal[t] + b.flowrate_moist_vaporized[t]

        # fraction of daf elements on dry basis
        @self.Expression(doc="mf daf dry")
        def mf_daf_dry(b):
            return 1-b.mf_Ash_coal_dry

        @self.Expression(self.flowsheet().config.time, doc="ash flow rate")
        def flowrate_ash(b, t):
            return b.flowrate_coal[t]*(1-b.mf_H2O_coal[t])*b.mf_Ash_coal_dry

        @self.Expression(self.flowsheet().config.time, doc="daf coal flow rate in fuel fed to the boiler")
        def flowrate_daf_fuel(b, t):
            return b.flowrate_coal[t]*(1-b.mf_H2O_coal[t])*b.mf_daf_dry

        @self.Expression(self.flowsheet().config.time, doc="daf coal flow rate in fly ash")
        def flowrate_daf_flyash(b, t):
            return b.flowrate_ash[t]/(1.0-b.out_ubc_flyash[t])*b.out_ubc_flyash[t]

        @self.Expression(self.flowsheet().config.time, doc="burnerd daf coal flow rate")
        def flowrate_daf_burned(b, t):
            return b.flowrate_daf_fuel[t] - b.flowrate_daf_flyash[t]

        @self.Expression(self.flowsheet().config.time, doc="coal burnout")
        def coal_burnout(b, t):
            return b.flowrate_daf_burned[t]/b.flowrate_daf_fuel[t]

        @self.Expression(doc="daf C mass fraction")
        def mf_C_daf(b):
            return b.mf_C_coal_dry/b.mf_daf_dry

        @self.Expression(doc="daf H mass fraction")
        def mf_H_daf(b):
            return b.mf_H_coal_dry/b.mf_daf_dry

        @self.Expression(doc="daf O mass fraction")
        def mf_O_daf(b):
            return b.mf_O_coal_dry/b.mf_daf_dry

        @self.Expression(doc="daf N mass fraction")
        def mf_N_daf(b):
            return b.mf_N_coal_dry/b.mf_daf_dry

        @self.Expression(doc="daf S mass fraction")
        def mf_S_daf(b):
            return b.mf_S_coal_dry/b.mf_daf_dry

        @self.Expression(doc="daf hhv")
        def hhv_daf(b):
            return b.hhv_coal_dry/b.mf_daf_dry

        @self.Expression(doc="dhcoal")
        def dhcoal(b):
            return -b.hhv_daf + 8.3143*298.15/2*(-b.mf_H_daf/2/b.am_H+b.mf_O_daf/b.am_O+b.mf_N_daf/b.am_N)

        @self.Expression(doc="dhc")
        def dhc(b):
            return -94052*4.184/b.am_C*b.mf_C_daf

        @self.Expression(doc="dhh")
        def dhh(b):
            return -68317.4*4.184/b.am_H/2*b.mf_H_daf

        @self.Expression(doc="dhs")
        def dhs(b):
            return -70940*4.184/b.am_S*b.mf_S_daf

        @self.Expression(doc="daf hf")
        def hf_daf(b):
            return b.dhc+b.dhh+b.dhs-b.dhcoal

        @self.Expression(self.flowsheet().config.time, doc="hf_coal, coal heat of formation")
        def hf_coal(b,t):
            return b.hf_daf*(1-b.mf_H2O_coal[t])*b.mf_daf_dry + b.mf_H2O_coal[t]*(-68317.4)*4.184/(b.am_H*2+b.am_O)

        @self.Expression(doc="average atomic mass in kg/mol")
        def a_daf(b):
            return 1/(b.mf_C_daf/b.am_C+b.mf_H_daf/b.am_H+b.mf_O_daf/b.am_O+b.mf_N_daf/b.am_N+b.mf_S_daf/b.am_S)

        @self.Expression(self.flowsheet().config.time, doc="gt1")
        def gt1(b, t):
            return 1/(exp(380/b.temp_coal[t])-1)

        #declare variable rather than use expression since out temperature is not fixed
        self.gt1_flyash = Var(self.flowsheet().config.time,
                        initialize=1,
                        doc='gt1 for fly ash daf part')

        @self.Constraint(self.flowsheet().config.time, doc="gt1 for fly ash daf part")
        def gt1_flyash_eqn(b, t):
            return b.gt1_flyash[t] * (exp(380/b.side_1.properties_out[t].temperature)-1) == 1

        @self.Expression(self.flowsheet().config.time, doc="gt2")
        def gt2(b, t):
            return 1/(exp(1800/b.temp_coal[t])-1)

        #declare variable rather than use expression since out temperature is not fixed
        self.gt2_flyash = Var(self.flowsheet().config.time,
                        initialize=1,
                        doc='gt2 for fly ash daf part')

        @self.Constraint(self.flowsheet().config.time, doc="gt2 for fly ash daf part")
        def gt2_flyash_eqn(b, t):
            return b.gt2_flyash[t] * (exp(1800/b.side_1.properties_out[t].temperature)-1) == 1

        @self.Expression(self.flowsheet().config.time, doc="hs_daf")
        def hs_daf(b, t):
            return 8.3143/b.a_daf*(380*(b.gt1[t]-0.3880471566)+3600*(b.gt2[t]-0.002393883))

        @self.Expression(self.flowsheet().config.time, doc="hs_daf for fly ash daf part")
        def hs_daf_flyash(b, t):
            return 8.3143/b.a_daf*(380*(b.gt1_flyash[t]-0.3880471566)+3600*(b.gt2_flyash[t]-0.002393883))

        @self.Expression(self.flowsheet().config.time, doc="hs_coal, sensible heat of coal")
        def hs_coal(b, t):
            return (1-b.mf_H2O_coal[t])*b.mf_daf_dry*b.hs_daf[t]+b.mf_H2O_coal[t]*4184*(b.temp_coal[t]-298.15)+\
                   (1-b.mf_H2O_coal[t])*b.mf_Ash_coal_dry*(593*(b.temp_coal[t]-298.15)+0.293*(b.temp_coal[t]**2-298.15**2))

        @self.Expression(self.flowsheet().config.time, doc="h_coal, total enthalpy of coal")
        def h_coal(b, t):
            return b.hs_coal[t]+b.hf_coal[t]

    def _make_mass_balance(self):
        # scaling factor for minor species
        sff_minor = 100
        # PA flow rate calculated based on PA/coal ratio
        @self.Constraint(self.flowsheet().config.time, doc="PA mass flow rate")
        def flowrate_PA_eqn(b, t):
            return b.flowrate_PA[t] == b.flowrate_coal_raw[t]*b.ratio_PA2coal[t]

        @self.Expression(self.flowsheet().config.time, doc="C molar flow from coal")
        def molflow_C_fuel(b, t):
            return b.flowrate_daf_fuel[t]*b.mf_C_daf/b.am_C

        @self.Expression(self.flowsheet().config.time, doc="H molar flow from coal")
        def molflow_H_fuel(b, t):
            return b.flowrate_daf_fuel[t]*b.mf_H_daf/b.am_H

        @self.Expression(self.flowsheet().config.time, doc="O molar flow from coal")
        def molflow_O_fuel(b, t):
            return b.flowrate_daf_fuel[t]*b.mf_O_daf/b.am_O

        @self.Expression(self.flowsheet().config.time, doc="N molar flow from coal")
        def molflow_N_fuel(b, t):
            return b.flowrate_daf_fuel[t]*b.mf_N_daf/b.am_N

        @self.Expression(self.flowsheet().config.time, doc="S molar flow from coal")
        def molflow_S_fuel(b, t):
            return b.flowrate_daf_fuel[t]*b.mf_S_daf/b.am_S

        # Equation between Stoichiometric ratio and total combustion air
        @self.Constraint(self.flowsheet().config.time, doc="SR equation")
        def SR_eqn(b, t):
            return b.SR[t]*(b.molflow_C_fuel[t]+b.molflow_H_fuel[t]/4+b.molflow_S_fuel[t]-b.molflow_O_fuel[t]/2)==\
                   b.flowrate_TCA[t]/b.mw_air*b.molf_O2_air

        # Equation to calculate SA
        @self.Constraint(self.flowsheet().config.time, doc="SA flow rate equation")
        def flowrate_SA_eqn(b, t):
            return b.flowrate_SA[t] + b.flowrate_PA[t] == b.flowrate_TCA[t]

        # Calculate PA component flow including vaporized moisture
        @self.Constraint(self.flowsheet().config.time, doc="PA O2 molar flow")
        def molar_flow_O2_PA_eqn(b, t):
            return b.side_1_inlet.flow_mol_comp[t,"O2"] == b.flowrate_PA[t]/b.mw_air*b.molf_O2_air

        @self.Constraint(self.flowsheet().config.time, doc="PA N2 molar flow")
        def molar_flow_N2_PA_eqn(b, t):
            return b.side_1_inlet.flow_mol_comp[t,"N2"] == b.flowrate_PA[t]/b.mw_air*b.molf_N2_air

        @self.Constraint(self.flowsheet().config.time, doc="PA CO2 molar flow")
        def molar_flow_CO2_PA_eqn(b, t):
            return sff_minor * b.side_1_inlet.flow_mol_comp[t,"CO2"] == sff_minor * b.flowrate_PA[t]/b.mw_air*b.molf_CO2_air

        @self.Constraint(self.flowsheet().config.time, doc="PA H2O molar flow including vaporized moisture in coal")
        def molar_flow_H2O_PA_eqn(b, t):
            return b.side_1_inlet.flow_mol_comp[t,"H2O"] == b.flowrate_PA[t]/b.mw_air*b.molf_H2O_air\
                   + b.flowrate_moist_vaporized[t]/(b.am_H*2+b.am_O)

        @self.Constraint(self.flowsheet().config.time, doc="PA SO2 molar flow")
        def molar_flow_SO2_PA_eqn(b, t):
            return sff_minor * b.side_1_inlet.flow_mol_comp[t,"SO2"] == sff_minor * b.flowrate_PA[t]/b.mw_air*b.molf_SO2_air

        @self.Constraint(self.flowsheet().config.time, doc="PA NO molar flow")
        def molar_flow_NO_PA_eqn(b, t):
            return sff_minor * b.side_1_inlet.flow_mol_comp[t,"NO"] == sff_minor * b.flowrate_PA[t]/b.mw_air*b.molf_NO_air


        # Calculate SA component flow
        @self.Constraint(self.flowsheet().config.time, doc="SA O2 molar flow")
        def molar_flow_O2_SA_eqn(b, t):
            return b.side_2_inlet.flow_mol_comp[t,"O2"] == b.flowrate_SA[t]/b.mw_air*b.molf_O2_air

        @self.Constraint(self.flowsheet().config.time, doc="SA N2 molar flow")
        def molar_flow_N2_SA_eqn(b, t):
            return b.side_2_inlet.flow_mol_comp[t,"N2"] == b.flowrate_SA[t]/b.mw_air*b.molf_N2_air

        @self.Constraint(self.flowsheet().config.time, doc="SA CO2 molar flow")
        def molar_flow_CO2_SA_eqn(b, t):
            return sff_minor * b.side_2_inlet.flow_mol_comp[t,"CO2"] == sff_minor * b.flowrate_SA[t]/b.mw_air*b.molf_CO2_air

        @self.Constraint(self.flowsheet().config.time, doc="SA H2O molar flow")
        def molar_flow_H2O_SA_eqn(b, t):
            return b.side_2_inlet.flow_mol_comp[t,"H2O"] == b.flowrate_SA[t]/b.mw_air*b.molf_H2O_air

        @self.Constraint(self.flowsheet().config.time, doc="SA SO2 molar flow")
        def molar_flow_SO2_SA_eqn(b, t):
            return sff_minor * b.side_2_inlet.flow_mol_comp[t,"SO2"] == sff_minor * b.flowrate_SA[t]/b.mw_air*b.molf_SO2_air

        @self.Constraint(self.flowsheet().config.time, doc="SA NO molar flow")
        def molar_flow_NO_SA_eqn(b, t):
            return sff_minor * b.side_2_inlet.flow_mol_comp[t,"NO"] == sff_minor * b.flowrate_SA[t]/b.mw_air*b.molf_NO_air

        # molar flow rate of elements in flue gas (gas phase)
        self.molflow_C_fluegas = Var(self.flowsheet().config.time,
                        initialize=1,
                        doc='Mole flow of C')

        self.molflow_H_fluegas = Var(self.flowsheet().config.time,
                        initialize=1,
                        doc='Mole flow of H')

        self.molflow_O_fluegas = Var(self.flowsheet().config.time,
                        initialize=1,
                        doc='Mole flow of O')

        self.molflow_N_fluegas = Var(self.flowsheet().config.time,
                        initialize=1,
                        doc='Mole flow of N')

        self.molflow_S_fluegas = Var(self.flowsheet().config.time,
                        initialize=1,
                        doc='Mole flow of S')


        @self.Constraint(self.flowsheet().config.time, doc="C mole flow")
        def molflow_C_fluegas_eqn(b, t):
            return b.molflow_C_fluegas[t] == b.flowrate_daf_burned[t]*b.mf_C_daf/b.am_C + \
                   b.side_1_inlet.flow_mol_comp[t,"CO2"] + b.side_2_inlet.flow_mol_comp[t,"CO2"]

        @self.Constraint(self.flowsheet().config.time, doc="H mole flow")
        def molflow_H_fluegas_eqn(b, t):
            return b.molflow_H_fluegas[t] == b.flowrate_daf_burned[t]*b.mf_H_daf/b.am_H + \
                   b.side_1_inlet.flow_mol_comp[t,"H2O"]*2 + b.side_2_inlet.flow_mol_comp[t,"H2O"]*2 + \
                   b.flowrate_coal[t]*b.mf_H2O_coal[t]/(b.am_H*2+b.am_O)*2

        @self.Constraint(self.flowsheet().config.time, doc="O mole flow")
        def molflow_O_fluegas_eqn(b, t):
            return b.molflow_O_fluegas[t] == b.flowrate_daf_burned[t]*b.mf_O_daf/b.am_O + \
                   b.side_1_inlet.flow_mol_comp[t,"O2"]*2 + b.side_1_inlet.flow_mol_comp[t,"CO2"]*2 + \
                   b.side_1_inlet.flow_mol_comp[t,"H2O"] + b.side_1_inlet.flow_mol_comp[t,"SO2"]*2 + \
                   b.side_1_inlet.flow_mol_comp[t,"NO"] + \
                   b.side_2_inlet.flow_mol_comp[t,"O2"]*2 + b.side_2_inlet.flow_mol_comp[t,"CO2"]*2 + \
                   b.side_2_inlet.flow_mol_comp[t,"H2O"] + b.side_2_inlet.flow_mol_comp[t,"SO2"]*2 + \
                   b.side_2_inlet.flow_mol_comp[t,"NO"] + \
                   b.flowrate_coal[t]*b.mf_H2O_coal[t]/(b.am_H*2+b.am_O)

        @self.Constraint(self.flowsheet().config.time, doc="N mole flow")
        def molflow_N_fluegas_eqn(b, t):
            return b.molflow_N_fluegas[t] == b.flowrate_daf_burned[t]*b.mf_N_daf/b.am_N + \
                   b.side_1_inlet.flow_mol_comp[t,"N2"]*2 + b.side_2_inlet.flow_mol_comp[t,"N2"]*2 + \
                   b.side_1_inlet.flow_mol_comp[t,"NO"] + b.side_2_inlet.flow_mol_comp[t,"NO"]

        @self.Constraint(self.flowsheet().config.time, doc="S mole flow")
        def molflow_S_fluegas_eqn(b, t):
            return b.molflow_S_fluegas[t] == b.flowrate_daf_burned[t]*b.mf_S_daf/b.am_S + \
                   b.side_1_inlet.flow_mol_comp[t,"SO2"] + b.side_2_inlet.flow_mol_comp[t,"SO2"]

        # calculate flue gas flow component at side_1_outlet
        # NO at outlet set to surrogate prediction, ignore its effect on N balance due to its low mole fraction
        # assume mass fraction is same as the mole fraction since the molecular weights of NO and flue gas are usually very close
        @self.Constraint(self.flowsheet().config.time, doc="NO at outlet")
        def NO_eqn(b, t):
            return b.side_1_outlet.flow_mol_comp[t,"NO"] == b.out_mf_NO_FG[t]*b.side_1.properties_out[t].flow_mol

        # N2 at outlet
        @self.Constraint(self.flowsheet().config.time, doc="N2 at outlet")
        def N2_eqn(b, t):
            return b.side_1_outlet.flow_mol_comp[t,"N2"] == (b.molflow_N_fluegas[t] - b.side_1_outlet.flow_mol_comp[t,"NO"])/2

        # SO2 at outlet
        @self.Constraint(self.flowsheet().config.time, doc="SO2 at outlet")
        def SO2_eqn(b, t):
            return b.side_1_outlet.flow_mol_comp[t,"SO2"] == b.molflow_S_fluegas[t]

        # H2O at outlet
        @self.Constraint(self.flowsheet().config.time, doc="H2O at outlet")
        def H2O_eqn(b, t):
            return b.side_1_outlet.flow_mol_comp[t,"H2O"] == b.molflow_H_fluegas[t]/2

        # CO2 at outlet
        @self.Constraint(self.flowsheet().config.time, doc="CO2 at outlet")
        def CO2_eqn(b, t):
            return b.side_1_outlet.flow_mol_comp[t,"CO2"] == b.molflow_C_fluegas[t]

        # O2 at outlet
        @self.Constraint(self.flowsheet().config.time, doc="O2 at outlet")
        def O2_eqn(b, t):
            return b.side_1_outlet.flow_mol_comp[t,"O2"] == (b.molflow_O_fluegas[t] - b.molflow_C_fluegas[t]*2 - \
                   b.molflow_H_fluegas[t]/2 - b.molflow_S_fluegas[t]*2 - b.side_1_outlet.flow_mol_comp[t,"NO"])/2

        # O2 mole fraction on dry basis, needed to compare with measurement value
        self.fluegas_o2_pct_dry = Var(self.flowsheet().config.time,
                        initialize=3,
                        doc='mol percent of O2 on dry basis')

        # mol percent of O2 on dry basis
        @self.Constraint(self.flowsheet().config.time, doc="mol percent of O2 on dry basis")
        def fluegas_o2_pct_dry_eqn(b, t):
            return b.fluegas_o2_pct_dry[t]*(b.side_1_outlet.flow_mol_comp[t,"O2"]+\
                   b.side_1_outlet.flow_mol_comp[t,"N2"]+b.side_1_outlet.flow_mol_comp[t,"CO2"]+\
                   b.side_1_outlet.flow_mol_comp[t,"SO2"]+b.side_1_outlet.flow_mol_comp[t,"NO"])/100 \
                   == b.side_1_outlet.flow_mol_comp[t,"O2"]

        # convert unburned carbon and ash to N2 at side_2_outlet
        # side_2_outlet as N2
        @self.Constraint(self.flowsheet().config.time, doc="N2 at side_2_outlet")
        def N2_side_2_eqn(b, t):
            return b.side_2_outlet.flow_mol_comp[t,"N2"] == (b.flowrate_daf_flyash[t] + b.flowrate_ash[t])/b.am_N/2

        @self.Constraint(self.flowsheet().config.time, doc="O2 at side_2_outlet")
        def O2_side_2_eqn(b, t):
            return sff_minor * b.side_2_outlet.flow_mol_comp[t,"O2"] == sff_minor * b.side_2_outlet.flow_mol_comp[t,"N2"]*0.000001

        @self.Constraint(self.flowsheet().config.time, doc="CO2 at side_2_outlet")
        def CO2_side_2_eqn(b, t):
            return sff_minor * b.side_2_outlet.flow_mol_comp[t,"CO2"] == sff_minor * b.side_2_outlet.flow_mol_comp[t,"N2"]*0.000001

        @self.Constraint(self.flowsheet().config.time, doc="H2O at side_2_outlet")
        def H2O_side_2_eqn(b, t):
            return sff_minor * b.side_2_outlet.flow_mol_comp[t,"H2O"] == sff_minor * b.side_2_outlet.flow_mol_comp[t,"N2"]*0.000001

        @self.Constraint(self.flowsheet().config.time, doc="SO2 at side_2_outlet")
        def SO2_side_2_eqn(b, t):
            return sff_minor * b.side_2_outlet.flow_mol_comp[t,"SO2"] == sff_minor * b.side_2_outlet.flow_mol_comp[t,"N2"]*0.000001

        @self.Constraint(self.flowsheet().config.time, doc="NO at side_2_outlet")
        def NO_side_2_eqn(b, t):
            return sff_minor * b.side_2_outlet.flow_mol_comp[t,"NO"] == sff_minor * b.side_2_outlet.flow_mol_comp[t,"N2"]*0.000001

        #expression to calculate PA mass flow rate including vaporized moisture
        @self.Expression(self.flowsheet().config.time, doc="PA mass flow rate with vaporized coal moisture")
        def flowrate_PA_with_moisture(b,t):
            return sum(b.side_1.properties_in[t].flow_mol_comp[j]*b.side_1.properties_in[t].mw_comp[j] \
                   for j in b.side_1.properties_in[t].config.parameters.component_list)

        #expression to calculate flue gas mass flow rate (gas phase only)
        @self.Expression(self.flowsheet().config.time, doc="flue gas mass flow rate")
        def flowrate_fluegas(b,t):
            return sum(b.side_1.properties_out[t].flow_mol_comp[j]*b.side_1.properties_out[t].mw_comp[j] \
                   for j in b.side_1.properties_out[t].config.parameters.component_list)


    def _make_energy_balance(self):
        # temperature at side_1 inlet, PA temperature
        @self.Constraint(self.flowsheet().config.time, doc="side_1 inlet temperature")
        def side_1_inlet_temperature_eqn(b, t):
            return b.side_1_inlet.temperature[t] == b.temp_coal[t]

        # temperature at side_2 inlet, SA temperature
        @self.Constraint(self.flowsheet().config.time, doc="side_2 inlet temperature")
        def side_2_inlet_temperature_eqn(b, t):
            return b.side_2_inlet.temperature[t] == b.in_2ndAir_Temp[t]

        # overall energy balance to calculate FEGT
        @self.Constraint(self.flowsheet().config.time, doc="temperature at outlet")
        def side_1_temp_eqn(b, t):
            return (b.side_1.properties_in[t].flow_mol*b.side_1.properties_in[t].enth_mol + \
                   b.side_2.properties_in[t].flow_mol*b.side_2.properties_in[t].enth_mol + \
                   b.h_coal[t]*b.flowrate_coal[t]) == (sum(b.out_heat[t,j] for j in b.zones) + \
                   b.side_1.properties_out[t].flow_mol*b.side_1.properties_out[t].enth_mol + \
                   b.flowrate_ash[t]*(593*(b.side_1.properties_out[t].temperature-298.15)+0.293*\
                   (b.side_1.properties_out[t].temperature**2.0-298.15**2.0)) \
                   + b.flowrate_daf_flyash[t]*b.hs_daf_flyash[t])


        # let side_2_outlet temperature same as side_1_outlet temperature
        @self.Constraint(self.flowsheet().config.time, doc="temperature at outlet")
        def side_2_temp_eqn(b, t):
             return  b.side_1.properties_out[t].temperature == b.side_2.properties_out[t].temperature

        # Surrogate model predictions

        # Constraints for heat of zone 1
        @self.Constraint(self.flowsheet().config.time, doc="heat in zone 1")
        def eq_surr_heat_1(b, t):
            X1 = b.in_WWTemp[t, 1]
            X2 = b.in_WWTemp[t, 2]
            X3 = b.in_WWTemp[t, 3]
            X4 = b.in_WWTemp[t, 4]
            X5 = b.in_WWTemp[t, 5]
            X6 = b.in_WWTemp[t, 6]
            X7 = b.in_WWTemp[t, 7]
            X8 = b.in_WWTemp[t, 8]
            X9 = b.in_WWTemp[t, 9]
            X10 = b.in_WWTemp[t, 10]
            X11 = b.in_WWTemp[t, 11]
            X12 = b.in_WWTemp[t, 12]
            X13 = b.in_WWTemp[t, 13]
            X14 = b.in_WWTemp[t, 14]
            X15 = b.in_WWTemp[t, 15]
            X16 = b.in_WWTemp[t, 16]
            X17 = b.flowrate_coal_raw[t]
            X18 = b.mf_H2O_coal_raw[t]
            X19 = b.SR[t]
            X20 = b.SR_lf[t]
            X21 = b.in_2ndAir_Temp[t]
            X22 = b.ratio_PA2coal[t]
            return b.out_heat[t, 1] * b.fheat_ww[t] == (\
            -20468.5 * X1 \
            +7668.88 * X2 \
            +1921.7 * X3 \
            -86.992 * X4 \
            +981.208 * X5 \
            -68.5187 * X8 \
            +555.499 * X9 \
            +2248.18 * X10 \
            +132.475 * X12 \
            +246.492 * X13 \
            -353.649 * X14 \
            -291.361 * X15 \
            +1.24568e+06 * X17 \
            -2.64772e+06 * X18 \
            +9.40752e+06 * X19 \
            -4.0975e+07 * X20 \
            +12419.4 * X21 \
            +3.15232e+06 * X22 \
            +1.22739e+07 * log(X1) \
            -3.60253e+06 * log(X2) \
            +1.61347e+06 * log(X17) \
            +140299 * log(X18) \
            -8.21691e+06 * log(X19) \
            +5.56498e+07 * log(X20) \
            -1.57741e+07 * exp(X18) \
            -13350.1 * X17**2 \
            +104.063 * X17**3 \
            -2.9325 * X1*X17 \
            +2113.57 * X1*X18 \
            -1757.68 * X1*X20 \
            -2.56178 * X1*X21 \
            -22.6412 * X1*X22 \
            +0.0808129 * X2*X4 \
            +0.106031 * X2*X9 \
            -1347.41 * X2*X18 \
            -153.95 * X2*X19 \
            -1.08721 * X3*X9 \
            -0.715745 * X3*X10 \
            +0.65708 * X4*X9 \
            -0.440032 * X5*X14 \
            +0.970767 * X5*X15 \
            -9.41161 * X5*X17 \
            -1.47402 * X5*X21 \
            +0.266019 * X6*X7 \
            +0.100127 * X6*X11 \
            +9.25277 * X8*X17 \
            -10.9006 * X9*X22 \
            -881.502 * X10*X19 \
            -162.924 * X10*X22 \
            -2.30811 * X12*X17 \
            -12.41 * X13*X17 \
            +264.662 * X14*X22 \
            -8.35253 * X15*X17 \
            -755153 * X17*X18 \
            -24434.5 * X17*X19 \
            -177888 * X17*X20 \
            +283.412 * X17*X21 \
            -30830.4 * X17*X22 \
            +4.8456e+06 * X18*X19 \
            +1.10696e+07 * X18*X20 \
            -13063.8 * X18*X21 \
            +334734 * X18*X22 \
            -4.42412e+06 * X19*X20 \
            -2.056e+06 * X20*X22 \
            -1842.84 * X21*X22 \
            -2.17806 * (X9*X18)**2 \
            +17955.8 * (X17*X18)**2 \
            -2566.5 * (X17*X19)**2 \
            -1645.33 * (X17*X20)**2 \
            +24.0689 * (X17*X19)**3)


        # Constraints for heat of zone 2
        @self.Constraint(self.flowsheet().config.time, doc="heat in zone 2")
        def eq_surr_heat_2(b, t):
            X1 = b.in_WWTemp[t, 1]
            X2 = b.in_WWTemp[t, 2]
            X3 = b.in_WWTemp[t, 3]
            X4 = b.in_WWTemp[t, 4]
            X5 = b.in_WWTemp[t, 5]
            X6 = b.in_WWTemp[t, 6]
            X7 = b.in_WWTemp[t, 7]
            X8 = b.in_WWTemp[t, 8]
            X9 = b.in_WWTemp[t, 9]
            X10 = b.in_WWTemp[t, 10]
            X11 = b.in_WWTemp[t, 11]
            X12 = b.in_WWTemp[t, 12]
            X13 = b.in_WWTemp[t, 13]
            X14 = b.in_WWTemp[t, 14]
            X15 = b.in_WWTemp[t, 15]
            X16 = b.in_WWTemp[t, 16]
            X17 = b.flowrate_coal_raw[t]
            X18 = b.mf_H2O_coal_raw[t]
            X19 = b.SR[t]
            X20 = b.SR_lf[t]
            X21 = b.in_2ndAir_Temp[t]
            X22 = b.ratio_PA2coal[t]
            return b.out_heat[t, 2] * b.fheat_ww[t] == (\
            11226 * X1 \
            -19181.2 * X2 \
            +1375.11 * X3 \
            +63.5608 * X4 \
            +2126.95 * X5 \
            -122.416 * X8 \
            +277.16 * X9 \
            +1555.31 * X10 \
            -237.805 * X12 \
            +643.574 * X13 \
            -229.832 * X15 \
            +1.09238e+06 * X17 \
            +1.56393e+07 * X18 \
            -9.81805e+06 * X19 \
            +6.62157e+07 * X20 \
            +8604.71 * X21 \
            +2.8848e+06 * X22 \
            -4.88529e+06 * log(X1) \
            +9.8727e+06 * log(X2) \
            -608919 * log(X4) \
            -923964 * log(X5) \
            -440697 * log(X13) \
            +1.52179e+06 * log(X17) \
            -5919.2 * log(X18) \
            +6.89016e+06 * log(X19) \
            -2.45425e+07 * exp(X18) \
            +1.77338e+06 * exp(X19) \
            -1.9837e+07 * exp(X20) \
            -14309.8 * X17**2 \
            +126.81 * X17**3 \
            -917.773 * X1*X20 \
            -2.06667 * X1*X21 \
            -5.44227 * X1*X22 \
            +0.0596749 * X2*X4 \
            +0.179798 * X2*X9 \
            -6.76515 * X2*X17 \
            -0.729652 * X3*X9 \
            -0.56988 * X3*X10 \
            +0.596081 * X4*X9 \
            +0.912202 * X4*X11 \
            -0.311133 * X5*X14 \
            +0.749249 * X5*X15 \
            -6.59018 * X5*X17 \
            -1.31366 * X5*X21 \
            +0.219365 * X6*X7 \
            -0.172934 * X8*X11 \
            +0.311788 * X8*X14 \
            +6.39066 * X8*X17 \
            -0.631784 * X9*X11 \
            +0.407995 * X9*X12 \
            -712.693 * X10*X19 \
            -11.3242 * X13*X17 \
            +1474.6 * X13*X18 \
            -5.83138 * X15*X17 \
            -537596 * X17*X18 \
            -120688 * X17*X19 \
            -130260 * X17*X20 \
            +213.556 * X17*X21 \
            -24659.7 * X17*X22 \
            +2.70784e+06 * X18*X19 \
            +6.81893e+06 * X18*X20 \
            -10045.4 * X18*X21 \
            -2.92414e+06 * X19*X20 \
            +1012.77 * X19*X21 \
            -1.91845e+06 * X20*X22 \
            -1401.78 * X21*X22 \
            +3.1403 * (X1*X18)**2 \
            -0.0118774 * (X10*X22)**2 \
            +13365.9 * (X17*X18)**2 \
            +342.938 * (X17*X19)**2 \
            -1590.17 * (X17*X20)**2)


        # Constraints for heat of zone 3
        @self.Constraint(self.flowsheet().config.time, doc="heat in zone 3")
        def eq_surr_heat_3(b, t):
            X1 = b.in_WWTemp[t, 1]
            X2 = b.in_WWTemp[t, 2]
            X3 = b.in_WWTemp[t, 3]
            X4 = b.in_WWTemp[t, 4]
            X5 = b.in_WWTemp[t, 5]
            X6 = b.in_WWTemp[t, 6]
            X7 = b.in_WWTemp[t, 7]
            X8 = b.in_WWTemp[t, 8]
            X9 = b.in_WWTemp[t, 9]
            X10 = b.in_WWTemp[t, 10]
            X11 = b.in_WWTemp[t, 11]
            X12 = b.in_WWTemp[t, 12]
            X13 = b.in_WWTemp[t, 13]
            X14 = b.in_WWTemp[t, 14]
            X15 = b.in_WWTemp[t, 15]
            X16 = b.in_WWTemp[t, 16]
            X17 = b.flowrate_coal_raw[t]
            X18 = b.mf_H2O_coal_raw[t]
            X19 = b.SR[t]
            X20 = b.SR_lf[t]
            X21 = b.in_2ndAir_Temp[t]
            X22 = b.ratio_PA2coal[t]
            return b.out_heat[t, 3] * b.fheat_ww[t] == (\
            3924.61 * X1 \
            +439.568 * X2 \
            -9453.81 * X3 \
            +134.941 * X4 \
            +120.143 * X5 \
            -25.8399 * X8 \
            +1066.82 * X10 \
            +41.873 * X12 \
            -28.4594 * X13 \
            +185.728 * X15 \
            +616025 * X17 \
            +7.30432e+06 * X18 \
            +2.68117e+06 * X19 \
            +5.59578e+06 * X20 \
            +5879.89 * X21 \
            +2.2435e+06 * X22 \
            -1.18028e+06 * log(X1) \
            +5.25007e+06 * log(X3) \
            +1.06392e+06 * log(X17) \
            +36200.8 * log(X18) \
            -2.17338e+06 * log(X19) \
            +2.14962e+07 * log(X20) \
            -117387 * log(X21) \
            -1.2551e+07 * exp(X18) \
            -6.91025e+06 * exp(X20) \
            -8248.39 * X17**2 \
            -5.70434e-05 * X14**3 \
            +75.3371 * X17**3 \
            +0.135016 * X1*X14 \
            -1.74455 * X1*X17 \
            +703.013 * X1*X18 \
            -1146.84 * X1*X20 \
            -1.21497 * X1*X21 \
            -13.3324 * X1*X22 \
            -0.149176 * X2*X21 \
            -0.357391 * X3*X10 \
            -0.0244073 * X5*X14 \
            +0.0594707 * X6*X11 \
            +0.0959327 * X7*X9 \
            +3.23037 * X8*X17 \
            -453.628 * X10*X19 \
            -65.7458 * X10*X22 \
            -0.842008 * X12*X17 \
            -5.48772 * X13*X17 \
            +878.866 * X13*X18 \
            -3.45295 * X15*X17 \
            -343651 * X17*X18 \
            -52137.5 * X17*X19 \
            -52566.5 * X17*X20 \
            +134.596 * X17*X21 \
            -15808.1 * X17*X22 \
            +1.16101e+06 * X18*X19 \
            +3.44401e+06 * X18*X20 \
            -6122.41 * X18*X21 \
            -1.1101e+06 * X19*X20 \
            -1.55646e+06 * X20*X22 \
            -921.577 * X21*X22 \
            +0.831261 * (X6*X18)**2 \
            +8413.53 * (X17*X18)**2 \
            +154.629 * (X17*X19)**2 \
            -1314.13 * (X17*X20)**2)


        # Constraints for heat of zone 4
        @self.Constraint(self.flowsheet().config.time, doc="heat in zone 4")
        def eq_surr_heat_4(b, t):
            X1 = b.in_WWTemp[t, 1]
            X2 = b.in_WWTemp[t, 2]
            X3 = b.in_WWTemp[t, 3]
            X4 = b.in_WWTemp[t, 4]
            X5 = b.in_WWTemp[t, 5]
            X6 = b.in_WWTemp[t, 6]
            X7 = b.in_WWTemp[t, 7]
            X8 = b.in_WWTemp[t, 8]
            X9 = b.in_WWTemp[t, 9]
            X10 = b.in_WWTemp[t, 10]
            X11 = b.in_WWTemp[t, 11]
            X12 = b.in_WWTemp[t, 12]
            X13 = b.in_WWTemp[t, 13]
            X14 = b.in_WWTemp[t, 14]
            X15 = b.in_WWTemp[t, 15]
            X16 = b.in_WWTemp[t, 16]
            X17 = b.flowrate_coal_raw[t]
            X18 = b.mf_H2O_coal_raw[t]
            X19 = b.SR[t]
            X20 = b.SR_lf[t]
            X21 = b.in_2ndAir_Temp[t]
            X22 = b.ratio_PA2coal[t]
            return b.out_heat[t, 4] * b.fheat_ww[t] == (\
            3141.14 * X1 \
            +172.215 * X2 \
            +847.946 * X3 \
            +30263.6 * X4 \
            +459.024 * X5 \
            -255.837 * X6 \
            -253.656 * X7 \
            -111.033 * X8 \
            -139.598 * X9 \
            +987.465 * X10 \
            -229.451 * X12 \
            +139.255 * X13 \
            -342.179 * X15 \
            +808422 * X17 \
            +4.59898e+06 * X18 \
            +2.52239e+06 * X19 \
            -1.81024e+08 * X20 \
            +5802 * X21 \
            +1.79667e+06 * X22 \
            -1.01439e+06 * log(X1) \
            -9.25322e+06 * log(X4) \
            -106900 * log(X13) \
            -376731 * log(X15) \
            +1.08793e+06 * log(X17) \
            +43814.2 * log(X18) \
            -2.68644e+06 * log(X19) \
            +1.10291e+08 * log(X20) \
            -1.29535e+07 * exp(X18) \
            +1.51376e+08 * exp(X20) \
            -13.9677 * X4**2 \
            -10570.2 * X17**2 \
            -1.67364e+08 * X20**2 \
            +93.7301 * X17**3 \
            -1.45162 * X1*X17 \
            +958.42 * X1*X18 \
            -501.786 * X1*X20 \
            -1.46924 * X1*X21 \
            -12.8843 * X1*X22 \
            +0.158379 * X2*X4 \
            +0.186963 * X2*X9 \
            -825.009 * X2*X18 \
            -0.534193 * X3*X9 \
            -0.459179 * X3*X10 \
            +376.092 * X3*X18 \
            +0.326125 * X4*X9 \
            -8.35979 * X4*X17 \
            +0.103319 * X4*X21 \
            +0.522501 * X5*X15 \
            -4.9824 * X5*X17 \
            -1.00721 * X5*X21 \
            +0.477989 * X6*X15 \
            +0.455732 * X7*X15 \
            +3.84995 * X8*X17 \
            +37.2828 * X8*X22 \
            +0.35644 * X9*X12 \
            -452.508 * X10*X19 \
            -6.23326 * X13*X17 \
            +930.344 * X13*X18 \
            +30.5111 * X14*X18 \
            -4.00393 * X15*X17 \
            -371726 * X17*X18 \
            -71393.4 * X17*X19 \
            -133621 * X17*X20 \
            +151.271 * X17*X21 \
            -18420.1 * X17*X22 \
            +1.36928e+06 * X18*X19 \
            +6.15109e+06 * X18*X20 \
            -6679.38 * X18*X21 \
            -1.20513e+06 * X19*X20 \
            +909.363 * X19*X21 \
            -1.17463e+06 * X20*X22 \
            -1017.96 * X21*X22 \
            +8948.92 * (X17*X18)**2 \
            +216.961 * (X17*X19)**2 \
            -909.761 * (X17*X20)**2)


        # Constraints for heat of zone 5
        @self.Constraint(self.flowsheet().config.time, doc="heat in zone 5")
        def eq_surr_heat_5(b, t):
            X1 = b.in_WWTemp[t, 1]
            X2 = b.in_WWTemp[t, 2]
            X3 = b.in_WWTemp[t, 3]
            X4 = b.in_WWTemp[t, 4]
            X5 = b.in_WWTemp[t, 5]
            X6 = b.in_WWTemp[t, 6]
            X7 = b.in_WWTemp[t, 7]
            X8 = b.in_WWTemp[t, 8]
            X9 = b.in_WWTemp[t, 9]
            X10 = b.in_WWTemp[t, 10]
            X11 = b.in_WWTemp[t, 11]
            X12 = b.in_WWTemp[t, 12]
            X13 = b.in_WWTemp[t, 13]
            X14 = b.in_WWTemp[t, 14]
            X15 = b.in_WWTemp[t, 15]
            X16 = b.in_WWTemp[t, 16]
            X17 = b.flowrate_coal_raw[t]
            X18 = b.mf_H2O_coal_raw[t]
            X19 = b.SR[t]
            X20 = b.SR_lf[t]
            X21 = b.in_2ndAir_Temp[t]
            X22 = b.ratio_PA2coal[t]
            return b.out_heat[t, 5] * b.fheat_ww[t] == (\
            1260.21 * X1 \
            +187.803 * X2 \
            +90.5357 * X3 \
            +115.405 * X4 \
            -25719.1 * X5 \
            -3.01496 * X8 \
            +227.995 * X9 \
            +1055.92 * X10 \
            +21.3607 * X12 \
            +41.0148 * X13 \
            +120.366 * X15 \
            +893437 * X17 \
            +9.6191e+06 * X18 \
            -172486 * X19 \
            +3.5745e+07 * X20 \
            +4195.57 * X21 \
            +1.65447e+06 * X22 \
            -705858 * log(X1) \
            -317431 * log(X4) \
            +3.84944e+06 * log(X5) \
            +1.23538e+06 * log(X17) \
            -4336.85 * log(X18) \
            -1.18493e+06 * log(X19) \
            -1.77068e+07 * exp(X18) \
            +366166 * exp(X19) \
            -1.12866e+07 * exp(X20) \
            +25.482 * X5**2 \
            +0.548182 * X16**2 \
            -10961.7 * X17**2 \
            -0.0120321 * X5**3 \
            -0.000138663 * X14**3 \
            +86.5756 * X17**3 \
            -1.51235 * X1*X17 \
            -21.3116 * X1*X20 \
            +12.2766 * X1*X22 \
            +0.0610504 * X2*X4 \
            -0.441485 * X3*X9 \
            -0.257743 * X3*X16 \
            +564.113 * X3*X20 \
            +0.480054 * X4*X9 \
            +0.148867 * X4*X21 \
            -0.203093 * X5*X14 \
            +0.556015 * X5*X15 \
            -13.7815 * X5*X17 \
            +0.11546 * X6*X7 \
            +0.0549771 * X6*X11 \
            +3.49661 * X8*X17 \
            -56.8992 * X9*X22 \
            -0.387365 * X10*X16 \
            -410.497 * X10*X19 \
            -64.3265 * X10*X22 \
            -8.94896 * X13*X17 \
            +969.651 * X13*X18 \
            +139.144 * X14*X22 \
            -0.438688 * X15*X16 \
            -3.76038 * X15*X17 \
            -380005 * X17*X18 \
            -94555.8 * X17*X19 \
            -213867 * X17*X20 \
            +154.784 * X17*X21 \
            -18526 * X17*X22 \
            +1.86237e+06 * X18*X19 \
            +6.75155e+06 * X18*X20 \
            -6754.36 * X18*X21 \
            -1.09309e+06 * X19*X20 \
            +823.608 * X19*X21 \
            -1.03983e+06 * X20*X22 \
            -1040.61 * X21*X22 \
            +9386.36 * (X17*X18)**2 \
            +317.752 * (X17*X19)**2 \
            +0.00653287 * (X1*X18)**3)


        # Constraints for heat of zone 6
        @self.Constraint(self.flowsheet().config.time, doc="heat in zone 6")
        def eq_surr_heat_6(b, t):
            X1 = b.in_WWTemp[t, 1]
            X2 = b.in_WWTemp[t, 2]
            X3 = b.in_WWTemp[t, 3]
            X4 = b.in_WWTemp[t, 4]
            X5 = b.in_WWTemp[t, 5]
            X6 = b.in_WWTemp[t, 6]
            X7 = b.in_WWTemp[t, 7]
            X8 = b.in_WWTemp[t, 8]
            X9 = b.in_WWTemp[t, 9]
            X10 = b.in_WWTemp[t, 10]
            X11 = b.in_WWTemp[t, 11]
            X12 = b.in_WWTemp[t, 12]
            X13 = b.in_WWTemp[t, 13]
            X14 = b.in_WWTemp[t, 14]
            X15 = b.in_WWTemp[t, 15]
            X16 = b.in_WWTemp[t, 16]
            X17 = b.flowrate_coal_raw[t]
            X18 = b.mf_H2O_coal_raw[t]
            X19 = b.SR[t]
            X20 = b.SR_lf[t]
            X21 = b.in_2ndAir_Temp[t]
            X22 = b.ratio_PA2coal[t]
            return b.out_heat[t, 6] * b.fheat_ww[t] == (\
            2524.13 * X1 \
            -473.39 * X2 \
            -611.819 * X3 \
            -327.538 * X4 \
            +23715.9 * X5 \
            -69507.5 * X6 \
            -277.542 * X7 \
            +75.8839 * X8 \
            -382.351 * X9 \
            +1063.19 * X10 \
            +1804.61 * X11 \
            -361.396 * X12 \
            +168.069 * X13 \
            -16.2907 * X15 \
            +940653 * X17 \
            +4.23171e+06 * X18 \
            +2.28657e+06 * X19 \
            +3.45724e+07 * X20 \
            +5518.2 * X21 \
            +1.71137e+06 * X22 \
            -893878 * log(X1) \
            -149762 * log(X4) \
            -8.86277e+06 * log(X5) \
            +1.25918e+07 * log(X6) \
            +1.11622e+06 * log(X17) \
            +49611.9 * log(X18) \
            -4.8279e+06 * log(X19) \
            -1.32067e+07 * exp(X18) \
            -1.13998e+07 * exp(X20) \
            -7.92713 * X5**2 \
            +61.0611 * X6**2 \
            -0.1904 * X14**2 \
            -11705.1 * X17**2 \
            -0.0247076 * X6**3 \
            +92.3209 * X17**3 \
            -0.310677 * X1*X13 \
            +0.172257 * X1*X14 \
            -1.44116 * X1*X21 \
            -19.4625 * X1*X22 \
            +0.684303 * X2*X3 \
            +0.238784 * X2*X4 \
            -0.391165 * X3*X10 \
            +541.242 * X3*X20 \
            +0.437355 * X4*X9 \
            +0.28436 * X4*X21 \
            +0.350263 * X5*X9 \
            -0.632562 * X5*X11 \
            +0.536186 * X5*X12 \
            -0.318697 * X5*X14 \
            +0.480491 * X5*X15 \
            -5.50346 * X5*X17 \
            +0.460134 * X6*X15 \
            -11.674 * X6*X17 \
            +298.115 * X7*X19 \
            -17.6983 * X9*X22 \
            -542.341 * X10*X19 \
            -0.637401 * X11*X15 \
            -1.31338 * X11*X21 \
            -8.51399 * X13*X17 \
            +1398.38 * X13*X18 \
            -60.938 * X14*X18 \
            +150.91 * X14*X22 \
            -4.01606 * X15*X17 \
            -378070 * X17*X18 \
            -128539 * X17*X19 \
            -188207 * X17*X20 \
            +160.627 * X17*X21 \
            -18193.8 * X17*X22 \
            +2.45023e+06 * X18*X19 \
            +5.92299e+06 * X18*X20 \
            -7319.98 * X18*X21 \
            +1379.5 * X19*X21 \
            -1.15268e+06 * X20*X22 \
            -1055.24 * X21*X22 \
            +8780.41 * (X17*X18)**2 \
            +420.738 * (X17*X19)**2)


        # Constraints for heat of zone 7
        @self.Constraint(self.flowsheet().config.time, doc="heat in zone 7")
        def eq_surr_heat_7(b, t):
            X1 = b.in_WWTemp[t, 1]
            X2 = b.in_WWTemp[t, 2]
            X3 = b.in_WWTemp[t, 3]
            X4 = b.in_WWTemp[t, 4]
            X5 = b.in_WWTemp[t, 5]
            X6 = b.in_WWTemp[t, 6]
            X7 = b.in_WWTemp[t, 7]
            X8 = b.in_WWTemp[t, 8]
            X9 = b.in_WWTemp[t, 9]
            X10 = b.in_WWTemp[t, 10]
            X11 = b.in_WWTemp[t, 11]
            X12 = b.in_WWTemp[t, 12]
            X13 = b.in_WWTemp[t, 13]
            X14 = b.in_WWTemp[t, 14]
            X15 = b.in_WWTemp[t, 15]
            X16 = b.in_WWTemp[t, 16]
            X17 = b.flowrate_coal_raw[t]
            X18 = b.mf_H2O_coal_raw[t]
            X19 = b.SR[t]
            X20 = b.SR_lf[t]
            X21 = b.in_2ndAir_Temp[t]
            X22 = b.ratio_PA2coal[t]
            return b.out_heat[t, 7] * b.fheat_ww[t] == (\
            474.27 * X1 \
            +45.9059 * X2 \
            +320.83 * X3 \
            -280.775 * X4 \
            +1072.55 * X5 \
            +142.021 * X6 \
            -9016.5 * X7 \
            +20.6564 * X8 \
            -137.75 * X9 \
            +1141.63 * X10 \
            +976.004 * X11 \
            +26.7798 * X12 \
            -86.7844 * X13 \
            -632.481 * X14 \
            +386.622 * X15 \
            +798165 * X17 \
            -8.86116e+06 * X18 \
            +2.67217e+06 * X19 \
            -2.23579e+07 * X20 \
            +4616.63 * X21 \
            +1.26689e+06 * X22 \
            -457474 * log(X1) \
            -456908 * log(X5) \
            +4.97543e+06 * log(X7) \
            -324054 * log(X15) \
            +859905 * log(X17) \
            +51982.4 * log(X18) \
            -6.09034e+06 * log(X19) \
            +2.51526e+07 * log(X20) \
            -2.00793e+06 * exp(X18) \
            -8569.95 * X17**2 \
            +58.8983 * X17**3 \
            +693.438 * X1*X18 \
            +238.611 * X1*X20 \
            -5.93448 * X1*X22 \
            +0.301957 * X2*X9 \
            -709.971 * X2*X18 \
            -0.527198 * X3*X9 \
            -0.286885 * X3*X10 \
            +541.712 * X3*X18 \
            -9.71413 * X3*X19 \
            +301.185 * X3*X20 \
            +0.27972 * X4*X9 \
            -2.08875 * X4*X19 \
            +0.29136 * X4*X21 \
            +0.214989 * X5*X9 \
            -0.259724 * X5*X14 \
            +0.490069 * X5*X15 \
            -4.44612 * X5*X17 \
            -0.947716 * X5*X21 \
            -13.9317 * X7*X17 \
            +3.06822 * X8*X17 \
            +0.168198 * X9*X15 \
            -1.26534 * X9*X17 \
            -458.871 * X10*X19 \
            -84.5202 * X10*X22 \
            -0.313148 * X11*X15 \
            -1.13435 * X11*X21 \
            +0.353432 * X13*X14 \
            -7.75573 * X13*X17 \
            +4286.46 * X14*X18 \
            +116.804 * X14*X22 \
            -4.25261 * X15*X17 \
            -331508 * X17*X18 \
            -89127.3 * X17*X19 \
            -149368 * X17*X20 \
            +150.79 * X17*X21 \
            -15081.2 * X17*X22 \
            +2.98611e+06 * X18*X19 \
            +4.35995e+06 * X18*X20 \
            -6541.34 * X18*X21 \
            +1362.51 * X19*X21 \
            +93589.9 * X19*X22 \
            -821249 * X20*X22 \
            -970.474 * X21*X22 \
            -9.97514 * (X14*X18)**2 \
            +7495.68 * (X17*X18)**2 \
            -853.77 * (X17*X19)**2 \
            +11.0189 * (X17*X19)**3)


        # Constraints for heat of zone 8
        @self.Constraint(self.flowsheet().config.time, doc="heat in zone 8")
        def eq_surr_heat_8(b, t):
            X1 = b.in_WWTemp[t, 1]
            X2 = b.in_WWTemp[t, 2]
            X3 = b.in_WWTemp[t, 3]
            X4 = b.in_WWTemp[t, 4]
            X5 = b.in_WWTemp[t, 5]
            X6 = b.in_WWTemp[t, 6]
            X7 = b.in_WWTemp[t, 7]
            X8 = b.in_WWTemp[t, 8]
            X9 = b.in_WWTemp[t, 9]
            X10 = b.in_WWTemp[t, 10]
            X11 = b.in_WWTemp[t, 11]
            X12 = b.in_WWTemp[t, 12]
            X13 = b.in_WWTemp[t, 13]
            X14 = b.in_WWTemp[t, 14]
            X15 = b.in_WWTemp[t, 15]
            X16 = b.in_WWTemp[t, 16]
            X17 = b.flowrate_coal_raw[t]
            X18 = b.mf_H2O_coal_raw[t]
            X19 = b.SR[t]
            X20 = b.SR_lf[t]
            X21 = b.in_2ndAir_Temp[t]
            X22 = b.ratio_PA2coal[t]
            return b.out_heat[t, 8] * b.fheat_ww[t] == (\
            1203.05 * X1 \
            -344.597 * X2 \
            +19.5032 * X3 \
            -510.784 * X4 \
            +43.0538 * X5 \
            -16414.7 * X6 \
            -166.079 * X7 \
            +5262.84 * X8 \
            +951.118 * X9 \
            +1944.21 * X10 \
            -524.099 * X11 \
            +286.689 * X12 \
            +376.151 * X13 \
            -311.631 * X14 \
            +94.5494 * X15 \
            +403.623 * X16 \
            +729460 * X17 \
            -8.10765e+06 * X18 \
            +5.12163e+06 * X19 \
            +340605 * X20 \
            +10439.8 * X21 \
            +377793 * X22 \
            -597041 * log(X1) \
            -455726 * log(X9) \
            +715043 * log(X17) \
            -1.31066e+07 * log(X19) \
            +23.7108 * X6**2 \
            -5.33194 * X8**2 \
            +0.642661 * X11**2 \
            +0.66995 * X16**2 \
            -7235.52 * X17**2 \
            -9.50943e+06 * X18**2 \
            -0.0109567 * X6**3 \
            -7.59659e-05 * X14**3 \
            +43.2711 * X17**3 \
            +5.74405e+06 * X18**3 \
            -0.319511 * X1*X13 \
            +0.647747 * X2*X3 \
            +0.0386179 * X2*X9 \
            -0.00978949 * X2*X10 \
            -0.289837 * X3*X11 \
            +3.11788 * X3*X17 \
            -188.549 * X3*X19 \
            +0.180779 * X4*X11 \
            +0.421665 * X4*X15 \
            +0.556927 * X4*X16 \
            -179.362 * X4*X19 \
            +0.489665 * X5*X12 \
            +0.525549 * X5*X15 \
            -1.11894 * X5*X21 \
            -0.651636 * X6*X12 \
            +0.26951 * X7*X14 \
            +91.4022 * X7*X19 \
            -0.337552 * X8*X11 \
            -13.1205 * X8*X17 \
            +625.447 * X8*X18 \
            +111.871 * X8*X19 \
            -0.143315 * X9*X16 \
            -1.43078 * X9*X17 \
            -0.437508 * X10*X11 \
            -392.538 * X10*X19 \
            -1.03531 * X10*X21 \
            -132.023 * X10*X22 \
            -0.366015 * X11*X15 \
            +454.417 * X11*X19 \
            -0.0654458 * X12*X13 \
            -80.8874 * X12*X19 \
            -9.01814 * X13*X17 \
            +0.0887393 * X14*X21 \
            +69.5916 * X14*X22 \
            -0.507975 * X15*X16 \
            -12.9206 * X15*X17 \
            -166.031 * X15*X19 \
            +343.605 * X15*X20 \
            -351.38 * X16*X19 \
            -939.183 * X16*X20 \
            +36.4681 * X16*X22 \
            -276870 * X17*X18 \
            -323281 * X17*X19 \
            +146293 * X17*X20 \
            +186.683 * X17*X21 \
            -6502.59 * X17*X22 \
            +7.74397e+06 * X18*X19 \
            -7126.9 * X18*X21 \
            +1.75851e+06 * X19*X20 \
            -953.168 * X19*X21 \
            +207351 * X19*X22 \
            -4595.65 * X20*X21 \
            -1113.66 * X21*X22 \
            +2.11788 * (X13*X18)**2 \
            +0.00013867 * (X15*X17)**2 \
            +5802.79 * (X17*X18)**2 \
            -282.498 * (X17*X19)**2 \
            -1510.03 * (X17*X20)**2 \
            +1.34 * (X19*X21)**2 \
            +14.3624 * (X17*X19)**3)


        # Constraints for heat of zone 9
        @self.Constraint(self.flowsheet().config.time, doc="heat in zone 9")
        def eq_surr_heat_9(b, t):
            X1 = b.in_WWTemp[t, 1]
            X2 = b.in_WWTemp[t, 2]
            X3 = b.in_WWTemp[t, 3]
            X4 = b.in_WWTemp[t, 4]
            X5 = b.in_WWTemp[t, 5]
            X6 = b.in_WWTemp[t, 6]
            X7 = b.in_WWTemp[t, 7]
            X8 = b.in_WWTemp[t, 8]
            X9 = b.in_WWTemp[t, 9]
            X10 = b.in_WWTemp[t, 10]
            X11 = b.in_WWTemp[t, 11]
            X12 = b.in_WWTemp[t, 12]
            X13 = b.in_WWTemp[t, 13]
            X14 = b.in_WWTemp[t, 14]
            X15 = b.in_WWTemp[t, 15]
            X16 = b.in_WWTemp[t, 16]
            X17 = b.flowrate_coal_raw[t]
            X18 = b.mf_H2O_coal_raw[t]
            X19 = b.SR[t]
            X20 = b.SR_lf[t]
            X21 = b.in_2ndAir_Temp[t]
            X22 = b.ratio_PA2coal[t]
            return b.out_heat[t, 9] * b.fheat_ww[t] == (\
            1836.1 * X1 \
            +462.437 * X2 \
            +700.405 * X3 \
            -1498.3 * X4 \
            +1209.82 * X5 \
            -38993.2 * X6 \
            +19929.9 * X7 \
            +542.635 * X8 \
            -19440.9 * X9 \
            +5496.56 * X10 \
            +1289.79 * X11 \
            -851.372 * X12 \
            -799.691 * X13 \
            +383.718 * X14 \
            +1749 * X15 \
            +3770.54 * X16 \
            +1.70303e+06 * X17 \
            +934596 * X18 \
            +1.37643e+07 * X19 \
            -1.30096e+07 * X20 \
            +16055.5 * X21 \
            +201380 * X22 \
            -1.07631e+06 * log(X1) \
            -450288 * log(X4) \
            +1.08255e+07 * log(X9) \
            -1.17544e+06 * log(X10) \
            -1.53803e+06 * log(X11) \
            -177549 * log(X12) \
            +39948.2 * log(X15) \
            -2.20703e+06 * log(X16) \
            +992927 * log(X17) \
            -1.86841e+07 * log(X19) \
            -2.31856e+07 * exp(X18) \
            +54.2939 * X6**2 \
            -8.06834 * X7**2 \
            -12035.4 * X17**2 \
            +6.59153e+06 * X20**2 \
            -0.024925 * X6**3 \
            -0.000354734 * X14**3 \
            +41.1893 * X17**3 \
            -0.103124 * X1*X13 \
            -0.211392 * X2*X9 \
            +1.2001 * X3*X4 \
            -1.26509 * X3*X11 \
            +0.00199659 * X3*X16 \
            -390.09 * X3*X19 \
            +0.0340134 * X4*X8 \
            -0.0875401 * X4*X11 \
            +0.921992 * X4*X15 \
            +1.14569 * X4*X16 \
            -201.086 * X4*X19 \
            +0.358484 * X4*X21 \
            +1.2154 * X5*X12 \
            -3.04479 * X5*X21 \
            +494.397 * X6*X18 \
            +0.851817 * X7*X12 \
            +0.936564 * X7*X14 \
            +566.986 * X7*X18 \
            -14307 * X7*X19 \
            -0.0178573 * X8*X15 \
            -258.651 * X8*X20 \
            -0.14334 * X9*X15 \
            -0.425254 * X9*X16 \
            -43.2167 * X9*X17 \
            -1.90981 * X10*X14 \
            -1062.96 * X10*X19 \
            -235.592 * X10*X22 \
            +1.84247 * X11*X13 \
            -1.02394 * X11*X15 \
            +1212.52 * X11*X19 \
            -217.572 * X12*X19 \
            -20.4304 * X13*X17 \
            +1.06493 * X14*X21 \
            -1.12386 * X15*X16 \
            -45.3926 * X15*X17 \
            +74.1654 * X16*X22 \
            -655976 * X17*X18 \
            -639368 * X17*X19 \
            +256375 * X17*X20 \
            +444.669 * X17*X21 \
            -13228.3 * X17*X22 \
            +1.97974e+07 * X18*X19 \
            -15073.3 * X18*X21 \
            -3797.82 * X19*X21 \
            +358253 * X19*X22 \
            -6507.56 * X20*X21 \
            +621411 * X20*X22 \
            -2266.27 * X21*X22 \
            +3.79183 * (X7*X19)**2 \
            -0.214096 * (X9*X19)**2 \
            +0.0246641 * (X14*X22)**2 \
            +0.000570728 * (X15*X17)**2 \
            -0.240504 * (X16*X19)**2 \
            +13497.1 * (X17*X18)**2 \
            -3995.29 * (X17*X19)**2 \
            -2167.38 * (X17*X20)**2 \
            +3.74466 * (X19*X21)**2 \
            +54.9147 * (X17*X19)**3)


        # Constraints for heat of zone 10
        @self.Constraint(self.flowsheet().config.time, doc="heat in zone 10")
        def eq_surr_heat_10(b, t):
            X1 = b.in_WWTemp[t, 1]
            X2 = b.in_WWTemp[t, 2]
            X3 = b.in_WWTemp[t, 3]
            X4 = b.in_WWTemp[t, 4]
            X5 = b.in_WWTemp[t, 5]
            X6 = b.in_WWTemp[t, 6]
            X7 = b.in_WWTemp[t, 7]
            X8 = b.in_WWTemp[t, 8]
            X9 = b.in_WWTemp[t, 9]
            X10 = b.in_WWTemp[t, 10]
            X11 = b.in_WWTemp[t, 11]
            X12 = b.in_WWTemp[t, 12]
            X13 = b.in_WWTemp[t, 13]
            X14 = b.in_WWTemp[t, 14]
            X15 = b.in_WWTemp[t, 15]
            X16 = b.in_WWTemp[t, 16]
            X17 = b.flowrate_coal_raw[t]
            X18 = b.mf_H2O_coal_raw[t]
            X19 = b.SR[t]
            X20 = b.SR_lf[t]
            X21 = b.in_2ndAir_Temp[t]
            X22 = b.ratio_PA2coal[t]
            return b.out_heat[t, 10] * b.fheat_ww[t] == (\
            1625.59 * X1 \
            -48971 * X2 \
            +1099.29 * X3 \
            -1530.47 * X4 \
            +1344.08 * X5 \
            +49046.3 * X6 \
            -3856.09 * X7 \
            -433.777 * X8 \
            +3144.18 * X9 \
            +23674 * X10 \
            +2410.01 * X11 \
            -840.61 * X12 \
            +45.6992 * X13 \
            +232.693 * X14 \
            +2993.11 * X15 \
            +3534.78 * X16 \
            +1.65729e+06 * X17 \
            -3.16276e+07 * X18 \
            +4.18764e+06 * X19 \
            -2.68781e+07 * X20 \
            -459.342 * X21 \
            -300318 * X22 \
            -1.06227e+06 * log(X1) \
            +2.24874e+07 * log(X2) \
            +402203 * log(X4) \
            -2.3312e+07 * log(X6) \
            +1.08194e+07 * log(X7) \
            -1.39558e+06 * log(X9) \
            -7.72647e+06 * log(X10) \
            -247038 * log(X12) \
            +1.0317e+06 * log(X14) \
            -362815 * log(X15) \
            -1.6513e+06 * log(X16) \
            -62425.8 * log(X17) \
            +360156 * log(X18) \
            -1.04043e+07 * log(X19) \
            +628079 * exp(X19) \
            +8.19933e+06 * exp(X20) \
            -8564.38 * X17**2 \
            +0.0113407 * X2**3 \
            -0.0103721 * X6**3 \
            -0.0091608 * X10**3 \
            -0.955465 * X17**3 \
            -0.269058 * X1*X5 \
            +0.942715 * X1*X8 \
            -0.599579 * X1*X13 \
            +0.479366 * X2*X9 \
            -0.158162 * X2*X15 \
            -1.09217 * X3*X11 \
            +4.91327 * X3*X17 \
            -224.355 * X3*X19 \
            +1.22198 * X4*X16 \
            +184.763 * X4*X19 \
            +1.23891 * X5*X12 \
            -3.08247 * X5*X21 \
            +1.09686 * X7*X12 \
            +1232.56 * X7*X18 \
            -18397.5 * X7*X19 \
            +1489.57 * X8*X18 \
            -187.361 * X8*X20 \
            +0.0815017 * X9*X14 \
            -0.230493 * X9*X15 \
            -0.67902 * X9*X16 \
            -9.33015 * X9*X17 \
            -1.12153 * X10*X11 \
            -0.176876 * X10*X12 \
            -1.88087 * X10*X14 \
            -37.1044 * X10*X17 \
            -1188.3 * X10*X19 \
            -257.4 * X10*X22 \
            -2.17159 * X11*X14 \
            -0.784482 * X11*X15 \
            +1274.39 * X11*X19 \
            +0.136902 * X12*X13 \
            -188.324 * X12*X19 \
            +1.0627 * X13*X14 \
            -19.5148 * X13*X17 \
            +0.653504 * X14*X21 \
            -37.4635 * X15*X17 \
            -560.229 * X15*X19 \
            -10.567 * X16*X17 \
            -1082.38 * X16*X19 \
            -601636 * X17*X18 \
            -470098 * X17*X19 \
            +124754 * X17*X20 \
            +430.837 * X17*X21 \
            -9270.15 * X17*X22 \
            +2.08162e+07 * X18*X19 \
            -12796 * X18*X21 \
            +8170.38 * X19*X21 \
            +225163 * X19*X22 \
            +1.12393e+06 * X20*X22 \
            -1763.29 * X21*X22 \
            +4.89596 * (X7*X19)**2 \
            +0.000433641 * (X15*X17)**2 \
            +9195.08 * (X17*X18)**2 \
            -7066.47 * (X17*X19)**2 \
            +72.3083 * (X17*X19)**3)


        # Constraints for heat of zone 11
        @self.Constraint(self.flowsheet().config.time, doc="heat in zone 11")
        def eq_surr_heat_11(b, t):
            X1 = b.in_WWTemp[t, 1]
            X2 = b.in_WWTemp[t, 2]
            X3 = b.in_WWTemp[t, 3]
            X4 = b.in_WWTemp[t, 4]
            X5 = b.in_WWTemp[t, 5]
            X6 = b.in_WWTemp[t, 6]
            X7 = b.in_WWTemp[t, 7]
            X8 = b.in_WWTemp[t, 8]
            X9 = b.in_WWTemp[t, 9]
            X10 = b.in_WWTemp[t, 10]
            X11 = b.in_WWTemp[t, 11]
            X12 = b.in_WWTemp[t, 12]
            X13 = b.in_WWTemp[t, 13]
            X14 = b.in_WWTemp[t, 14]
            X15 = b.in_WWTemp[t, 15]
            X16 = b.in_WWTemp[t, 16]
            X17 = b.flowrate_coal_raw[t]
            X18 = b.mf_H2O_coal_raw[t]
            X19 = b.SR[t]
            X20 = b.SR_lf[t]
            X21 = b.in_2ndAir_Temp[t]
            X22 = b.ratio_PA2coal[t]
            return b.out_heat[t, 11] * b.fheat_ww[t] == (\
            1149.96 * X1 \
            +486.815 * X2 \
            +985.013 * X3 \
            -1168.2 * X4 \
            +1424.95 * X5 \
            +191.148 * X6 \
            +23467.9 * X7 \
            -382.945 * X8 \
            +3211.32 * X9 \
            +4291.96 * X10 \
            -19488.8 * X11 \
            +2120.76 * X12 \
            +39.2439 * X13 \
            -1725.39 * X14 \
            +4715.43 * X15 \
            +7808.52 * X16 \
            +1.2977e+06 * X17 \
            -7.18061e+06 * X18 \
            +6.93565e+06 * X19 \
            -2.37482e+07 * X20 \
            -2853.01 * X21 \
            -268510 * X22 \
            -892091 * log(X1) \
            -1.66828e+06 * log(X9) \
            -1.82491e+06 * log(X10) \
            +1.05208e+07 * log(X11) \
            -967600 * log(X12) \
            +235346 * log(X14) \
            -1.48537e+06 * log(X15) \
            -2.37767e+06 * log(X16) \
            -644766 * log(X17) \
            -9.59737e+06 * log(X19) \
            -1.57952e+07 * exp(X18) \
            +8.10169e+06 * exp(X20) \
            -0.414281 * X2**2 \
            -8.98981 * X7**2 \
            -3654.23 * X17**2 \
            -21.8999 * X17**3 \
            +0.503427 * X1*X8 \
            -0.089291 * X1*X13 \
            +0.414521 * X2*X9 \
            -0.0926811 * X2*X21 \
            -1.27019 * X3*X11 \
            +7.5139 * X3*X17 \
            -112.029 * X3*X19 \
            +1.43679 * X4*X16 \
            +192.576 * X4*X19 \
            -2.14251 * X5*X21 \
            +1535.88 * X7*X18 \
            -16151.4 * X7*X19 \
            +1659.33 * X8*X18 \
            -0.767835 * X9*X16 \
            -0.600427 * X10*X11 \
            -38.1635 * X11*X17 \
            -250.644 * X12*X19 \
            +0.779036 * X13*X14 \
            -19.9073 * X13*X17 \
            +1.45918 * X14*X21 \
            -1.06075 * X15*X16 \
            -8.19138 * X15*X17 \
            -682.175 * X15*X19 \
            -621.094 * X16*X19 \
            -3438.25 * X16*X20 \
            +30.9672 * X16*X22 \
            -587541 * X17*X18 \
            -242606 * X17*X19 \
            +106700 * X17*X20 \
            +385.744 * X17*X21 \
            -6676.14 * X17*X22 \
            +1.82598e+07 * X18*X19 \
            -10529.7 * X18*X21 \
            +7471.83 * X19*X21 \
            +1.02902e+06 * X20*X22 \
            -1482.81 * X21*X22 \
            +4.28189 * (X7*X19)**2 \
            -0.228191 * (X10*X19)**2 \
            +9238.91 * (X17*X18)**2 \
            -8721.36 * (X17*X19)**2 \
            +75.6653 * (X17*X19)**3)


        # Constraints for heat of zone 12
        @self.Constraint(self.flowsheet().config.time, doc="heat in zone 12")
        def eq_surr_heat_12(b, t):
            X1 = b.in_WWTemp[t, 1]
            X2 = b.in_WWTemp[t, 2]
            X3 = b.in_WWTemp[t, 3]
            X4 = b.in_WWTemp[t, 4]
            X5 = b.in_WWTemp[t, 5]
            X6 = b.in_WWTemp[t, 6]
            X7 = b.in_WWTemp[t, 7]
            X8 = b.in_WWTemp[t, 8]
            X9 = b.in_WWTemp[t, 9]
            X10 = b.in_WWTemp[t, 10]
            X11 = b.in_WWTemp[t, 11]
            X12 = b.in_WWTemp[t, 12]
            X13 = b.in_WWTemp[t, 13]
            X14 = b.in_WWTemp[t, 14]
            X15 = b.in_WWTemp[t, 15]
            X16 = b.in_WWTemp[t, 16]
            X17 = b.flowrate_coal_raw[t]
            X18 = b.mf_H2O_coal_raw[t]
            X19 = b.SR[t]
            X20 = b.SR_lf[t]
            X21 = b.in_2ndAir_Temp[t]
            X22 = b.ratio_PA2coal[t]
            return b.out_heat[t, 12] * b.fheat_ww[t] == (\
            3872.84 * X1 \
            -356.661 * X2 \
            -314.183 * X3 \
            -1041.41 * X4 \
            +1293.47 * X5 \
            +127.535 * X6 \
            -587.39 * X7 \
            -30.5484 * X8 \
            -233.72 * X9 \
            +1226.77 * X10 \
            +4906.7 * X11 \
            -21142 * X12 \
            +654.435 * X13 \
            -2689.62 * X14 \
            +6699.87 * X15 \
            +1567.79 * X16 \
            +1.50371e+06 * X17 \
            -1.84478e+07 * X18 \
            -7.69392e+06 * X19 \
            -1.79886e+07 * X20 \
            -4798.9 * X21 \
            -183598 * X22 \
            -1.7949e+06 * log(X1) \
            -1.73896e+06 * log(X11) \
            +1.08083e+07 * log(X12) \
            -1.46925e+06 * log(X13) \
            +1.14705e+06 * log(X14) \
            -3.79629e+06 * log(X15) \
            -995945 * log(X17) \
            +213553 * log(X18) \
            +1.0122e+06 * exp(X19) \
            -15401.3 * X17**2 \
            +7.49425e+06 * X20**2 \
            +124.313 * X17**3 \
            +0.328915 * X1*X5 \
            -0.444571 * X1*X13 \
            -1.75528 * X1*X21 \
            +0.529667 * X2*X9 \
            +0.335859 * X2*X15 \
            -1210.92 * X2*X18 \
            -0.797568 * X3*X11 \
            +8.74363 * X3*X17 \
            +590.456 * X3*X19 \
            +0.538997 * X4*X11 \
            +579.815 * X4*X19 \
            -2.30992 * X5*X21 \
            +0.779268 * X7*X9 \
            +1291.09 * X7*X18 \
            +973.997 * X8*X18 \
            +0.113785 * X9*X15 \
            -0.48171 * X9*X16 \
            -0.928725 * X10*X11 \
            -1.30138 * X11*X14 \
            +0.196111 * X12*X13 \
            -51.1074 * X12*X17 \
            +0.807018 * X13*X14 \
            -13.8925 * X13*X17 \
            +2.44503 * X13*X21 \
            +2.33683 * X14*X21 \
            -5.50345 * X15*X17 \
            -10.5945 * X16*X17 \
            -741.123 * X16*X19 \
            -510935 * X17*X18 \
            -484123 * X17*X19 \
            +82219.5 * X17*X20 \
            +307.455 * X17*X21 \
            -5683.32 * X17*X22 \
            +1.31002e+07 * X18*X19 \
            -8586.79 * X18*X21 \
            +6331.18 * X19*X21 \
            +665512 * X20*X22 \
            -915.588 * X21*X22 \
            +8162.68 * (X17*X18)**2)


        # Constraints for heat of zone 13
        @self.Constraint(self.flowsheet().config.time, doc="heat in zone 13")
        def eq_surr_heat_13(b, t):
            X1 = b.in_WWTemp[t, 1]
            X2 = b.in_WWTemp[t, 2]
            X3 = b.in_WWTemp[t, 3]
            X4 = b.in_WWTemp[t, 4]
            X5 = b.in_WWTemp[t, 5]
            X6 = b.in_WWTemp[t, 6]
            X7 = b.in_WWTemp[t, 7]
            X8 = b.in_WWTemp[t, 8]
            X9 = b.in_WWTemp[t, 9]
            X10 = b.in_WWTemp[t, 10]
            X11 = b.in_WWTemp[t, 11]
            X12 = b.in_WWTemp[t, 12]
            X13 = b.in_WWTemp[t, 13]
            X14 = b.in_WWTemp[t, 14]
            X15 = b.in_WWTemp[t, 15]
            X16 = b.in_WWTemp[t, 16]
            X17 = b.flowrate_coal_raw[t]
            X18 = b.mf_H2O_coal_raw[t]
            X19 = b.SR[t]
            X20 = b.SR_lf[t]
            X21 = b.in_2ndAir_Temp[t]
            X22 = b.ratio_PA2coal[t]
            return b.out_heat[t, 13] * b.fheat_ww[t] == (\
            1109.75 * X1 \
            +24.7662 * X2 \
            +360.653 * X3 \
            -711.742 * X4 \
            +27073.4 * X5 \
            -1532.11 * X7 \
            +430.757 * X9 \
            +1286.7 * X10 \
            +1670.22 * X11 \
            -25340.7 * X12 \
            +27233.9 * X13 \
            +70.5537 * X14 \
            -28386.6 * X15 \
            +1829.98 * X16 \
            +440346 * X17 \
            -4.85573e+06 * X18 \
            -958725 * X19 \
            -2.32709e+06 * X20 \
            -4208.62 * X21 \
            -264105 * X22 \
            -330702 * log(X1) \
            +33237.7 * log(X4) \
            -1.00714e+07 * log(X5) \
            +569845 * log(X7) \
            +8.28968e+06 * log(X12) \
            -5.87903e+06 * log(X13) \
            +253169 * log(X14) \
            +8.38567e+06 * log(X15) \
            -737308 * log(X16) \
            -1.06106e+06 * log(X17) \
            -2.49954e+06 * log(X19) \
            -9.68149 * X5**2 \
            +9.41298 * X12**2 \
            -17.7275 * X13**2 \
            +14.4525 * X15**2 \
            +981.461 * X17**2 \
            -1.18891e+07 * X18**2 \
            +4.30556e-05 * X6**3 \
            -26.2018 * X17**3 \
            +1.68091e+07 * X18**3 \
            +0.0939684 * X1*X8 \
            -0.20426 * X1*X13 \
            -0.0883019 * X1*X17 \
            -309.54 * X1*X20 \
            -0.879469 * X1*X21 \
            +129.155 * X1*X22 \
            +0.157583 * X2*X9 \
            -518.811 * X2*X18 \
            +0.530736 * X3*X4 \
            -0.603945 * X3*X9 \
            -0.753218 * X3*X11 \
            +199.589 * X3*X19 \
            +0.176046 * X4*X11 \
            +152.519 * X4*X19 \
            +1000.99 * X5*X20 \
            +0.36363 * X7*X9 \
            +0.788635 * X7*X12 \
            -0.0930805 * X9*X16 \
            -604.051 * X9*X18 \
            -0.655676 * X10*X11 \
            -0.817886 * X10*X14 \
            -0.711442 * X11*X14 \
            +0.0229465 * X12*X13 \
            +0.715136 * X13*X14 \
            -21.9173 * X13*X17 \
            -520.634 * X13*X19 \
            +1.08606 * X13*X21 \
            +0.590081 * X14*X21 \
            -0.778078 * X15*X17 \
            -501.52 * X15*X19 \
            -458.257 * X16*X19 \
            -300782 * X17*X18 \
            +69179.1 * X17*X19 \
            +45189.7 * X17*X20 \
            +168.826 * X17*X21 \
            -2752.67 * X17*X22 \
            +6.64076e+06 * X18*X19 \
            -4219.58 * X18*X21 \
            +4085.68 * X19*X21 \
            +437897 * X20*X22 \
            -492.504 * X21*X22 \
            +4885.73 * (X17*X18)**2 \
            -5626.05 * (X17*X19)**2 \
            +40.4748 * (X17*X19)**3)


        # Constraints for heat of zone 14
        @self.Constraint(self.flowsheet().config.time, doc="heat in zone 14")
        def eq_surr_heat_14(b, t):
            X1 = b.in_WWTemp[t, 1]
            X2 = b.in_WWTemp[t, 2]
            X3 = b.in_WWTemp[t, 3]
            X4 = b.in_WWTemp[t, 4]
            X5 = b.in_WWTemp[t, 5]
            X6 = b.in_WWTemp[t, 6]
            X7 = b.in_WWTemp[t, 7]
            X8 = b.in_WWTemp[t, 8]
            X9 = b.in_WWTemp[t, 9]
            X10 = b.in_WWTemp[t, 10]
            X11 = b.in_WWTemp[t, 11]
            X12 = b.in_WWTemp[t, 12]
            X13 = b.in_WWTemp[t, 13]
            X14 = b.in_WWTemp[t, 14]
            X15 = b.in_WWTemp[t, 15]
            X16 = b.in_WWTemp[t, 16]
            X17 = b.flowrate_coal_raw[t]
            X18 = b.mf_H2O_coal_raw[t]
            X19 = b.SR[t]
            X20 = b.SR_lf[t]
            X21 = b.in_2ndAir_Temp[t]
            X22 = b.ratio_PA2coal[t]
            return b.out_heat[t, 14] * b.fheat_ww[t] == (\
            354.487 * X1 \
            -60.8645 * X2 \
            +327.396 * X3 \
            -1456.36 * X7 \
            +362.079 * X9 \
            +985.094 * X10 \
            -15673.4 * X11 \
            -83.4322 * X12 \
            -16319.3 * X13 \
            +68526.3 * X14 \
            -32620.5 * X15 \
            +3619.67 * X16 \
            +320878 * X17 \
            -4.78943e+06 * X18 \
            -1.57962e+07 * X19 \
            -1.23409e+06 * X20 \
            -1973.52 * X21 \
            -219033 * X22 \
            -200963 * log(X1) \
            +98301.9 * log(X3) \
            +817844 * log(X7) \
            +7.5546e+06 * log(X11) \
            -91544.2 * log(X12) \
            +5.29877e+06 * log(X13) \
            -1.95837e+07 * log(X14) \
            +9.89108e+06 * log(X15) \
            -1.68907e+06 * log(X16) \
            -950165 * log(X17) \
            +107663 * log(X18) \
            +1.00977e+07 * log(X19) \
            -135582 * log(X21) \
            +1.54014e+06 * exp(X19) \
            +6.45563 * X13**2 \
            -32.5108 * X14**2 \
            +16.0267 * X15**2 \
            +1020.99 * X17**2 \
            +0.0038864 * X11**3 \
            -23.3243 * X17**3 \
            -0.19899 * X1*X13 \
            -0.363486 * X1*X17 \
            -0.239156 * X1*X21 \
            +93.3993 * X1*X22 \
            +0.156182 * X2*X9 \
            -0.0202177 * X2*X15 \
            -0.436573 * X3*X9 \
            -0.455785 * X3*X11 \
            +149.628 * X3*X19 \
            +0.0307972 * X4*X8 \
            +0.0633193 * X5*X6 \
            +0.2472 * X7*X9 \
            +0.539863 * X7*X12 \
            -24.6396 * X7*X17 \
            -0.174682 * X9*X16 \
            -535.921 * X9*X18 \
            -0.557081 * X10*X14 \
            +2.71052 * X10*X17 \
            -0.794394 * X10*X21 \
            -0.594428 * X11*X14 \
            +0.242245 * X13*X14 \
            -4.38229 * X13*X17 \
            -14.2876 * X14*X17 \
            -491.212 * X15*X18 \
            -406.335 * X15*X19 \
            -379.661 * X16*X19 \
            -257600 * X17*X18 \
            +99274 * X17*X19 \
            +28183 * X17*X20 \
            +80.0701 * X17*X21 \
            -1951.5 * X17*X22 \
            +4.3693e+06 * X18*X19 \
            -3278.37 * X18*X21 \
            +3236.46 * X19*X21 \
            +320678 * X20*X22 \
            -309.344 * X21*X22 \
            +0.000468526 * (X7*X17)**2 \
            +4.7235e-05 * (X8*X17)**2 \
            +4052.08 * (X17*X18)**2 \
            -4222.33 * (X17*X19)**2 \
            +0.000904438 * (X17*X21)**2 \
            +27.4312 * (X17*X19)**3)


        # Constraints for heat of zone 15
        @self.Constraint(self.flowsheet().config.time, doc="heat in zone 15")
        def eq_surr_heat_15(b, t):
            X1 = b.in_WWTemp[t, 1]
            X2 = b.in_WWTemp[t, 2]
            X3 = b.in_WWTemp[t, 3]
            X4 = b.in_WWTemp[t, 4]
            X5 = b.in_WWTemp[t, 5]
            X6 = b.in_WWTemp[t, 6]
            X7 = b.in_WWTemp[t, 7]
            X8 = b.in_WWTemp[t, 8]
            X9 = b.in_WWTemp[t, 9]
            X10 = b.in_WWTemp[t, 10]
            X11 = b.in_WWTemp[t, 11]
            X12 = b.in_WWTemp[t, 12]
            X13 = b.in_WWTemp[t, 13]
            X14 = b.in_WWTemp[t, 14]
            X15 = b.in_WWTemp[t, 15]
            X16 = b.in_WWTemp[t, 16]
            X17 = b.flowrate_coal_raw[t]
            X18 = b.mf_H2O_coal_raw[t]
            X19 = b.SR[t]
            X20 = b.SR_lf[t]
            X21 = b.in_2ndAir_Temp[t]
            X22 = b.ratio_PA2coal[t]
            return b.out_heat[t, 15] * b.fheat_platen[t] == (\
            -1166.56 * X1 \
            -675.095 * X2 \
            +2514.51 * X3 \
            -983.723 * X4 \
            +2330.69 * X5 \
            +362.806 * X6 \
            -7375.6 * X7 \
            -4202.08 * X8 \
            +1141.48 * X9 \
            +8484.16 * X10 \
            +5905.73 * X11 \
            +7386.94 * X12 \
            +11007.4 * X13 \
            -706.412 * X14 \
            +189001 * X15 \
            -663.012 * X16 \
            +3.37318e+06 * X17 \
            -3.87781e+07 * X18 \
            -7.55876e+06 * X19 \
            +5.36976e+08 * X20 \
            -24912.5 * X21 \
            -1.62488e+06 * X22 \
            +191925 * log(X1) \
            +280264 * log(X5) \
            +5.72596e+06 * log(X7) \
            -2.47873e+06 * log(X10) \
            -2.90641e+06 * log(X12) \
            -8.91549e+06 * log(X13) \
            -4.73042e+07 * log(X15) \
            -6.16404e+06 * log(X17) \
            +699130 * log(X18) \
            -5.10172e+06 * log(X19) \
            -2.9072e+08 * log(X20) \
            -9.40002e+07 * exp(X20) \
            -95.6481 * X15**2 \
            +6.12999 * X16**2 \
            -2066.95 * X17**2 \
            -83.1843 * X17**3 \
            -0.554054 * X1*X13 \
            -6.63217 * X1*X17 \
            +643.644 * X1*X22 \
            +2.74375 * X2*X9 \
            -900.098 * X2*X19 \
            +0.248714 * X2*X21 \
            -2.85519 * X3*X9 \
            -2.73086 * X3*X11 \
            +1241.63 * X3*X19 \
            +1.78941 * X4*X11 \
            +1.17457 * X5*X12 \
            +0.870737 * X5*X13 \
            -2873.6 * X5*X18 \
            -5.59825 * X5*X21 \
            +1.62361 * X7*X9 \
            -218.918 * X7*X17 \
            +2919.25 * X7*X18 \
            +5.36545 * X8*X14 \
            +3158.34 * X8*X18 \
            +322.653 * X8*X19 \
            -1.50986 * X9*X16 \
            -2.46079 * X10*X11 \
            -3.55384 * X10*X14 \
            +23.7953 * X10*X17 \
            -4.9544 * X11*X14 \
            -2.72784 * X12*X16 \
            -1658.9 * X12*X18 \
            +152.683 * X12*X19 \
            +1.7608 * X13*X14 \
            -36.076 * X13*X17 \
            +6.70234 * X13*X21 \
            +7.0803 * X14*X21 \
            -623.313 * X15*X17 \
            +6995.08 * X15*X18 \
            -7698.95 * X15*X19 \
            -2378.81 * X16*X19 \
            -38.7111 * X16*X22 \
            -1.97965e+06 * X17*X18 \
            +303198 * X17*X19 \
            +228298 * X17*X20 \
            +655.481 * X17*X21 \
            -14129.1 * X17*X22 \
            +2.62538e+07 * X18*X19 \
            -22257.2 * X18*X21 \
            +22108.4 * X19*X21 \
            +2.40177e+06 * X20*X22 \
            -2276.36 * X21*X22 \
            +0.00398202 * (X7*X17)**2 \
            +0.000472314 * (X11*X17)**2 \
            +0.513637 * (X11*X19)**2 \
            +24092.9 * (X17*X18)**2 \
            -23924.9 * (X17*X19)**2 \
            +0.00567433 * (X17*X21)**2 \
            +158.634 * (X17*X19)**3)


        # Constraints for heat of zone 16
        @self.Constraint(self.flowsheet().config.time, doc="heat in zone 16")
        def eq_surr_heat_16(b, t):
            X1 = b.in_WWTemp[t, 1]
            X2 = b.in_WWTemp[t, 2]
            X3 = b.in_WWTemp[t, 3]
            X4 = b.in_WWTemp[t, 4]
            X5 = b.in_WWTemp[t, 5]
            X6 = b.in_WWTemp[t, 6]
            X7 = b.in_WWTemp[t, 7]
            X8 = b.in_WWTemp[t, 8]
            X9 = b.in_WWTemp[t, 9]
            X10 = b.in_WWTemp[t, 10]
            X11 = b.in_WWTemp[t, 11]
            X12 = b.in_WWTemp[t, 12]
            X13 = b.in_WWTemp[t, 13]
            X14 = b.in_WWTemp[t, 14]
            X15 = b.in_WWTemp[t, 15]
            X16 = b.in_WWTemp[t, 16]
            X17 = b.flowrate_coal_raw[t]
            X18 = b.mf_H2O_coal_raw[t]
            X19 = b.SR[t]
            X20 = b.SR_lf[t]
            X21 = b.in_2ndAir_Temp[t]
            X22 = b.ratio_PA2coal[t]
            return b.out_heat[t, 16] == (\
            21726.1 * X1 \
            -594.784 * X2 \
            +262.963 * X3 \
            -63.3506 * X5 \
            +135.253 * X6 \
            -1427.81 * X7 \
            -427.891 * X8 \
            +113.322 * X9 \
            +1368.63 * X10 \
            +66.8422 * X11 \
            -16289.4 * X12 \
            +1417.92 * X13 \
            +1512.89 * X14 \
            -30229.9 * X15 \
            +13784 * X16 \
            +243631 * X17 \
            -5.91623e+06 * X18 \
            -1.93078e+06 * X19 \
            -1.34208e+06 * X20 \
            -2139.39 * X21 \
            -216815 * X22 \
            -7.73099e+06 * log(X1) \
            +334378 * log(X2) \
            +78052.6 * log(X5) \
            +556781 * log(X7) \
            +5.57566e+06 * log(X12) \
            -770763 * log(X13) \
            -1.04818e+06 * log(X14) \
            +9.08822e+06 * log(X15) \
            -5.07528e+06 * log(X16) \
            -775440 * log(X17) \
            +91240.4 * log(X18) \
            -354870 * log(X19) \
            -7.57492 * X1**2 \
            +5.8658 * X12**2 \
            +14.7655 * X15**2 \
            +1803.07 * X17**2 \
            -0.0065371 * X16**3 \
            -25.4959 * X17**3 \
            -0.335457 * X1*X13 \
            +91.6571 * X1*X22 \
            +0.228331 * X2*X9 \
            -0.418731 * X3*X9 \
            +0.321897 * X3*X12 \
            -0.215099 * X3*X16 \
            +107.049 * X4*X18 \
            -0.076551 * X6*X14 \
            -0.0291378 * X6*X16 \
            +0.396889 * X7*X9 \
            +0.601134 * X7*X14 \
            +0.576428 * X8*X14 \
            +0.0946749 * X8*X15 \
            -0.153844 * X9*X16 \
            -0.408191 * X10*X11 \
            -0.631577 * X10*X14 \
            +4.26333 * X10*X17 \
            -0.909089 * X10*X21 \
            +305.541 * X11*X19 \
            +0.37501 * X13*X14 \
            -5.55432 * X13*X17 \
            +0.459927 * X15*X17 \
            -362.269 * X15*X19 \
            -11.4827 * X16*X17 \
            -542.021 * X16*X19 \
            -20.7479 * X16*X22 \
            -251820 * X17*X18 \
            +87360.4 * X17*X19 \
            +29535.7 * X17*X20 \
            +121.994 * X17*X21 \
            -2034.64 * X17*X22 \
            +4.68421e+06 * X18*X19 \
            -3289.59 * X18*X21 \
            +2922.8 * X19*X21 \
            +346073 * X20*X22 \
            -327.061 * X21*X22 \
            +3924.73 * (X17*X18)**2 \
            -4066.05 * (X17*X19)**2 \
            +27.2454 * (X17*X19)**3)

        # Constraints for ln of unburned carbon
        @self.Constraint(self.flowsheet().config.time, doc="ln of unburned carbon")
        def eq_surr_ln_ubc(b, t):
            X1 = b.in_WWTemp[t,1]
            X2 = b.in_WWTemp[t,2]
            X3 = b.in_WWTemp[t,3]
            X4 = b.in_WWTemp[t,4]
            X5 = b.in_WWTemp[t,5]
            X6 = b.in_WWTemp[t,6]
            X7 = b.in_WWTemp[t,7]
            X8 = b.in_WWTemp[t,8]
            X9 = b.in_WWTemp[t,9]
            X10 = b.in_WWTemp[t,10]
            X11 = b.in_WWTemp[t,11]
            X12 = b.in_WWTemp[t,12]
            X13 = b.in_WWTemp[t,13]
            X14 = b.in_WWTemp[t,14]
            X15 = b.in_WWTemp[t,15]
            X16 = b.in_WWTemp[t,16]
            X17 = b.flowrate_coal_raw[t]
            X18 = b.mf_H2O_coal_raw[t]
            X19 = b.SR[t]
            X20 = b.SR_lf[t]
            X21 = b.in_2ndAir_Temp[t]
            X22 = b.ratio_PA2coal[t]
            return b.out_ubc_flyash[t] == exp(\
            -0.00622998 * X1 \
            +0.000753772 * X2 \
            -0.000120601 * X3 \
            -0.000451688 * X8 \
            -0.000571427 * X9 \
            -0.00172271 * X10 \
            -0.00232964 * X11 \
            -0.000550575 * X14 \
            -0.0020169 * X15 \
            +1.27841 * X17 \
            -2.31661 * X18 \
            +144.922 * X19 \
            -689.253 * X20 \
            +0.00445568 * X21 \
            +4.16487 * log(X1) \
            +0.978169 * log(X2) \
            -8.20549 * log(X17) \
            -106.113 * log(X19) \
            +354.404 * log(X20) \
            +41.4263 * log(X21) \
            -7.96304 * exp(X19) \
            +123.098 * exp(X20) \
            -0.0239891 * X17**2 \
            +0.000136547 * X17**3 \
            -0.00161642 * X2*X19 \
            -1.68072e-08 * X4*X6 \
            -1.21607e-07 * X5*X12 \
            -2.29202e-05 * X7*X22 \
            +1.62909e-05 * X8*X17 \
            +1.41773e-05 * X9*X17 \
            +9.95354e-05 * X10*X17 \
            +2.11008e-06 * X11*X15 \
            +2.50283e-05 * X11*X17 \
            +0.00279471 * X14*X18 \
            +1.07429e-05 * X15*X17 \
            -0.29238 * X17*X18 \
            -0.276297 * X17*X20 \
            -4.82612e-05 * X17*X21 \
            +0.00955259 * X18*X21 \
            -0.100695 * X19*X21 \
            -0.00535995 * X20*X21 \
            -4.13947e-06 * (X4*X18)**2 \
            -9.92196e-10 * (X10*X17)**2 \
            +0.00992987 * (X17*X18)**2 \
            +0.00443128 * (X17*X20)**2 \
            +4.68424e-09 * (X17*X21)**2 \
            +2.68687e-05 * (X19*X21)**2)


        # Constraints for NOx in ppm from surrogate model, converted to mass fraction
        # This is a place holder for a NOx model.  This model does not predict NOx formation
        @self.Constraint(self.flowsheet().config.time, doc="NOx in ppm mass fraction")
        def eq_surr_nox(b, t):
            return b.out_mf_NO_FG[t]*1e4 == 10

        #expression to calculate total heat loss through water wall, platen SH, and roof
        @self.Expression(self.flowsheet().config.time, doc="Total heat loss")
        def heat_total(b,t):
            return sum(b.out_heat[t,j] for j in b.zones)

        self.heat_total_ww = Var(self.flowsheet().config.time,
                        initialize=2e8,
                        doc='heat duty to all water walls')

        # Constraints for total water wall heat duty
        @self.Constraint(self.flowsheet().config.time, doc="heat duty to all water walls")
        def heat_total_ww_eqn(b, t):
            return b.heat_total_ww[t] == (b.heat_total[t] - b.out_heat[t,15] - b.out_heat[t,16])

    def set_initial_condition(self):
        pass

    def initialize(blk, state_args_1=None, state_args_2=None,
                   outlvl=0, solver='ipopt', optarg={'tol': 1e-6}):
        '''
        General Heat Exchanger initialisation routine.

        Keyword Arguments:
            state_args_1 : a dict of arguments to be passed to the property
                           package(s) for side 1 of the heat exchanger to
                           provide an initial state for initialization
                           (see documentation of the specific property package)
                           (default = None).
            state_args_2 : a dict of arguments to be passed to the property
                           package(s) for side 2 of the heat exchanger to
                           provide an initial state for initialization
                           (see documentation of the specific property package)
                           (default = None).
            outlvl : sets output level of initialisation routine
            optarg : solver options dictionary object (default={'tol': 1e-6})
            solver : str indicating whcih solver to use during
                     initialization (default = 'ipopt')

        Returns:
            None
        '''
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="unit")

        opt = SolverFactory(solver)
        opt.options = optarg

        # ---------------------------------------------------------------------
        # Initialize inlet property blocks
        flags1 = blk.side_1.initialize(
            outlvl=0,
            optarg=optarg,
            solver=solver,
            state_args=state_args_1
        )
        flags2 = blk.side_2.initialize(
            outlvl=0,
            optarg=optarg,
            solver=solver,
            state_args=state_args_2
        )
        init_log.info_high("Initialization Step 1 Complete.")

        # It seems that we need to release the states otherwise we cannot
        # calculate the PA and SA composition at inlets
        blk.side_1.release_state(flags1, outlvl+1)
        blk.side_2.release_state(flags2, outlvl+1)
        dof = degrees_of_freedom(blk)
        init_log.info_high(
                "Degree of freedom {}.".format(dof)
            )
        assert dof == 0

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info_high(
                "Initialization Step 2 {}.".format(idaeslog.condition(res))
            )
        init_log.info("Initialization Complete.")

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        # set a default total heat scaling factor
        for v in self.heat_total_ww.values():
            if iscale.get_scaling_factor(v, warning=True) is None:
                iscale.set_scaling_factor(v, 1e-8)

        # set a default zone heat scaling factor
        for v in self.out_heat.values():
            if iscale.get_scaling_factor(v, warning=True) is None:
                iscale.set_scaling_factor(v, 1e-7)

        for t, c in self.side_1_temp_eqn.items():
            sf = iscale.get_scaling_factor(
                self.heat_total_ww[t], default=1, warning=True)
            iscale.constraint_scaling_transform(c, sf)

        for i in range(1, 17):
            ic = getattr(self, f"eq_surr_heat_{i}")
            for t, c in ic.items():
                sf = iscale.get_scaling_factor(
                    self.out_heat[t, i], default=1, warning=True)
                iscale.constraint_scaling_transform(c, sf)

        for t, c in self.heat_total_ww_eqn.items():
            sf = iscale.get_scaling_factor(
                self.heat_total_ww[t], default=1, warning=True)
            iscale.constraint_scaling_transform(c, sf)

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
Standard IDAES heat exchanger model.
Updates: convective heat transfer correlations
"""
# Import Python libraries
import math

# Import Pyomo libraries
from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (
    ControlVolume0DBlock,
    declare_process_block_class,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
    UnitModelBlockData,
    useDefault,
)
import idaes.logger as idaeslog
from idaes.core.util.config import (
    is_physical_parameter_block,
    is_reaction_parameter_block,
)
from idaes.core.util.misc import add_object_reference
from idaes.core.util.constants import Constants as const
import idaes.core.util.scaling as iscale


# Additional import for the unit operation
from pyomo.environ import SolverFactory, value, Var, Param, exp, sqrt, log, sin, cos
from pyomo.opt import TerminationCondition

__author__ = "Jinliang Ma <jinliang.ma.doe.gov>"
__version__ = "2.0.0"


@declare_process_block_class("HeatExchanger")
class HeatExchangerData(UnitModelBlockData):
    """
    Standard Heat Exchanger Unit Model Class
    """
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
    CONFIG.declare("flow_type", ConfigValue(
        default='counter-current',
        domain=In(['counter-current','con-current']),
        description="Flow configuration in unit",
        doc="""Flag indicating type of flow arrangement to use for heat
             exchanger (default = 'counter-current')
                - 'counter-current' : counter-current flow arrangement"""))
    CONFIG.declare("tube_arrangement",ConfigValue(
        default='in-line',
        domain=In(['in-line','staggered']),
        description='tube configuration',
        doc='tube arrangement could be in-line and staggered'))
    CONFIG.declare("side_1_water_phase",ConfigValue(
        default='Liq',
        domain=In(['Liq','Vap']),
        description='side 1 water phase',
        doc='define water phase for property calls'))
    CONFIG.declare("has_radiation",ConfigValue(
        default=False,
        domain=In([False,True]),
        description='Has side 2 gas radiation',
        doc='define if side 2 gas radiation is to be considered'))

    def build(self):
        """
        Begin building model-DAE transformation)

        Args:
            None

        Returns:
            None
        """
        # Call UnitModel.build to setup dynamics
        super(HeatExchangerData, self).build()

        # Build Holdup Block
        self.side_1 = ControlVolume0DBlock(default={
        "dynamic": self.config.dynamic,
        "has_holdup": self.config.has_holdup,
        "property_package": self.config.side_1_property_package,
        "property_package_args": self.config.side_1_property_package_args})

        self.side_2 = ControlVolume0DBlock(default={
        "dynamic": self.config.dynamic,
        "has_holdup": self.config.has_holdup,
        "property_package": self.config.side_2_property_package,
        "property_package_args": self.config.side_2_property_package_args})

        # Add Geometry
        self.side_1.add_geometry()
        self.side_2.add_geometry()

        # Add state block
        self.side_1.add_state_blocks(has_phase_equilibrium=False)

        # Add material balance
        self.side_1.add_material_balances(
            balance_type=self.config.material_balance_type)
        # add energy balance
        self.side_1.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_heat_transfer=self.config.has_heat_transfer)
        # add momentum balance
        self.side_1.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=self.config.has_pressure_change)


        # Add state block
        self.side_2.add_state_blocks(has_phase_equilibrium=False)

        # Add material balance
        self.side_2.add_material_balances(
            balance_type=self.config.material_balance_type)
        # add energy balance
        self.side_2.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_heat_transfer=self.config.has_heat_transfer)
        # add momentum balance
        self.side_2.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=self.config.has_pressure_change)

        # Set Unit Geometry and Holdup Volume
        self._set_geometry()

        self.phase_1 = self.config.side_1_water_phase

        # Construct performance equations
        self._make_performance()

        # Construct performance equations
        if self.config.flow_type == "counter-current":
            self._make_counter_current()
        else:
            self._make_con_current()

        self.add_inlet_port(name="side_1_inlet", block=self.side_1)
        self.add_inlet_port(name="side_2_inlet", block=self.side_2)
        self.add_outlet_port(name="side_1_outlet", block=self.side_1)
        self.add_outlet_port(name="side_2_outlet", block=self.side_2)


    def _set_geometry(self):
        """
        Define the geometry of the unit as necessary, and link to holdup volume

        Args:
            None

        Returns:
            None
        """

        # Elevation difference (outlet - inlet) for static pressure calculation
        self.delta_elevation = Var(initialize=0.0,
                             doc='Elevation increase used for static pressure calculation')

        # Number of tube columns in the cross section plane perpendicular to shell side fluid flow (y direction)
        self.ncol_tube = Var(initialize=10.0,
                             doc='number of tube columns')

        # Number of tube rows in the direction of shell side fluid flow (x direction)
        self.nrow_tube = Var(initialize=10.0,
                             doc='number of tube rows')

        # Number of inlet tube rows
        self.nrow_inlet = Var(initialize=1,
                              doc='number of inlet tube rows')

        # Length of a tube in z direction for each path
        self.length_tube = Var(initialize=5.0,
                               doc='tube length')

        # Inner diameter of tubes
        self.di_tube = Var(initialize=0.05,
                           doc='inner diameter of tube')

        # Thickness of tube
        self.thickness_tube = Var(initialize=0.005,
                                  doc='tube thickness')

        # Pitch of tubes between two neighboring columns (in y direction). Always greater than tube outside diameter
        self.pitch_y = Var(initialize=0.1,
                           doc='pitch between two neighboring columns')

        # Pitch of tubes between two neighboring rows (in x direction). Always greater than tube outside diameter
        self.pitch_x = Var(initialize=0.1,
                           doc='pitch between two neighboring rows')

        # Tube outside diameter
        @self.Expression(doc="outside diameter of tube")
        def do_tube(b):
            return b.di_tube + b.thickness_tube * 2.0

        # Mean bean length for radiation
        @self.Expression(doc="mean bean length")
        def mbl(b):
            return 3.6*(b.pitch_x*b.pitch_y/const.pi/b.do_tube - b.do_tube/4.0)

        # Mean bean length for radiation divided by sqrt(2)
        @self.Expression(doc="mean bean length")
        def mbl_div2(b):
            return b.mbl/sqrt(2.0)

        # Mean bean length for radiation multiplied by sqrt(2)
        @self.Expression(doc="mean bean length")
        def mbl_mul2(b):
            return b.mbl*sqrt(2.0)

        # Number of 180 bends for the tube
        @self.Expression(doc="nbend_tube")
        def nbend_tube(b):
            return b.nrow_tube / b.nrow_inlet

        # Total flow area on tube side
        @self.Expression(doc="total flow area on tube side")
        def area_flow_tube(b):
            return 0.25 * const.pi * b.di_tube**2.0 * b.ncol_tube * b.nrow_inlet

        # Total flow area on shell side
        @self.Expression(doc="total flow area on tube side")
        def area_flow_shell(b):
            return b.length_tube * (b.pitch_y - b.do_tube) * b.ncol_tube

        # Total heat transfer area based on outside diameter
        @self.Expression(doc="total heat transfer area based on tube outside diamer")
        def area_heat_transfer(b):
            return const.pi * b.do_tube * b.length_tube * b.ncol_tube * b.nrow_tube

        # Ratio of pitch_x/do_tube
        @self.Expression(doc="ratio of pitch in x direction to tube outside diamer")
        def pitch_x_to_do(b):
            return b.pitch_x / b.do_tube

        # Ratio of pitch_y/do_tube
        @self.Expression(doc="ratio of pitch in y direction to tube outside diamer")
        def pitch_y_to_do(b):
            return b.pitch_y / b.do_tube

        if self.config.has_holdup  is True:
            add_object_reference(self, "volume_side_1", self.side_1.volume)
            add_object_reference(self, "volume_side_2", self.side_2.volume)
            # Total tube side valume
            self.Constraint(doc="total tube side volume")
            def volume_side_1_eqn(b):
                return b.volumne_side_1 == 0.25 * const.pi * b.di_tube**2.0 * b.length_tube * b.ncol_tube * b.nrow_tube
            # Total shell side valume
            self.Constraint(doc="total shell side volume")
            def volume_side_2_eqn(b):
                return b.volumne_side_2 == b.ncol_tube * b.pitch_y * b.length_tube * b.nrow_tube * b.pitch_x - \
                       0.25 * const.pi * b.do_tube**2.0 * b.length_tube * b.ncol_tube * b.nrow_tube

    def _make_performance(self):
        """
        Define constraints which describe the behaviour of the unit model.

        Args:
            None

        Returns:
            None
        """
        # Set references to balance terms at unit level
        add_object_reference(self, "heat_duty", self.side_1.heat)
        add_object_reference(self, "deltaP_tube", self.side_1.deltaP)
        add_object_reference(self, "deltaP_shell", self.side_2.deltaP)

        # Performance parameters and variables
        # Shell side wall emissivity, converted from parameter to variable
        self.emissivity_wall = Var(initialize=0.7,
                                   doc='shell side wall emissivity')

        # Wall thermal conductivity
        self.therm_cond_wall = Param(initialize=43.0,
                                     doc='loss coefficient of a tube u-turn')

        # Loss coefficient for a 180 degree bend (u-turn), usually related to radius to inside diameter ratio
        self.kloss_uturn = Param(initialize=0.5,
                                 mutable=True,
                                 doc='loss coefficient of a tube u-turn')

        # Heat transfer resistance due to the fouling on tube side
        self.rfouling_tube = Param(initialize=0.0,
                                   mutable=True,
                                   doc='fouling resistance on tube side')

        # Heat transfer resistance due to the fouling on shell side
        self.rfouling_shell = Param(initialize=0.0001,
                                    mutable=True,
                                    doc='fouling resistance on tube side')

        # Correction factor for overall heat transfer coefficient, converted from parameter to variable
        self.fcorrection_htc = Var(initialize=1.0,
                                     doc="correction factor for HTC")

        # Correction factor for tube side pressure drop due to friction
        self.fcorrection_dp_tube = Var(initialize=1.0,
                                     doc="correction factor for tube side pressure drop")

        # Correction factor for shell side pressure drop due to friction
        self.fcorrection_dp_shell = Var(initialize=1.0,
                                     doc="correction factor for shell side pressure drop")

        # Temperature driving force
        self.temperature_driving_force = Var(self.flowsheet().time,
                                             initialize=1.0,
                                             doc='Mean driving force for heat exchange')

        # Gas emissivity at mbl
        self.gas_emissivity = Var(self.flowsheet().time,
                                   initialize=0.5,
                                   doc='emissivity at given mean beam length')

        # Gas emissivity at mbl/sqrt(2)
        self.gas_emissivity_div2 = Var(self.flowsheet().time,
                                   initialize=0.4,
                                   doc='emissivity at mean beam length divided by sqrt of 2')

        # Gas emissivity at mbl*sqrt(2)
        self.gas_emissivity_mul2 = Var(self.flowsheet().time,
                                   initialize=0.6,
                                   doc='emissivity at mean beam length multiplied by sqrt of 2')

        # Gray fraction of gas in entire spectrum
        self.gas_gray_fraction = Var(self.flowsheet().time,
                                   initialize=0.5,
                                   doc='gray fraction of gas in entire spectrum')

        # Gas-surface radiation exchange factor for shell side wall
        self.frad_gas_shell = Var(self.flowsheet().time,
                                   initialize=0.5,
                                   doc='gas-surface radiation exchange factor for shell side wall')

        # Temperature difference at side 1 inlet
        self.side_1_inlet_dT = Var(self.flowsheet().time,
                                   initialize=1.0,
                                   doc='Temperature difference at side 1 inlet')

        # Temperature difference at side 1 outlet
        self.side_1_outlet_dT = Var(self.flowsheet().time,
                                    initialize=1.0,
                                    doc='Temperature difference at side 1 outlet')

        # Overall heat transfer coefficient
        self.overall_heat_transfer_coefficient = Var(self.flowsheet().time,
                                                     initialize=1.0,
                                                     doc='overall heat transfer coefficient')

        # Tube side convective heat transfer coefficient
        self.hconv_tube = Var(self.flowsheet().time,
                              initialize=100.0,
                              doc='tube side convective heat transfer coefficient')

        # Shell side equivalent convective heat transfer coefficient due to radiation
        self.hconv_shell_rad = Var(self.flowsheet().time,
                               initialize=100.0,
                               doc='shell side convective heat transfer coefficient due to radiation')

        # Shell side convective heat transfer coefficient due to convection only
        self.hconv_shell_conv = Var(self.flowsheet().time,
                               initialize=100.0,
                               doc='shell side convective heat transfer coefficient due to convection')

        # Total shell side convective heat transfer coefficient including convection and radiation
        self.hconv_shell_total = Var(self.flowsheet().time,
                               initialize=150.0,
                               doc='total shell side convective heat transfer coefficient')

        # Heat conduction resistance of tube wall
        self.rcond_wall = Var(initialize=1.0,
                              doc='heat conduction resistance of wall')

        # Constraints for gas emissivity
        @self.Constraint(self.flowsheet().time, doc="Gas emissivity")
        def gas_emissivity_eqn(b, t):
            X1 = (b.side_2.properties_in[t].temperature + b.side_2.properties_out[t].temperature)/2
            X2 = b.mbl
            X3 = b.side_2.properties_in[t].pressure
            X4 = b.side_2.properties_in[t].mole_frac_comp['CO2']
            X5 = b.side_2.properties_in[t].mole_frac_comp['H2O']
            X6 = b.side_2.properties_in[t].mole_frac_comp['O2']
            return b.gas_emissivity[t] == \
            -0.000116906 * X1 \
            +1.02113 * X2 \
            +4.81687e-07 * X3 \
            +0.922679 * X4 \
            -0.0708822 * X5 \
            -0.0368321 * X6 \
            +0.121843 * log(X1) \
            +0.0353343 * log(X2) \
            +0.0346181 * log(X3) \
            +0.0180859 * log(X5) \
            -0.256274 * exp(X2) \
            -0.674791 * exp(X4) \
            -0.724802 * sin(X2) \
            -0.0206726 * cos(X2) \
            -9.01012e-05 * cos(X3) \
            -3.09283e-05 * X1*X2 \
            -5.44339e-10 * X1*X3 \
            -0.000196134 * X1*X5 \
            +4.54838e-05 * X1*X6 \
            +7.57411e-07 * X2*X3 \
            +0.0395456 * X2*X4 \
            +0.726625 * X2*X5 \
            -0.034842 * X2*X6 \
            +4.00056e-06 * X3*X5 \
            +5.71519e-09 * (X1*X2)**2 \
            -1.27853 * (X2*X5)**2

        # Constraints for gas emissivity at mbl/sqrt(2)
        @self.Constraint(self.flowsheet().time, doc="Gas emissivity at a lower mean beam length")
        def gas_emissivity_div2_eqn(b, t):
            X1 = (b.side_2.properties_in[t].temperature + b.side_2.properties_out[t].temperature)/2
            X2 = b.mbl_div2
            X3 = b.side_2.properties_in[t].pressure
            X4 = b.side_2.properties_in[t].mole_frac_comp['CO2']
            X5 = b.side_2.properties_in[t].mole_frac_comp['H2O']
            X6 = b.side_2.properties_in[t].mole_frac_comp['O2']
            return b.gas_emissivity_div2[t] == \
            -0.000116906 * X1 \
            +1.02113 * X2 \
            +4.81687e-07 * X3 \
            +0.922679 * X4 \
            -0.0708822 * X5 \
            -0.0368321 * X6 \
            +0.121843 * log(X1) \
            +0.0353343 * log(X2) \
            +0.0346181 * log(X3) \
            +0.0180859 * log(X5) \
            -0.256274 * exp(X2) \
            -0.674791 * exp(X4) \
            -0.724802 * sin(X2) \
            -0.0206726 * cos(X2) \
            -9.01012e-05 * cos(X3) \
            -3.09283e-05 * X1*X2 \
            -5.44339e-10 * X1*X3 \
            -0.000196134 * X1*X5 \
            +4.54838e-05 * X1*X6 \
            +7.57411e-07 * X2*X3 \
            +0.0395456 * X2*X4 \
            +0.726625 * X2*X5 \
            -0.034842 * X2*X6 \
            +4.00056e-06 * X3*X5 \
            +5.71519e-09 * (X1*X2)**2 \
            -1.27853 * (X2*X5)**2

        # Constraints for gas emissivity at mbl*sqrt(2)
        @self.Constraint(self.flowsheet().time, doc="Gas emissivity at a higher mean beam length")
        def gas_emissivity_mul2_eqn(b, t):
            X1 = (b.side_2.properties_in[t].temperature + b.side_2.properties_out[t].temperature)/2
            X2 = b.mbl_mul2
            X3 = b.side_2.properties_in[t].pressure
            X4 = b.side_2.properties_in[t].mole_frac_comp['CO2']
            X5 = b.side_2.properties_in[t].mole_frac_comp['H2O']
            X6 = b.side_2.properties_in[t].mole_frac_comp['O2']
            return b.gas_emissivity_mul2[t] == \
            -0.000116906 * X1 \
            +1.02113 * X2 \
            +4.81687e-07 * X3 \
            +0.922679 * X4 \
            -0.0708822 * X5 \
            -0.0368321 * X6 \
            +0.121843 * log(X1) \
            +0.0353343 * log(X2) \
            +0.0346181 * log(X3) \
            +0.0180859 * log(X5) \
            -0.256274 * exp(X2) \
            -0.674791 * exp(X4) \
            -0.724802 * sin(X2) \
            -0.0206726 * cos(X2) \
            -9.01012e-05 * cos(X3) \
            -3.09283e-05 * X1*X2 \
            -5.44339e-10 * X1*X3 \
            -0.000196134 * X1*X5 \
            +4.54838e-05 * X1*X6 \
            +7.57411e-07 * X2*X3 \
            +0.0395456 * X2*X4 \
            +0.726625 * X2*X5 \
            -0.034842 * X2*X6 \
            +4.00056e-06 * X3*X5 \
            +5.71519e-09 * (X1*X2)**2 \
            -1.27853 * (X2*X5)**2

        # fraction of gray gas spectrum
        @self.Constraint(self.flowsheet().time, doc="fraction of gray gas spectrum")
        def gas_gray_fraction_eqn(b, t):
            return b.gas_gray_fraction[t]*(2*b.gas_emissivity_div2[t] - b.gas_emissivity_mul2[t]) == \
                   b.gas_emissivity_div2[t]**2

        # gas-surface radiation exchange factor between gas and shell side wall
        @self.Constraint(self.flowsheet().time, doc="gas-surface radiation exchange factor between gas and shell side wall")
        def frad_gas_shell_eqn(b, t):
            return b.frad_gas_shell[t]*((1/b.emissivity_wall-1)*b.gas_emissivity[t] + b.gas_gray_fraction[t]) \
                   == b.gas_gray_fraction[t]*b.gas_emissivity[t]

        # equivalent convective heat transfer coefficent due to radiation
        @self.Constraint(self.flowsheet().time, doc="equivalent convective heat transfer coefficent due to radiation")
        def hconv_shell_rad_eqn(b, t):
            return b.hconv_shell_rad[t] == const.stefan_constant*b.frad_gas_shell[t]*((b.side_2.properties_in[t].temperature + \
                   b.side_2.properties_out[t].temperature)/2 + b.side_1.properties_in[t].temperature) * \
                   (((b.side_2.properties_in[t].temperature + b.side_2.properties_out[t].temperature)/2)**2 + \
                   b.side_1.properties_in[t].temperature**2)

        # Energy balance equation
        @self.Constraint(self.flowsheet().time, doc="Energy balance between two sides")
        def energy_balance(b, t):
            return b.side_1.heat[t] == -b.side_2.heat[t]

        # Heat transfer correlation
        @self.Constraint(self.flowsheet().time, doc="Heat transfer correlation")
        def heat_transfer_correlation(b, t):
            return b.heat_duty[t] == (b.overall_heat_transfer_coefficient[t] *
                                  b.area_heat_transfer *
                                  b.temperature_driving_force[t])


        # Driving force
        @self.Constraint(self.flowsheet().time, doc="Log mean temperature difference calculation")
        def LMTD(b, t):
            return b.temperature_driving_force[t] == ((b.side_1_inlet_dT[t]**0.3241 +
                   b.side_1_outlet_dT[t]**0.3241)/1.99996)**(1/0.3241)

        # Tube side heat transfer coefficient and pressure drop
        # -----------------------------------------------------
        # Velocity on tube side
        self.v_tube = Var(self.flowsheet().time, initialize=1.0, doc="velocity on tube side")

        # Reynalds number on tube side
        self.N_Re_tube = Var(self.flowsheet().time, initialize=10000.0, doc="Reynolds number on tube side")

        # Friction factor on tube side
        self.friction_factor_tube = Var(self.flowsheet().time, initialize=1.0, doc='friction factor on tube side')

        # Pressure drop due to friction on tube side
        self.deltaP_tube_friction = Var(self.flowsheet().time, initialize= -10.0, doc="pressure drop due to friction on tube side")

        # Pressure drop due to 180 degree turn on tube side
        self.deltaP_tube_uturn = Var(self.flowsheet().time, initialize= -10.0, doc="pressure drop due to u-turn on tube side")

        # Prandtl number on tube side
        self.N_Pr_tube = Var(self.flowsheet().time, initialize=1, doc="Prandtl number on tube side")

        # Nusselt number on tube side
        self.N_Nu_tube = Var(self.flowsheet().time, initialize=1, doc="Nusselts number on tube side")

        # Velocity equation
        @self.Constraint(self.flowsheet().time, doc="tube side velocity equation")
        def v_tube_eqn(b, t):
            return b.v_tube[t]* b.area_flow_tube * \
                   b.side_1.properties_in[t].dens_mol_phase[self.phase_1] == \
                   b.side_1.properties_in[t].flow_mol

        # Reynolds number
        @self.Constraint(self.flowsheet().time, doc="Reynolds number equation on tube side")
        def N_Re_tube_eqn(b, t):
            return b.N_Re_tube[t] * b.side_1.properties_in[t].visc_d_phase[self.phase_1] == \
                   b.di_tube * b.v_tube[t] * b.side_1.properties_in[t].dens_mass_phase[self.phase_1]

        # Friction factor
        @self.Constraint(self.flowsheet().time, doc="Darcy friction factor on tube side")
        def friction_factor_tube_eqn(b, t):
            return b.friction_factor_tube[t]*b.N_Re_tube[t]**0.25 == 0.3164*b.fcorrection_dp_tube

        # Pressure drop due to friction
        @self.Constraint(self.flowsheet().time, doc="pressure drop due to friction on tube side")
        def deltaP_tube_friction_eqn(b, t):
            return b.deltaP_tube_friction[t] * b.di_tube * b.nrow_inlet == \
                   -0.5 * b.side_1.properties_in[t].dens_mass_phase[self.phase_1] * \
                   b.v_tube[t]**2 * b.friction_factor_tube[t] * b.length_tube * \
                   b.nrow_tube

        # Pressure drop due to u-turn, assuming each path has a one U-turn (consistent with 1-D model)
        @self.Constraint(self.flowsheet().time, doc="pressure drop due to u-turn on tube side")
        def deltaP_tube_uturn_eqn(b, t):
            return b.deltaP_tube_uturn[t] == \
                   -0.5 * b.side_1.properties_in[t].dens_mass_phase[self.phase_1] * \
                   b.v_tube[t]**2 * b.kloss_uturn * b.nrow_tube / b.nrow_inlet

        # Total pressure drop on tube side
        @self.Constraint(self.flowsheet().time, doc="total pressure drop on tube side")
        def deltaP_tube_eqn(b, t):
            return b.deltaP_tube[t] == b.deltaP_tube_friction[t] + b.deltaP_tube_uturn[t] \
                   - b.delta_elevation*const.acceleration_gravity*(b.side_1.properties_in[t].dens_mass_phase[self.phase_1] \
                   + b.side_1.properties_out[t].dens_mass_phase[self.phase_1])/2.0

        # Prandtl number
        @self.Constraint(self.flowsheet().time, doc="Prandtl number equation on tube side")
        def N_Pr_tube_eqn(b, t):
            return b.N_Pr_tube[t] *\
                   b.side_1.properties_in[t].therm_cond_phase[self.phase_1] * \
                   b.side_1.properties_in[t].mw == \
                   b.side_1.properties_in[t].cp_mol_phase[self.phase_1] * \
                   b.side_1.properties_in[t].visc_d_phase[self.phase_1]

        # Nusselts number
        @self.Constraint(self.flowsheet().time, doc="Nusselts number equation on tube side")
        def N_Nu_tube_eqn(b, t):
            return b.N_Nu_tube[t] == 0.023 * b.N_Re_tube[t]**0.8 * b.N_Pr_tube[t]**0.4

        # Heat transfer coefficient
        @self.Constraint(self.flowsheet().time, doc="convective heat transfer coefficient equation on tube side")
        def hconv_tube_eqn(b, t):
            return b.hconv_tube[t]*self.di_tube/1000 == b.N_Nu_tube[t] * \
                   b.side_1.properties_in[t].therm_cond_phase[self.phase_1]/1000


        # Pressure drop and heat transfer coefficient on shell side
        # ----------------------------------------------------------
        # Tube arrangement factor
        if self.config.tube_arrangement == 'in-line':
            self.f_arrangement = Param(initialize=0.788, doc="in-line tube arrangement factor")
        elif self.config.tube_arrangement == 'staggered':
            self.f_arrangement = Param(initialize=1.0, doc="staggered tube arrangement factor")
        else:
            raise Exception()

        # Velocity on shell side
        self.v_shell = Var(self.flowsheet().time, initialize=1.0, doc="velocity on shell side")

        # Reynalds number on shell side
        self.N_Re_shell = Var(self.flowsheet().time, initialize=10000.0, doc="Reynolds number on shell side")

        # Friction factor on shell side
        self.friction_factor_shell = Var(self.flowsheet().time, initialize=1.0, doc='friction factor on shell side')

        # Prandtl number on shell side
        self.N_Pr_shell = Var(self.flowsheet().time, initialize=1, doc="Prandtl number on shell side")

        # Nusselt number on shell side
        self.N_Nu_shell = Var(self.flowsheet().time, initialize=1, doc="Nusselts number on shell side")

        # Velocity equation on shell side
        @self.Constraint(self.flowsheet().time, doc="velocity on shell side")
        def v_shell_eqn(b, t):
            return b.v_shell[t] * b.side_2.properties_in[t].dens_mol_phase["Vap"] * \
                   b.area_flow_shell == b.side_2.properties_in[t].flow_mol

        # Reynolds number
        @self.Constraint(self.flowsheet().time, doc="Reynolds number equation on shell side")
        def N_Re_shell_eqn(b, t):
            return b.N_Re_shell[t] * b.side_2.properties_in[t].visc_d == \
                   b.do_tube * b.v_shell[t] * b.side_2.properties_in[t].dens_mol_phase["Vap"] *\
                   b.side_2.properties_in[t].mw

        # Friction factor on shell side
        if self.config.tube_arrangement == "in-line":
            @self.Constraint(self.flowsheet().time, doc="in-line friction factor on shell side")
            def friction_factor_shell_eqn(b, t):
                return b.friction_factor_shell[t] * b.N_Re_shell[t]**0.15 == \
                       (0.044 + 0.08 * b.pitch_x_to_do /(b.pitch_y_to_do - 1.0)**(0.43 + 1.13 / b.pitch_x_to_do))\
                       *b.fcorrection_dp_shell
        elif self.config.tube_arrangement == "staggered":
            @self.Constraint(self.flowsheet().time, doc="staggered friction factor on shell side")
            def friction_factor_shell_eqn(b, t):
                return b.friction_factor_shell[t] * b.N_Re_shell[t]**0.16 == \
                       (0.25 + 0.118 / (b.pitch_y_to_do - 1.0)**1.08)*b.fcorrection_dp_shell
        else:
            raise Exception()

        # Pressure drop on shell side
        @self.Constraint(self.flowsheet().time, doc="pressure change on shell side")
        def deltaP_shell_eqn(b, t):
            return b.deltaP_shell[t] == -1.4 * b.friction_factor_shell[t] * b.nrow_tube * b.side_2.properties_in[t].dens_mol_phase["Vap"] * \
                   b.side_2.properties_in[t].mw * \
                   b.v_shell[t]**2

        # Prandtl number
        @self.Constraint(self.flowsheet().time, doc="Prandtl number equation on shell side")
        def N_Pr_shell_eqn(b, t):
            return b.N_Pr_shell[t] * b.side_2.properties_in[t].therm_cond * \
                   b.side_2.properties_in[t].mw == \
                   b.side_2.properties_in[t].cp_mol * \
                   b.side_2.properties_in[t].visc_d

        # Nusselt number, currently assume Re>300
        @self.Constraint(self.flowsheet().time, doc="Nusselts number equation on shell side")
        def N_Nu_shell_eqn(b, t):
            return b.N_Nu_shell[t] == b.f_arrangement * 0.33 * b.N_Re_shell[t]**0.6 * b.N_Pr_shell[t]**0.333333

        # Convective heat transfer coefficient on shell side due to convection only
        @self.Constraint(self.flowsheet().time, doc = "Convective heat transfer coefficient equation on shell side due to convection")
        def hconv_shell_conv_eqn(b, t):
            return b.hconv_shell_conv[t] * b.do_tube / 1000 == \
                   b.N_Nu_shell[t] * b.side_2.properties_in[t].therm_cond / 1000

        # Total convective heat transfer coefficient on shell side
        @self.Constraint(self.flowsheet().time, doc = "Total convective heat transfer coefficient equation on shell side")
        def hconv_shell_total_eqn(b, t):
            if self.config.has_radiation == True:
                return b.hconv_shell_total[t] == b.hconv_shell_conv[t] + b.hconv_shell_rad[t]
            else:
                return b.hconv_shell_total[t] == b.hconv_shell_conv[t]

        # Wall conduction heat transfer resistance based on outside surface area
        @self.Constraint(doc="wall conduction heat transfer resistance")
        def rcond_wall_eqn(b):
            return b.rcond_wall * b.therm_cond_wall == 0.5 * b.do_tube * log(b.do_tube / b.di_tube)

        # Overall heat transfer coefficient
        @self.Constraint(self.flowsheet().time, doc="wall conduction heat transfer resistance")
        def overall_heat_transfer_coefficient_eqn(b, t):
            return b.overall_heat_transfer_coefficient[t] * (b.rcond_wall + \
                   b.rfouling_tube + b.rfouling_shell + \
                   1.0 / b.hconv_shell_total[t] + b.do_tube / b.hconv_tube[t] / b.di_tube) == b.fcorrection_htc

    def _make_con_current(self):
        """
        Add temperature driving force Constraints for counter-current flow.

        Args:
            None

        Returns:
            None
        """
        # Temperature Differences
        @self.Constraint(self.flowsheet().time, doc="Side 1 inlet temperature difference")
        def temperature_difference_1(b, t):
            return b.side_1_inlet_dT[t] == (
                       b.side_2.properties_in[t].temperature -
                       b.side_1.properties_in[t].temperature)

        @self.Constraint(self.flowsheet().time, doc="Side 1 outlet temperature difference")
        def temperature_difference_2(b, t):
            return b.side_1_outlet_dT[t] == (
                       b.side_2.properties_out[t].temperature -
                       b.side_1.properties_out[t].temperature)

    def _make_counter_current(self):
        """
        Add temperature driving force Constraints for counter-current flow.

        Args:
            None

        Returns:
            None
        """
        # Temperature Differences
        @self.Constraint(self.flowsheet().time, doc="Side 1 inlet temperature difference")
        def temperature_difference_1(b, t):
            return b.side_1_inlet_dT[t] == (
                       b.side_2.properties_out[t].temperature -
                       b.side_1.properties_in[t].temperature)

        @self.Constraint(self.flowsheet().time, doc="Side 1 outlet temperature difference")
        def temperature_difference_2(b, t):
            return b.side_1_outlet_dT[t] == (
                       b.side_2.properties_in[t].temperature -
                       b.side_1.properties_out[t].temperature)

    def model_check(blk):
        """
        Model checks for unit - calls model checks for both Holdup Blocks.

        Args:
            None

        Returns:
            None
        """
        # Run holdup block model checks
        blk.side_1.model_check()
        blk.side_2.model_check()

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
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="unit")

        opt = SolverFactory(solver)
        opt.options = optarg

        # ---------------------------------------------------------------------
        # Initialize inlet property blocks
        flags1 = blk.side_1.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args_1,
        )
        flags2 = blk.side_2.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args_2,
        )
        init_log.info('Initialisation Step 1 Complete.')

        # ---------------------------------------------------------------------
        # Initialize temperature differentials
        for t in blk.flowsheet().time:
            blk.side_1.properties_out[t].pressure.fix(
                    value(blk.side_1.properties_in[t].pressure)-100.0)
            blk.side_2.properties_out[t].pressure.fix(
                    value(blk.side_2.properties_in[t].pressure)-100.0)
            blk.side_1.properties_out[t].enth_mol.fix(
                    value(blk.side_1.properties_in[t].enth_mol)+100.0)
            blk.side_2.properties_out[t].temperature.fix(
                    value(blk.side_2.properties_in[t].temperature)-1.0)
        # Deactivate Constraints
        blk.heat_transfer_correlation.deactivate()
        blk.LMTD.deactivate()
        blk.energy_balance.deactivate()
        blk.deltaP_tube_eqn.deactivate()
        blk.deltaP_shell_eqn.deactivate()

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info("Initialization Step 2 Complete: {}".format(
                idaeslog.condition(res)
            )
        )

        # Activate energy balance and driving force
        for t in blk.flowsheet().time:
            blk.side_1.properties_out[t].pressure.unfix()
            blk.side_2.properties_out[t].pressure.unfix()
            blk.side_1.properties_out[t].enth_mol.unfix()
            blk.side_2.properties_out[t].temperature.unfix()
        blk.heat_transfer_correlation.activate()
        blk.LMTD.activate()
        blk.energy_balance.activate()
        blk.deltaP_tube_eqn.activate()
        blk.deltaP_shell_eqn.activate()

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info("Initialization Step 3 Complete: {}".format(
                idaeslog.condition(res)
            )
        )
        # ---------------------------------------------------------------------
        # Release Inlet state
        blk.side_1.release_state(flags1, outlvl+1)
        blk.side_2.release_state(flags2, outlvl+1)
        init_log.info_low("Initialization Complete: {}".format(
                idaeslog.condition(res)
            )
        )

    def calculate_scaling_factors(self):
        for t, c in self.heat_transfer_correlation.items():
            sf = iscale.get_scaling_factor(
                self.heat_duty[t], default=1, warning=True)
            iscale.constraint_scaling_transform(c, sf)

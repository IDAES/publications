##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
1-D Cross Flow Heat Exchanger Model With Wall Temperatures

Discretization based on tube rows
"""
from __future__ import division

# Import Pyomo libraries
from pyomo.environ import (SolverFactory, Var, Param, Constraint,
                           value, TerminationCondition, exp, sqrt, log, sin, cos)
from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (ControlVolume1DBlock, UnitModelBlockData,
                        declare_process_block_class,
                        MaterialBalanceType,
                        EnergyBalanceType,
                        MomentumBalanceType,
                        FlowDirection,
                        UnitModelBlockData,
                        useDefault)
from pyomo.dae import DerivativeVar
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.misc import add_object_reference
from idaes.core.util.constants import Constants as const
import idaes.core.util.scaling as iscale

import idaes.logger as idaeslog

__author__ = "Jinliang Ma"

# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("HeatExchangerCrossFlow1D")
class HeatExchangerCrossFlow1DData(UnitModelBlockData):
    """Standard Heat Exchanger Cross Flow Unit Model Class."""

    CONFIG = UnitModelBlockData.CONFIG()
    # Template for config arguments for shell and tube side
    _SideTemplate = ConfigBlock()
    _SideTemplate.declare("dynamic", ConfigValue(
        default=useDefault,
        domain=In([useDefault, True, False]),
        description="Dynamic model flag",
        doc="""Indicates whether this model will be dynamic or not,
**default** = useDefault.
**Valid values:** {
**useDefault** - get flag from parent (default = False),
**True** - set as a dynamic model,
**False** - set as a steady-state model.}"""))
    _SideTemplate.declare("has_holdup", ConfigValue(
        default=False,
        domain=In([True, False]),
        description="Holdup construction flag",
        doc="""Indicates whether holdup terms should be constructed or not.
Must be True if dynamic = True,
**default** - False.
**Valid values:** {
**True** - construct holdup terms,
**False** - do not construct holdup terms}"""))
    _SideTemplate.declare("material_balance_type", ConfigValue(
        default=MaterialBalanceType.componentTotal,
        domain=In(MaterialBalanceType),
        description="Material balance construction flag",
        doc="""Indicates what type of mass balance should be constructed,
**default** - MaterialBalanceType.componentTotal.
**Valid values:** {
**MaterialBalanceType.none** - exclude material balances,
**MaterialBalanceType.componentPhase** - use phase component balances,
**MaterialBalanceType.componentTotal** - use total component balances,
**MaterialBalanceType.elementTotal** - use total element balances,
**MaterialBalanceType.total** - use total material balance.}"""))
    _SideTemplate.declare("energy_balance_type", ConfigValue(
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
    _SideTemplate.declare("momentum_balance_type", ConfigValue(
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
    _SideTemplate.declare("has_pressure_change", ConfigValue(
        default=False,
        domain=In([True, False]),
        description="Pressure change term construction flag",
        doc="""Indicates whether terms for pressure change should be
constructed,
**default** - False.
**Valid values:** {
**True** - include pressure change terms,
**False** - exclude pressure change terms.}"""))
    _SideTemplate.declare("has_phase_equilibrium", ConfigValue(
        default=False,
        domain=In([True, False]),
        description="Phase equilibrium term construction flag",
        doc="""Argument to enable phase equilibrium on the shell side.
- True - include phase equilibrium term
- False - do not include phase equilinrium term"""))
    _SideTemplate.declare("property_package", ConfigValue(
        default=None,
        domain=is_physical_parameter_block,
        description="Property package to use for control volume",
        doc="""Property parameter object used to define property calculations
(default = 'use_parent_value')
- 'use_parent_value' - get package from parent (default = None)
- a ParameterBlock object"""))
    _SideTemplate.declare("property_package_args", ConfigValue(
        default={},
        description="Arguments for constructing shell property package",
        doc="""A dict of arguments to be passed to the PropertyBlockData
and used when constructing these
(default = 'use_parent_value')
- 'use_parent_value' - get package from parent (default = None)
- a dict (see property package for documentation)"""))
    # TODO : We should probably think about adding a consistency check for the
    # TODO : discretisation methdos as well.
    _SideTemplate.declare("transformation_method", ConfigValue(
        default=useDefault,
        description="Discretization method to use for DAE transformation",
        doc="""Discretization method to use for DAE transformation. See Pyomo
documentation for supported transformations."""))
    _SideTemplate.declare("transformation_scheme", ConfigValue(
        default=useDefault,
        description="Discretization scheme to use for DAE transformation",
        doc="""Discretization scheme to use when transformating domain. See Pyomo
documentation for supported schemes."""))


    # Create individual config blocks for shell and tube side
    CONFIG.declare("shell_side",
                   _SideTemplate(doc="shell side config arguments"))
    CONFIG.declare("tube_side",
                   _SideTemplate(doc="tube side config arguments"))

    # Common config args for both sides
    CONFIG.declare("finite_elements", ConfigValue(
        default=5,
        domain=int,
        description="Number of finite elements length domain",
        doc="""Number of finite elements to use when discretizing length
domain (default=5). Should set to the number of tube rows"""))
    CONFIG.declare("collocation_points", ConfigValue(
        default=3,
        domain=int,
        description="Number of collocation points per finite element",
        doc="""Number of collocation points to use per finite element when
discretizing length domain (default=3)"""))
    CONFIG.declare("flow_type", ConfigValue(
        default="co_current",
        domain=In(['co_current', 'counter_current']),
        description="Flow configuration of heat exchanger",
        doc="""Flow configuration of heat exchanger
- co_current: shell and tube flows from 0 to 1
- counter_current: shell side flows from 0 to 1
tube side flows from 1 to 0"""))
    CONFIG.declare("tube_arrangement",ConfigValue(
        default='in-line',
        domain=In(['in-line','staggered']),
        description='tube configuration',
        doc='tube arrangement could be in-line or staggered'))
    CONFIG.declare("tube_side_water_phase",ConfigValue(
        default='Liq',
        domain=In(['Liq','Vap']),
        description='tube side water phase',
        doc='define water phase for property calls'))
    CONFIG.declare("has_radiation",ConfigValue(
        default=False,
        domain=In([False,True]),
        description='Has side 2 gas radiation',
        doc='define if shell side gas radiation is to be considered'))

    def build(self):
        """
        Begin building model (pre-DAE transformation).

        Args:
            None

        Returns:
            None
        """
        # Call UnitModel.build to setup dynamics
        super(HeatExchangerCrossFlow1DData, self).build()

        if (self.config.shell_side.transformation_method !=
                self.config.tube_side.transformation_method) or \
                (self.config.shell_side.transformation_scheme !=
                    self.config.tube_side.transformation_scheme):
                raise ConfigurationError(
                    "1D heatExchanger model only supports similar transformation "
                    "schemes on the shell side and tube side domains for "
                    "both cocurrent and countercurrent flow patterns.")

        # Set flow directions for the control volume blocks and specify
        # dicretisation if not specified.
        if self.config.flow_type == "co_current":
            set_direction_shell = FlowDirection.forward
            set_direction_tube = FlowDirection.forward
            if self.config.shell_side.transformation_method is useDefault:
                _log.warning("Discretization method was "
                             "not specified for the shell side of the "
                             "co-current heat exchanger. "
                             "Defaulting to finite "
                             "difference method on the shell side.")
                self.config.shell_side.transformation_method = \
                    "dae.finite_difference"
            if self.config.tube_side.transformation_method is useDefault:
                _log.warning("Discretization method was "
                             "not specified for the tube side of the "
                             "co-current heat exchanger. "
                             "Defaulting to finite "
                             "difference method on the tube side.")
                self.config.tube_side.transformation_method = \
                    "dae.finite_difference"
            if self.config.shell_side.transformation_scheme is useDefault:
                _log.warning("Discretization scheme was "
                             "not specified for the shell side of the "
                             "co-current heat exchanger. "
                             "Defaulting to backward finite "
                             "difference on the shell side.")
                self.config.shell_side.transformation_scheme = "BACKWARD"
            if self.config.tube_side.transformation_scheme is useDefault:
                _log.warning("Discretization scheme was "
                             "not specified for the tube side of the "
                             "co-current heat exchanger. "
                             "Defaulting to backward finite "
                             "difference on the tube side.")
                self.config.tube_side.transformation_scheme = "BACKWARD"
        else:
            set_direction_shell = FlowDirection.forward
            set_direction_tube = FlowDirection.backward
            if self.config.shell_side.transformation_method is useDefault:
                _log.warning("Discretization method was "
                             "not specified for the shell side of the "
                             "counter-current heat exchanger. "
                             "Defaulting to finite "
                             "difference method on the shell side.")
                self.config.shell_side.transformation_method = \
                    "dae.finite_difference"
            if self.config.tube_side.transformation_method is useDefault:
                _log.warning("Discretization method was "
                             "not specified for the tube side of the "
                             "counter-current heat exchanger. "
                             "Defaulting to finite "
                             "difference method on the tube side.")
                self.config.tube_side.transformation_method = \
                    "dae.finite_difference"
            if self.config.shell_side.transformation_scheme is useDefault:
                _log.warning("Discretization scheme was "
                             "not specified for the shell side of the "
                             "counter-current heat exchanger. "
                             "Defaulting to backward finite "
                             "difference on the shell side.")
                self.config.shell_side.transformation_scheme = "BACKWARD"
            if self.config.tube_side.transformation_scheme is useDefault:
                _log.warning("Discretization scheme was "
                             "not specified for the tube side of the "
                             "counter-current heat exchanger. "
                             "Defaulting to forward finite "
                             "difference on the tube side.")
                self.config.tube_side.transformation_scheme = "BACKWARD"


        # Control volume 1D for shell and tube, set to steady-state for fluid on both sides
        self.shell = ControlVolume1DBlock(default={
            "dynamic": False, #self.config.shell_side.dynamic,
            "has_holdup": self.config.shell_side.has_holdup,
            "property_package": self.config.shell_side.property_package,
            "property_package_args":
                self.config.shell_side.property_package_args,
            "transformation_method": self.config.shell_side.transformation_method,
            "transformation_scheme": self.config.shell_side.transformation_scheme,
            "finite_elements": self.config.finite_elements,
            "collocation_points": self.config.collocation_points})

        self.tube = ControlVolume1DBlock(default={
            "dynamic": False, #self.config.tube_side.dynamic,
            "has_holdup": self.config.tube_side.has_holdup,
            "property_package": self.config.tube_side.property_package,
            "property_package_args":
                self.config.tube_side.property_package_args,
            "transformation_method": self.config.tube_side.transformation_method,
            "transformation_scheme": self.config.tube_side.transformation_scheme,
            "finite_elements": self.config.finite_elements,
            "collocation_points": self.config.collocation_points})


        self.shell.add_geometry(flow_direction=set_direction_shell)
        self.tube.add_geometry(flow_direction=set_direction_tube)

        self.shell.add_state_blocks(
            information_flow=set_direction_shell,
            has_phase_equilibrium=self.config.shell_side.has_phase_equilibrium)
        self.tube.add_state_blocks(
            information_flow=set_direction_tube,
            has_phase_equilibrium=self.config.tube_side.has_phase_equilibrium)

        # Populate shell
        self.shell.add_material_balances(
            balance_type=self.config.shell_side.material_balance_type,
            has_phase_equilibrium=self.config.shell_side.has_phase_equilibrium)

        self.shell.add_energy_balances(
            balance_type=self.config.shell_side.energy_balance_type,
            has_heat_transfer=True)

        self.shell.add_momentum_balances(
            balance_type=self.config.shell_side.momentum_balance_type,
            has_pressure_change=self.config.shell_side.has_pressure_change)

        self.shell.apply_transformation()

        # Populate tube
        self.tube.add_material_balances(
            balance_type=self.config.tube_side.material_balance_type,
            has_phase_equilibrium=self.config.tube_side.has_phase_equilibrium)

        self.tube.add_energy_balances(
            balance_type=self.config.tube_side.energy_balance_type,
            has_heat_transfer=True)

        self.tube.add_momentum_balances(
            balance_type=self.config.tube_side.momentum_balance_type,
            has_pressure_change=self.config.tube_side.has_pressure_change)

        self.tube.apply_transformation()

        # Add Ports for shell side
        self.add_inlet_port(name="shell_inlet", block=self.shell)
        self.add_outlet_port(name="shell_outlet", block=self.shell)

        # Add Ports for tube side
        self.add_inlet_port(name="tube_inlet", block=self.tube)
        self.add_outlet_port(name="tube_outlet", block=self.tube)

        self.phase_s = self.config.tube_side_water_phase

        self._make_geometry()

        self._make_performance()

    def _make_geometry(self):
        """
        Constraints for unit model.

        Args:
            None

        Returns:
            None
        """
        # Add reference to control volume geometry
        add_object_reference(self,
                             "area_flow_shell",
                             self.shell.area)
        add_object_reference(self,
                             "length_flow_shell",
                             self.shell.length)
        add_object_reference(self,
                             "area_flow_tube",
                             self.tube.area)
        # total tube length of flow path
        add_object_reference(self,
                             "length_flow_tube",
                             self.tube.length)


        # Elevation difference (outlet - inlet) for static pressure calculation
        self.delta_elevation = Var(initialize=0.0,
                             doc='Elevation increase used for static pressure calculation')

        # Number of tube columns in the cross section plane perpendicular to shell side fluid flow (y direction)
        self.ncol_tube = Var(initialize=10.0,
                             doc='number of tube columns')

        # Number of segments of tube bundles
        self.nseg_tube = Var(initialize=10.0,
                             doc='number of tube segments')

        # Number of inlet tube rows
        self.nrow_inlet = Var(initialize=1,
                              doc='number of inlet tube rows')

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

        # Length of tube per segment in z direction
        self.length_tube_seg = Var(initialize=1.0,
                           doc='length of tube per segment')

        # Minimum cross section area on shell side
        self.area_flow_shell_min = Var(initialize=1.0,
                           doc='minimum flow area on shell side')

        # total number of tube rows
        @self.Expression(doc="total number of tube rows")
        def nrow_tube(b):
            return b.nseg_tube * b.nrow_inlet

        # Tube outside diameter
        @self.Expression(doc="outside diameter of tube")
        def do_tube(b):
            return b.di_tube + b.thickness_tube * 2.0

        # Mean bean length for radiation
        if self.config.has_radiation == True:
            @self.Expression(doc="mean bean length")
            def mbl(b):
                return 3.6*(b.pitch_x*b.pitch_y/const.pi/b.do_tube - b.do_tube/4.0)

            # Mean bean length for radiation divided by sqrt(2)
            @self.Expression(doc="sqrt(1/2) of mean bean length")
            def mbl_div2(b):
                return b.mbl/sqrt(2.0)

            # Mean bean length for radiation multiplied by sqrt(2)
            @self.Expression(doc="sqrt(2) of mean bean length")
            def mbl_mul2(b):
                return b.mbl*sqrt(2.0)

        # Ratio of pitch_x/do_tube
        @self.Expression(doc="ratio of pitch in x direction to tube outside diamer")
        def pitch_x_to_do(b):
            return b.pitch_x / b.do_tube

        # Ratio of pitch_y/do_tube
        @self.Expression(doc="ratio of pitch in y direction to tube outside diamer")
        def pitch_y_to_do(b):
            return b.pitch_y / b.do_tube

        # Total cross section area of tube metal per segment
        @self.Expression(doc="total cross section area of tube metal per segment")
        def area_wall_seg(b):
            return 0.25*const.pi*(b.do_tube**2 - b.di_tube**2)*b.ncol_tube*b.nrow_inlet

        # Length of shell side flow
        @self.Constraint(doc="Length of shell side flow")
        def length_flow_shell_eqn(b):
            return b.length_flow_shell == b.nrow_tube * b.pitch_x

        # Length of tube side flow
        @self.Constraint(doc="Length of tube side flow")
        def length_flow_tube_eqn(b):
            return b.length_flow_tube == b.nseg_tube * b.length_tube_seg

        # Total flow area on tube side
        @self.Constraint(doc="Total area of tube flow")
        def area_flow_tube_eqn(b):
            return b.area_flow_tube == 0.25 * const.pi * b.di_tube**2.0 * b.ncol_tube * b.nrow_inlet

        # Average flow area on shell side
        @self.Constraint(doc="Average cross section area of shell side flow")
        def area_flow_shell_eqn(b):
            return b.length_flow_shell*b.area_flow_shell == \
                   b.length_tube_seg*b.length_flow_shell*b.pitch_y*b.ncol_tube - \
                   b.ncol_tube*b.nrow_tube*0.25*const.pi*b.do_tube**2*b.length_tube_seg

        # Minimum flow area on shell side
        @self.Constraint(doc="Minimum flow area on shell side")
        def area_flow_shell_min_eqn(b):
            return b.area_flow_shell_min == b.length_tube_seg*(b.pitch_y-b.do_tube)*b.ncol_tube

        # Note that the volumes of both sides are calculated by the ControlVolume1D

    def _make_performance(self):
        """
        Constraints for unit model.

        Args:
            None

        Returns:
            None
        """
        # Reference
        add_object_reference(self, "heat_tube", self.tube.heat)
        add_object_reference(self, "heat_shell", self.shell.heat)
        add_object_reference(self, "deltaP_tube", self.tube.deltaP)
        add_object_reference(self, "deltaP_shell", self.shell.deltaP)

        # Parameters
        if self.config.has_radiation == True:
            # tube wall emissivity, converted from parameter to variable
            self.emissivity_wall = Var(initialize=0.7,
                                     doc='shell side wall emissivity')

        # Wall thermal conductivity
        self.therm_cond_wall = Param(initialize=1.0, mutable=True,
                                     doc='loss coefficient of a tube u-turn should be 43.0')

        # Wall heat capacity
        self.cp_wall = Param(initialize=502.4, mutable=True,
                                     doc='metal wall heat capacity')

        # Wall density
        self.density_wall = Param(initialize=7800.0, mutable=True,
                                     doc='metal wall density')

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

        # Correction factor for convective heat transfer coefficient on shell side
        self.fcorrection_htc_shell = Var(initialize=1.0,
                                     doc="correction factor for convective HTC on shell")

        # Correction factor for convective heat transfer coefficient on tube side
        self.fcorrection_htc_tube = Var(initialize=1.0,
                                     doc="correction factor for convective HTC on tube side")

        # Correction factor for tube side pressure drop due to friction
        self.fcorrection_dp_tube = Var(initialize=1.0,
                                     doc="correction factor for tube side pressure drop")

        # Correction factor for shell side pressure drop due to friction
        self.fcorrection_dp_shell = Var(initialize=1.0,
                                     doc="correction factor for shell side pressure drop")

        # Performance variables
        if self.config.has_radiation == True:
            # Gas emissivity at mbl
            self.gas_emissivity = Var(self.flowsheet().config.time,
                                  self.shell.length_domain,
                                  initialize=0.5,
                                  doc='emissivity at given mean beam length')

            # Gas emissivity at mbl/sqrt(2)
            self.gas_emissivity_div2 = Var(self.flowsheet().config.time,
                                   self.shell.length_domain,
                                   initialize=0.4,
                                   doc='emissivity at mean beam length divided by sqrt of 2')

            # Gas emissivity at mbl*sqrt(2)
            self.gas_emissivity_mul2 = Var(self.flowsheet().config.time,
                                   self.shell.length_domain,
                                   initialize=0.6,
                                   doc='emissivity at mean beam length multiplied by sqrt of 2')

            # Gray fraction of gas in entire spectrum
            self.gas_gray_fraction = Var(self.flowsheet().config.time,
                                   self.shell.length_domain,
                                   initialize=0.5,
                                   doc='gray fraction of gas in entire spectrum')

            # Gas-surface radiation exchange factor for shell side wall
            self.frad_gas_shell = Var(self.flowsheet().config.time,
                                   self.shell.length_domain,
                                   initialize=0.5,
                                   doc='gas-surface radiation exchange factor for shell side wall')

            # Shell side equivalent convective heat transfer coefficient due to radiation
            self.hconv_shell_rad = Var(self.flowsheet().config.time,
                               self.shell.length_domain,
                               initialize=100.0,
                               doc='shell side convective heat transfer coefficient due to radiation')

        # Tube side convective heat transfer coefficient
        self.hconv_tube = Var(self.flowsheet().config.time,
                              self.tube.length_domain,
                              initialize=100.0,
                              doc='tube side convective heat transfer coefficient')

        # Shell side convective heat transfer coefficient due to convection only
        self.hconv_shell_conv = Var(self.flowsheet().config.time,
                               self.shell.length_domain,
                               initialize=100.0,
                               doc='shell side convective heat transfer coefficient due to convection')

        # Total shell side convective heat transfer coefficient including convection and radiation
        self.hconv_shell_total = Var(self.flowsheet().config.time,
                               self.shell.length_domain,
                               initialize=150.0,
                               doc='total shell side convective heat transfer coefficient')

        # Boundary wall temperature on shell side
        self.temp_wall_shell = Var(self.flowsheet().config.time,
                               self.tube.length_domain,
                               initialize=500,
                               doc='boundary wall temperature on shell side')

        # Boundary wall temperature on tube side
        self.temp_wall_tube = Var(self.flowsheet().config.time,
                               self.tube.length_domain,
                               initialize=500,
                               doc='boundary wall temperature on tube side')

        # Centeral wall temperature of tube metal, used to calculate energy contained by tube metal
        self.temp_wall_center = Var(self.flowsheet().config.time,
                               self.tube.length_domain,
                               initialize=500,
                               doc='tube wall temperature at center')

        # Tube wall heat holdup per length of tube
        self.heat_holdup = Var(self.flowsheet().config.time,
                               self.tube.length_domain,
                               initialize=1,
                               doc='tube wall heat holdup per length of tube')

        # Tube wall heat accumulation term
        if self.config.dynamic is True:
            self.heat_accumulation = DerivativeVar(
                        self.heat_holdup,
                        wrt=self.flowsheet().config.time,
                        doc="Tube wall heat accumulation per unit length")

        def heat_accumulation_term(b, t, x):
               return b.heat_accumulation[t,x] if b.config.dynamic else 0

        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc="heat holdup of tube metal")
        def heat_holdup_eqn(b, t, x):
            return b.heat_holdup[t,x] == b.cp_wall*b.density_wall*b.area_wall_seg*b.temp_wall_center[t,x]

        if self.config.has_radiation == True:
            # Constraints for gas emissivity
            @self.Constraint(self.flowsheet().config.time, self.shell.length_domain, doc="Gas emissivity")
            def gas_emissivity_eqn(b, t, x):
                X1 = b.shell.properties[t, x].temperature
                X2 = b.mbl
                X3 = b.shell.properties[t, x].pressure
                X4 = b.shell.properties[t, x].mole_frac_comp['CO2']
                X5 = b.shell.properties[t, x].mole_frac_comp['H2O']
                X6 = b.shell.properties[t, x].mole_frac_comp['O2']
                return b.gas_emissivity[t,x] == \
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
            @self.Constraint(self.flowsheet().config.time, self.shell.length_domain, doc="Gas emissivity at a lower mean beam length")
            def gas_emissivity_div2_eqn(b, t, x):
                X1 = b.shell.properties[t, x].temperature
                X2 = b.mbl_div2
                X3 = b.shell.properties[t, x].pressure
                X4 = b.shell.properties[t, x].mole_frac_comp['CO2']
                X5 = b.shell.properties[t, x].mole_frac_comp['H2O']
                X6 = b.shell.properties[t, x].mole_frac_comp['O2']
                return b.gas_emissivity_div2[t,x] == \
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
            @self.Constraint(self.flowsheet().config.time, self.shell.length_domain, doc="Gas emissivity at a higher mean beam length")
            def gas_emissivity_mul2_eqn(b, t, x):
                X1 = b.shell.properties[t, x].temperature
                X2 = b.mbl_mul2
                X3 = b.shell.properties[t, x].pressure
                X4 = b.shell.properties[t, x].mole_frac_comp['CO2']
                X5 = b.shell.properties[t, x].mole_frac_comp['H2O']
                X6 = b.shell.properties[t, x].mole_frac_comp['O2']
                return b.gas_emissivity_mul2[t,x] == \
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
            @self.Constraint(self.flowsheet().config.time, self.shell.length_domain, doc="fraction of gray gas spectrum")
            def gas_gray_fraction_eqn(b, t, x):
                return b.gas_gray_fraction[t,x]*(2*b.gas_emissivity_div2[t,x] - b.gas_emissivity_mul2[t,x]) == \
            	   b.gas_emissivity_div2[t,x]**2

            # gas-surface radiation exchange factor between gas and shell side wall
            @self.Constraint(self.flowsheet().config.time, self.shell.length_domain,
                         doc="gas-surface radiation exchange factor between gas and shell side wall")
            def frad_gas_shell_eqn(b, t, x):
                return b.frad_gas_shell[t,x]*((1/b.emissivity_wall-1)*b.gas_emissivity[t,x] + b.gas_gray_fraction[t,x]) \
            	   == b.gas_gray_fraction[t,x]*b.gas_emissivity[t,x]

            # equivalent convective heat transfer coefficent due to radiation
            @self.Constraint(self.flowsheet().config.time, self.shell.length_domain,
                         doc="equivalent convective heat transfer coefficent due to radiation")
            def hconv_shell_rad_eqn(b, t, x):
                return b.hconv_shell_rad[t,x] == const.stefan_constant*b.frad_gas_shell[t,x]*(b.shell.properties[t,x].temperature \
            	   + b.temp_wall_shell[t,x]) * (b.shell.properties[t,x].temperature**2 + b.temp_wall_shell[t,x]**2)


        # Tube side heat transfer coefficient and pressure drop
        # -----------------------------------------------------
        # Velocity on tube side
        self.v_tube = Var(self.flowsheet().config.time, self.tube.length_domain, initialize=1.0, doc="velocity on tube side")

        # Reynalds number on tube side
        self.N_Re_tube = Var(self.flowsheet().config.time, self.tube.length_domain, initialize=10000.0,
                             doc="Reynolds number on tube side")

        # Friction factor on tube side
        self.friction_factor_tube = Var(self.flowsheet().config.time, self.tube.length_domain, initialize=1.0,
                                        doc='friction factor on tube side')

        # Pressure drop due to friction on tube side
        self.deltaP_tube_friction = Var(self.flowsheet().config.time, self.tube.length_domain, initialize= -10.0,
                    doc="pressure drop due to friction on tube side")

        # Pressure drop due to 180 degree turn on tube side
        self.deltaP_tube_uturn = Var(self.flowsheet().config.time, self.tube.length_domain, initialize= -10.0,
                    doc="pressure drop due to u-turn on tube side")

        # Prandtl number on tube side
        self.N_Pr_tube = Var(self.flowsheet().config.time, self.tube.length_domain, initialize=1.0,
                             doc="Prandtl number on tube side")

        # Nusselt number on tube side
        self.N_Nu_tube = Var(self.flowsheet().config.time, self.tube.length_domain, initialize=1,
                             doc="Nusselts number on tube side")

        # Velocity equation
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc="tube side velocity equation")
        def v_tube_eqn(b, t, x):
            return b.v_tube[t,x]* b.area_flow_tube * \
                   b.tube.properties[t,x].dens_mol_phase[self.phase_s] == \
                   b.tube.properties[t,x].flow_mol

        # Reynolds number
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc="Reynolds number equation on tube side")
        def N_Re_tube_eqn(b, t, x):
            return b.N_Re_tube[t,x] * b.tube.properties[t,x].visc_d_phase[self.phase_s] == \
                   b.di_tube * b.v_tube[t,x] * b.tube.properties[t,x].dens_mass_phase[self.phase_s]

        # Friction factor
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc="Darcy friction factor on tube side")
        def friction_factor_tube_eqn(b, t, x):
            return b.friction_factor_tube[t,x]*b.N_Re_tube[t,x]**0.25 == 0.3164*b.fcorrection_dp_tube

        # Pressure drop due to friction per tube length
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc="pressure drop due to friction per tube length")
        def deltaP_tube_friction_eqn(b, t, x):
            return b.deltaP_tube_friction[t,x] * b.di_tube == \
                   -0.5 * b.tube.properties[t,x].dens_mass_phase[self.phase_s] * \
                   b.v_tube[t,x]**2 * b.friction_factor_tube[t,x]

        # Pressure drop due to u-turn
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc="pressure drop due to u-turn on tube side")
        def deltaP_tube_uturn_eqn(b, t, x):
            return b.deltaP_tube_uturn[t,x]*b.length_tube_seg == \
                   -0.5 * b.tube.properties[t,x].dens_mass_phase[self.phase_s] * \
                   b.v_tube[t,x]**2*b.kloss_uturn

        # Total pressure drop on tube side
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc="total pressure drop on tube side")
        def deltaP_tube_eqn(b, t, x):
            return b.deltaP_tube[t, x] == (b.deltaP_tube_friction[t,x] + b.deltaP_tube_uturn[t,x] \
                   - b.delta_elevation/b.nseg_tube*const.acceleration_gravity*\
                   b.tube.properties[t,x].dens_mass_phase[self.phase_s]/b.length_tube_seg)

        # Prandtl number
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc="Prandtl number equation on tube side")
        def N_Pr_tube_eqn(b, t, x):
            return b.N_Pr_tube[t,x] * b.tube.properties[t,x].therm_cond_phase[self.phase_s] * \
                   b.tube.properties[t,x].mw == b.tube.properties[t,x].cp_mol_phase[self.phase_s] * \
                   b.tube.properties[t,x].visc_d_phase[self.phase_s]

        # Nusselts number
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc="Nusselts number equation on tube side")
        def N_Nu_tube_eqn(b, t, x):
            return b.N_Nu_tube[t,x] == 0.023 * b.N_Re_tube[t,x]**0.8 * b.N_Pr_tube[t,x]**0.4

        # Heat transfer coefficient
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain,
                         doc="convective heat transfer coefficient equation on tube side")
        def hconv_tube_eqn(b, t, x):
            return b.hconv_tube[t,x]*self.di_tube == b.N_Nu_tube[t,x] * \
                   b.tube.properties[t,x].therm_cond_phase[self.phase_s]*b.fcorrection_htc_tube



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
        self.v_shell = Var(self.flowsheet().config.time, self.shell.length_domain, initialize=1.0, doc="velocity on shell side")

        # Reynalds number on shell side
        self.N_Re_shell = Var(self.flowsheet().config.time, self.shell.length_domain, \
                              initialize=10000.0, doc="Reynolds number on shell side")

        # Friction factor on shell side
        self.friction_factor_shell = Var(self.flowsheet().config.time, self.shell.length_domain, \
                                        initialize=1.0, doc='friction factor on shell side')

        # Prandtl number on shell side
        self.N_Pr_shell = Var(self.flowsheet().config.time, self.shell.length_domain, \
                              initialize=1, doc="Prandtl number on shell side")

        # Nusselt number on shell side
        self.N_Nu_shell = Var(self.flowsheet().config.time, self.shell.length_domain, \
                              initialize=1, doc="Nusselts number on shell side")

        # Velocity equation on shell side, using inlet molar flow rate
        @self.Constraint(self.flowsheet().config.time, self.shell.length_domain, doc="velocity on shell side")
        def v_shell_eqn(b, t, x):
            return b.v_shell[t,x] * b.shell.properties[t,x].dens_mol_phase["Vap"] * \
                   b.area_flow_shell_min == b.shell.properties[t,0].flow_mol

        # Reynolds number
        @self.Constraint(self.flowsheet().config.time, self.shell.length_domain, doc="Reynolds number equation on shell side")
        def N_Re_shell_eqn(b, t, x):
            return b.N_Re_shell[t,x] * b.shell.properties[t,x].visc_d == \
                   b.do_tube * b.v_shell[t,x] * b.shell.properties[t,x].dens_mol_phase["Vap"] *\
                   b.shell.properties[t,x].mw

        # Friction factor on shell side
        if self.config.tube_arrangement == "in-line":
            @self.Constraint(self.flowsheet().config.time, self.shell.length_domain, doc="in-line friction factor on shell side")
            def friction_factor_shell_eqn(b, t, x):
                return b.friction_factor_shell[t,x] * b.N_Re_shell[t,x]**0.15 == (0.044 + \
                       0.08 * b.pitch_x_to_do / (b.pitch_y_to_do - 1.0)**(0.43 + 1.13 / b.pitch_x_to_do))\
                       *b.fcorrection_dp_shell
        elif self.config.tube_arrangement == "staggered":
            @self.Constraint(self.flowsheet().config.time, self.shell.length_domain,
                             doc="staggered friction factor on shell side")
            def friction_factor_shell_eqn(b, t, x):
                return b.friction_factor_shell[t,x] * b.N_Re_shell[t,x]**0.16 == (0.25 + 0.118 / \
                      (b.pitch_y_to_do - 1.0)**1.08)*b.fcorrection_dp_shell
        else:
            raise Exception()

        # Pressure drop on shell side
        @self.Constraint(self.flowsheet().config.time, self.shell.length_domain, doc="pressure change on shell side")
        def deltaP_shell_eqn(b, t, x):
            return b.deltaP_shell[t,x]*b.pitch_x == -1.4 * b.friction_factor_shell[t,x] * \
                   b.shell.properties[t,x].dens_mol_phase["Vap"] * \
                   b.shell.properties[t,x].mw * b.v_shell[t,x]**2

        # Prandtl number
        @self.Constraint(self.flowsheet().config.time, self.shell.length_domain, doc="Prandtl number equation on shell side")
        def N_Pr_shell_eqn(b, t, x):
            return b.N_Pr_shell[t,x] * b.shell.properties[t,x].therm_cond * b.shell.properties[t,x].mw == \
                   b.shell.properties[t,x].cp_mol * b.shell.properties[t,x].visc_d

        # Nusselt number, currently assume Re>300
        @self.Constraint(self.flowsheet().config.time, self.shell.length_domain, doc="Nusselts number equation on shell side")
        def N_Nu_shell_eqn(b, t, x):
            return b.N_Nu_shell[t,x] == b.f_arrangement * 0.33 * b.N_Re_shell[t,x]**0.6 * b.N_Pr_shell[t,x]**0.333333

        # Convective heat transfer coefficient on shell side due to convection only
        @self.Constraint(self.flowsheet().config.time, self.shell.length_domain,
                   doc = "Convective heat transfer coefficient equation on shell side due to convection")
        def hconv_shell_conv_eqn(b, t, x):
            return b.hconv_shell_conv[t,x] * b.do_tube == \
                   b.N_Nu_shell[t,x] * b.shell.properties[t,x].therm_cond*b.fcorrection_htc_shell

        # Total convective heat transfer coefficient on shell side
        @self.Constraint(self.flowsheet().config.time, self.shell.length_domain,
                   doc = "Total convective heat transfer coefficient equation on shell side")
        def hconv_shell_total_eqn(b, t, x):
            if self.config.has_radiation == True:
                return b.hconv_shell_total[t,x] == b.hconv_shell_conv[t,x] + b.hconv_shell_rad[t,x]
            else:
                return b.hconv_shell_total[t,x] == b.hconv_shell_conv[t,x]

        # Energy balance with tube wall
        # ------------------------------------
        # Heat to wall per length on tube side
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc = "heat per length on tube side")
        def heat_tube_eqn(b, t, x):
            return b.heat_tube[t,x]  == b.hconv_tube[t,x]*const.pi*b.di_tube*b.nrow_inlet*b.ncol_tube*\
                   (b.temp_wall_tube[t,x]-b.tube.properties[t,x].temperature)

        # Heat to wall per length on shell side
        @self.Constraint(self.flowsheet().config.time, self.shell.length_domain, doc = "heat per length on shell side")
        def heat_shell_eqn(b, t, x):
            return b.heat_shell[t,x]*b.length_flow_shell == b.length_flow_tube*b.hconv_shell_total[t,x]*\
                   const.pi*b.do_tube*b.nrow_inlet*b.ncol_tube*\
                   (b.temp_wall_shell[t,x]-b.shell.properties[t,x].temperature)

        # Tube side wall temperature
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc = "tube side wall temperature")
        def temp_wall_tube_eqn(b, t, x):
            return b.hconv_tube[t,x]*(b.tube.properties[t,x].temperature-b.temp_wall_tube[t,x])* \
                  (b.thickness_tube/2/b.therm_cond_wall + b.rfouling_tube) == \
                   b.temp_wall_tube[t,x]-b.temp_wall_center[t,x]

        # Shell side wall temperature
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc = "shell side wall temperature")
        def temp_wall_shell_eqn(b, t, x):
            return b.hconv_shell_total[t,x]*(b.shell.properties[t,x].temperature-b.temp_wall_shell[t,x])* \
                   (b.thickness_tube/2/b.therm_cond_wall + b.rfouling_shell) == \
                   b.temp_wall_shell[t,x]-b.temp_wall_center[t,x]

        # Center point wall temperature based on energy balance for tube wall heat holdup
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc = "center point wall temperature")
        def temp_wall_center_eqn(b, t, x):
            return -heat_accumulation_term(b,t,x) == \
                   (b.heat_shell[t,x]*b.length_flow_shell/b.length_flow_tube + b.heat_tube[t,x])

    def set_initial_condition(self):
        if self.config.dynamic is True:
            self.heat_accumulation[:,:].value = 0
            self.heat_accumulation[0,:].fix(0)
            # no accumulation term for fluid side models to avoid pressure waves

    def initialize(blk, shell_state_args=None, tube_state_args=None, outlvl=0,
                   solver='ipopt', optarg={'tol': 1e-6}):
        """
        HeatExchangerCrossFlow1D initialisation routine

        Keyword Arguments:
            state_args : a dict of arguments to be passed to the property
                         package(s) to provide an initial state for
                         initialization (see documentation of the specific
                         property package) (default = None).
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
        """
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="unit")

        opt = SolverFactory(solver)
        opt.options = optarg

        # ---------------------------------------------------------------------
        # Initialize shell block

        flags_tube = blk.tube.initialize(outlvl=0,
                                    optarg=optarg,
                                    solver=solver,
                                    state_args=tube_state_args)

        flags_shell = blk.shell.initialize(outlvl=0,
                                     optarg=optarg,
                                     solver=solver,
                                     state_args=shell_state_args)

        init_log.info_high("Initialization Step 1 Complete.")

        # Set tube thermal conductivity to a small value to avoid IPOPT unable to solve initially
        therm_cond_wall_save = blk.therm_cond_wall.value
        blk.therm_cond_wall = 0.05
        # In Step 2, fix tube metal temperatures fix fluid state variables (enthalpy/temperature and pressure)
        # calculate maximum heat duty assuming infinite area and use half of the maximum duty as initial guess to calculate outlet temperature
        if blk.config.flow_type == "co_current":
            mcp_shell = value(sum(blk.shell_inlet.flow_mol_comp[0,j] \
                   for j in blk.shell.properties[0,0].config.parameters.component_list) * blk.shell.properties[0,0].cp_mol)
            mcp_tube = value(blk.tube_inlet.flow_mol[0]*blk.tube.properties[0,0].cp_mol_phase[blk.phase_s])
            tout_max = (mcp_tube*value(blk.tube.properties[0,0].temperature) + mcp_shell*value(blk.shell.properties[0,0].temperature))/ \
                       (mcp_tube + mcp_shell)
            q_guess = mcp_tube*value(tout_max - value(blk.tube.properties[0,0].temperature))/2
            temp_out_tube_guess = value(blk.tube.properties[0,0].temperature) + q_guess/mcp_tube
            temp_out_shell_guess = value(blk.shell.properties[0,0].temperature) - q_guess/mcp_shell
            if blk.phase_s == 'Liq' and temp_out_tube_guess > value(blk.tube.properties[0,0].temperature_sat):
                print ('Estimated outlet liquid water temperature exceeds the saturation temperature.')
                print ('Estimated outlet liquid water temperature = ', temp_out_tube_guess)
                print ('Saturation temperature at inlet pressure = ', value(blk.tube.properties[0,0].temperature_sat))
                temp_out_tube_guess = value(0.9*blk.tube.properties[0,0].temperature_sat + 0.1*blk.tube.properties[0,0].temperature)
                print ('Reset estimated outlet liquid water temperature = ', temp_out_tube_guess)
        else:
            mcp_shell = value(blk.shell.properties[0,0].flow_mol * blk.shell.properties[0,0].cp_mol)
            mcp_tube = value(blk.tube_inlet.flow_mol[0]*blk.tube.properties[0,1].cp_mol_phase[blk.phase_s])
            print('mcp_shell=', mcp_shell)
            print('mcp_tube=', mcp_tube)
            if mcp_tube < mcp_shell:
                 q_guess = mcp_tube*value(blk.shell.properties[0,0].temperature - blk.tube.properties[0,1].temperature)/2
            else:
                 q_guess = mcp_shell*value(blk.shell.properties[0,0].temperature - blk.tube.properties[0,1].temperature)/2
            temp_out_tube_guess = value(blk.tube.properties[0,1].temperature) + q_guess/mcp_tube
            temp_out_shell_guess = value(blk.shell.properties[0,0].temperature) - q_guess/mcp_shell
            if blk.phase_s == 'Liq' and temp_out_tube_guess > value(blk.tube.properties[0,1].temperature_sat):
                print ('Estimated outlet liquid water temperature exceeds the saturation temperature.')
                print ('Estimated outlet liquid water temperature = ', temp_out_tube_guess)
                print ('Saturation temperature at inlet pressure = ', value(blk.tube.properties[0,1].temperature_sat))
                temp_out_tube_guess = value(0.9*blk.tube.properties[0,1].temperature_sat + 0.1*blk.tube.properties[0,1].temperature)
                print ('Reset estimated outlet liquid water temperature = ', temp_out_tube_guess)


        for t in blk.flowsheet().config.time:
            for z in blk.tube.length_domain:
                if blk.config.flow_type == "co_current":
                    blk.temp_wall_center[t, z].fix(value(
                          0.05*((1-z)*blk.shell.properties[0, 0].temperature + z*temp_out_shell_guess) +
                          0.95*((1-z)*blk.tube.properties[0, 0].temperature + z*temp_out_tube_guess)))
                else:
                    blk.temp_wall_center[t, z].fix(value(
                          0.05*((1-z)*blk.shell.properties[0, 0].temperature + z*temp_out_shell_guess) +
                          0.95*((1-z)*temp_out_tube_guess + z*blk.tube.properties[0, 1].temperature)))
                blk.temp_wall_shell[t, z].fix(blk.temp_wall_center[t, z].value)
                blk.temp_wall_tube[t, z].fix(blk.temp_wall_center[t, z].value)
                blk.temp_wall_shell[t, z].unfix()
                blk.temp_wall_tube[t, z].unfix()

        for t in blk.flowsheet().config.time:
            for z in blk.tube.length_domain:
                blk.tube.properties[t,z].enth_mol.fix(value(blk.tube.properties[t,0].enth_mol))
                blk.tube.properties[t,z].pressure.fix(value(blk.tube.properties[t,0].pressure))

        for t in blk.flowsheet().config.time:
            for z in blk.shell.length_domain:
                blk.shell.properties[t,z].temperature.fix(value(blk.shell.properties[t,0].temperature))
                blk.shell.properties[t,z].pressure.fix(value(blk.shell.properties[t,0].pressure))

        blk.temp_wall_center_eqn.deactivate()
        blk.deltaP_tube_eqn.deactivate()
        blk.deltaP_shell_eqn.deactivate()
        blk.heat_tube_eqn.deactivate()
        blk.heat_shell_eqn.deactivate()

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info_high(
                "Initialization Step 2 {}.".format(idaeslog.condition(res))
            )


        # In Step 3, unfix fluid state variables (enthalpy/temperature and pressure)
        # keep the inlet state variables fixed, otherwise, the degree of freedom > 0
        for t in blk.flowsheet().config.time:
            for z in blk.tube.length_domain:
                blk.tube.properties[t,z].enth_mol.unfix()
                blk.tube.properties[t,z].pressure.unfix()
            if blk.config.flow_type == "co_current":
                blk.tube.properties[t,0].enth_mol.fix(value(blk.tube_inlet.enth_mol[0]))
                blk.tube.properties[t,0].pressure.fix(value(blk.tube_inlet.pressure[0]))
            else:
                blk.tube.properties[t,1].enth_mol.fix(value(blk.tube_inlet.enth_mol[0]))
                blk.tube.properties[t,1].pressure.fix(value(blk.tube_inlet.pressure[0]))

        for t in blk.flowsheet().config.time:
            for z in blk.shell.length_domain:
                blk.shell.properties[t,z].temperature.unfix()
                blk.shell.properties[t,z].pressure.unfix()
            blk.shell.properties[t,0].temperature.fix(value(blk.shell_inlet.temperature[0]))
            blk.shell.properties[t,0].pressure.fix(value(blk.shell_inlet.pressure[0]))

        blk.deltaP_tube_eqn.activate()
        blk.deltaP_shell_eqn.activate()
        blk.heat_tube_eqn.activate()
        blk.heat_shell_eqn.activate()

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info_high(
                "Initialization Step 3 {}.".format(idaeslog.condition(res))
            )

        blk.temp_wall_center[:,:].unfix()
        blk.temp_wall_center_eqn.activate()

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info_high(
                "Initialization Step 4 {}.".format(idaeslog.condition(res))
            )

        # set the wall thermal conductivity back to the user specified value
        blk.therm_cond_wall = therm_cond_wall_save

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info_high(
                "Initialization Step 5 {}.".format(idaeslog.condition(res))
            )
        blk.tube.release_state(flags_tube)
        blk.shell.release_state(flags_shell)
        init_log.info("Initialization Complete.")

    def calculate_scaling_factors(self):
        for i, c in self.heat_tube_eqn.items():
            sf = iscale.get_scaling_factor(
                self.heat_tube[i], default=1, warning=True)
            iscale.constraint_scaling_transform(c, sf)

        for i, c in self.heat_shell_eqn.items():
            sf = iscale.get_scaling_factor(
                self.heat_shell[i], default=1, warning=True)
            iscale.constraint_scaling_transform(c, sf)

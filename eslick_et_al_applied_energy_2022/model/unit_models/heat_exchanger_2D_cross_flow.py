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

# Import Python libraries
import math

# Import Pyomo libraries
from pyomo.environ import (SolverFactory, Var, Param, Constraint, TransformationFactory,
                           value, TerminationCondition, exp, sqrt, log, log10, sin, cos)
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
from pyomo.dae import ContinuousSet, DerivativeVar
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.misc import add_object_reference
from idaes.core.util.constants import Constants as const
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

__author__ = "Jinliang Ma"

# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("HeatExchangerCrossFlow2D")
class HeatExchangerCrossFlow2DData(UnitModelBlockData):
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
        default='dae.finite_difference',
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
    CONFIG.declare("inside_diameter",ConfigValue(
        default= 0.05,
        description='inside diameter of tube',
        doc='define inside diameter of drum'))
    CONFIG.declare("thickness",ConfigValue(
        default= 0.01,
        description='tube wall thickness',
        doc='define tube wall thickness'))
    CONFIG.declare("radial_elements", ConfigValue(
        default=5,
        domain=int,
        description="Number of finite elements in radius domain",
        doc="""Number of finite elements to use when discretizing radius
domain (default=5)."""))
    def build(self):
        """
        Begin building model (pre-DAE transformation).

        Args:
            None

        Returns:
            None
        """
        # Call UnitModel.build to setup dynamics
        super(HeatExchangerCrossFlow2DData, self).build()

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
                self.config.shell_side.transformation_scheme = "BACKWARD"
            if self.config.tube_side.transformation_scheme is useDefault:
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
                self.config.shell_side.transformation_scheme = "BACKWARD"
            if self.config.tube_side.transformation_scheme is useDefault:
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

        di = self.config.inside_diameter
        thk = self.config.thickness

        # Inner diameter of tubes
        self.ri_scaling = Param(initialize=0.01, mutable=True,
                           doc='inner diameter of tube for scaling')

        # Inner diameter of tubes
        self.di_tube = Param(initialize=di,
                           doc='inner diameter of tube')

        # Thickness of tube
        self.thickness_tube = Param(initialize=thk,
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
        @self.Expression(doc="inside radius of tube")
        def ri_tube(b):
            return b.di_tube/2.0

        @self.Expression(doc="outside radius of tube")
        def ro_tube(b):
            return b.ri_tube + b.thickness_tube

        # Define the continuous domains for model
        self.r = ContinuousSet(bounds=(self.ri_tube/self.ri_scaling, self.ro_tube/self.ri_scaling))

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
        self.therm_cond_wall = Param(initialize=43.0, mutable=True,
                                     doc='thermal conductivity of tube wall')

        # Wall heat capacity
        self.cp_wall = Param(initialize=502.4, mutable=True,
                                     doc='metal wall heat capacity')

        # Wall density
        self.density_wall = Param(initialize=7800.0, mutable=True,
                                     doc='metal wall density')

        # Young modulus
        self.Young_modulus = Param(initialize = 1.90E5, mutable=True,
                                        doc='metal wall Young Modulus')

        # Poisson's ratio
        self.Poisson_ratio = Param(initialize = 0.29, mutable=True,
                                        doc='metal wall Poisson ratio')

        # Coefficient of thermal expansion
        self.coefficient_therm_expansion = Param(initialize = 1.2E-5, mutable=True,
                                            doc='metal wall coefficient of heat capacity')

        # thermal diffusivity of wall
        @self.Expression(doc="Thermal diffusivity of tube wall")
        def diff_therm_wall(b):
            return b.therm_cond_wall/(b.density_wall*b.cp_wall)

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

        # Tube side convective heat transfer coefficient combined with fouling
        self.hconv_tube_foul = Var(self.flowsheet().config.time,
                              self.tube.length_domain,
                              initialize=100.0,
                              doc='tube side convective heat transfer coefficient combined with fouling')

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

        # Total shell side convective heat transfer coefficient combined with fouling
        self.hconv_shell_foul = Var(self.flowsheet().config.time,
                               self.shell.length_domain,
                               initialize=150.0,
                               doc='shell side convective heat transfer coefficient combined with fouling')

        # Constraint for hconv_tube_foul
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc="tube side convective heat transfer coefficient with fouling")
        def hconv_tube_foul_eqn(b, t, x):
             return 0.01*b.hconv_tube_foul[t,x]*(1+b.hconv_tube[t,x]*b.rfouling_tube) == 0.01*b.hconv_tube[t,x]

        # Constraint for hconv_shell_foul
        @self.Constraint(self.flowsheet().config.time, self.shell.length_domain, doc="shell side convective heat transfer coefficient with fouling")
        def hconv_shell_foul_eqn(b, t, x):
             return 0.1*b.hconv_shell_foul[t,x]*(1+b.hconv_shell_total[t,x]*b.rfouling_shell) == 0.1*b.hconv_shell_total[t,x]

        # Tube metal wall temperature profile across radius
        self.T = Var(self.flowsheet().config.time,
                               self.tube.length_domain, self.r,
                               initialize=500,
                               doc='tube wall temperature')

        self.temp_wall_shell = Var(self.flowsheet().config.time,
                               self.shell.length_domain,
                               initialize=500,
                               doc='shell side fouling wall surface temperature')

        # Fouling wall surface temperature on shell side
        @self.Constraint(self.flowsheet().config.time, self.shell.length_domain, doc = "Fouling wall surface temperature on shell side")
        def temp_wall_shell_eqn(b, t, x):
            return b.temp_wall_shell[t,x] == b.T[t,x,b.r.last()] + b.rfouling_shell*b.hconv_shell_foul[t,x]*(b.shell.properties[t,x].temperature-b.T[t,x,b.r.last()])

        # Declare derivatives in the model
        if self.config.dynamic is True:
            self.dTdt = DerivativeVar(self.T, wrt = self.flowsheet().config.time)
        self.dTdr = DerivativeVar(self.T, wrt = self.r)
        self.d2Tdr2 = DerivativeVar(self.T, wrt = (self.r, self.r))

        discretizer = TransformationFactory('dae.finite_difference')
        discretizer.apply_to(self, nfe=self.config.radial_elements, wrt=self.r, scheme='CENTRAL')

        # Constraint for heat conduction equation
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, self.r, doc="1-D heat conduction equation through radius")
        def heat_conduction_eqn(b, t, x, r):
            if r == b.r.first() or r == b.r.last():
                return Constraint.Skip
            if self.config.dynamic is True:
                return b.dTdt[t,x,r] == b.diff_therm_wall/b.ri_scaling**2 * (b.d2Tdr2[t,x,r] + b.dTdr[t,x,r]/r)
            else:
                return 0 == b.diff_therm_wall/b.ri_scaling**2 * (b.d2Tdr2[t,x,r] + b.dTdr[t,x,r]/r)

        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc="inner wall BC")
        def inner_wall_bc_eqn(b,t,x):
            return 0.01*b.hconv_tube_foul[t,x]*(b.tube.properties[t,x].temperature - b.T[t,x,b.r.first()]) == \
                   -0.01*b.dTdr[t,x,b.r.first()]/b.ri_scaling*b.therm_cond_wall

        @self.Constraint(self.flowsheet().config.time, self.shell.length_domain, doc="outer wall BC")
        def outer_wall_bc_eqn(b,t,x):
            return 0.01*b.hconv_shell_foul[t,x]*(b.T[t,x,b.r.last()] - b.shell.properties[t,x].temperature) == \
                   -0.01*b.dTdr[t,x,b.r.last()]/b.ri_scaling*b.therm_cond_wall

        # Inner wall BC for dTdt
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc="extra inner wall temperature derivative")
        def extra_at_inner_wall_eqn(b,t,x):
            if self.config.dynamic is True:
                term = b.dTdt[t, x, b.r.first()]
            else:
                term = 0
            return term == 4*b.diff_therm_wall*(b.r.first()+b.r[2])/ \
                (b.r[2]-b.r.first())**2/(3*b.r.first()+b.r[2])/b.ri_scaling**2*(b.T[t,x,b.r[2]] \
                -b.T[t,x,b.r.first()]) + 8*b.diff_therm_wall/b.therm_cond_wall*b.hconv_tube_foul[t,x]*b.r.first()/ \
                (b.r[2]-b.r.first())/(3*b.r.first()+b.r[2])/b.ri_scaling*(b.tube.properties[t,x].temperature \
                -b.T[t,x,b.r.first()])

        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc="extra outer wall temperature derivative")
        def extra_at_outer_wall_eqn(b,t,x):
            if self.config.dynamic is True:
                term = b.dTdt[t, x, b.r.last()]
            else:
                term = 0
            return term == 4*b.diff_therm_wall*(b.r.last()+b.r[-2])/ \
                (b.r.last()-b.r[-2])**2/(3*b.r.last()+b.r[-2])/b.ri_scaling**2*(b.T[t,x,b.r[-2]] \
                -b.T[t,x,b.r.last()]) + 8*b.diff_therm_wall/b.therm_cond_wall*b.hconv_shell_foul[t,x]*b.r.last()/ \
                (b.r.last()-b.r[-2])/(3*b.r.last()+b.r[-2])/b.ri_scaling*(b.shell.properties[t,x].temperature \
                -b.T[t,x,b.r.last()])

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
            return 0.001 * b.v_tube[t,x] * b.area_flow_tube * \
                   b.tube.properties[t,x].dens_mol_phase[self.phase_s] == 0.001 * \
                   b.tube.properties[t,x].flow_mol

        # Reynolds number
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc="Reynolds number equation on tube side")
        def N_Re_tube_eqn(b, t, x):
            return b.N_Re_tube[t,x] * b.tube.properties[t,x].visc_d_phase[self.phase_s] == \
                   b.di_tube * b.v_tube[t,x] * b.tube.properties[t,x].dens_mass_phase[self.phase_s]

        # Friction factor
        self.ff_tube_a = Var(initialize=0.25)
        self.ff_tube_b = Var(initialize=0.3164)
        self.ff_tube_a.fix()
        self.ff_tube_b.fix()
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc="Darcy friction factor on tube side")
        def friction_factor_tube_eqn(b, t, x):
            return b.friction_factor_tube[t,x]*b.N_Re_tube[t,x]**self.ff_tube_a == self.ff_tube_b*b.fcorrection_dp_tube

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
        self.Nu_tube_a = Var(initialize=0.23)
        self.Nu_tube_b = Var(initialize=0.8)
        self.Nu_tube_c = Var(initialize=0.4)
        self.Nu_tube_a.fix()
        self.Nu_tube_b.fix()
        self.Nu_tube_c.fix()
        @self.Constraint(self.flowsheet().config.time, self.tube.length_domain, doc="Nusselts number equation on tube side")
        def N_Nu_tube_eqn(b, t, x):
            return b.N_Nu_tube[t,x] == b.Nu_tube_a * b.N_Re_tube[t,x]**b.Nu_tube_b * b.N_Pr_tube[t,x]**b.Nu_tube_c

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
        self.ff_shell_a = Var()
        if self.config.tube_arrangement == "in-line":
            self.ff_shell_a.fix(0.15)
            @self.Constraint(self.flowsheet().config.time, self.shell.length_domain, doc="in-line friction factor on shell side")
            def friction_factor_shell_eqn(b, t, x):
                return b.friction_factor_shell[t,x] * b.N_Re_shell[t,x]**b.ff_shell_a == (0.044 + \
                       0.08 * b.pitch_x_to_do / (b.pitch_y_to_do - 1.0)**(0.43 + 1.13 / b.pitch_x_to_do))\
                       *b.fcorrection_dp_shell
        elif self.config.tube_arrangement == "staggered":
            self.ff_shell_a.fix(0.16)
            @self.Constraint(self.flowsheet().config.time, self.shell.length_domain,
                             doc="staggered friction factor on shell side")
            def friction_factor_shell_eqn(b, t, x):
                return b.friction_factor_shell[t,x] * b.N_Re_shell[t,x]**b.ff_shell_a == (0.25 + 0.118 / \
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
        self.Nu_shell_a = Var(initialize=0.33)
        self.Nu_shell_b = Var(initialize=0.6)
        self.Nu_shell_c = Var(initialize=1.0/3.0)
        self.Nu_shell_a.fix()
        self.Nu_shell_b.fix()
        self.Nu_shell_c.fix()
        @self.Constraint(self.flowsheet().config.time, self.shell.length_domain, doc="Nusselts number equation on shell side")
        def N_Nu_shell_eqn(b, t, x):
            return b.N_Nu_shell[t,x] == b.f_arrangement * b.Nu_shell_a * b.N_Re_shell[t,x]**b.Nu_shell_b * b.N_Pr_shell[t,x]**b.Nu_shell_c

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
            return b.heat_tube[t,x]  == b.hconv_tube_foul[t,x]*const.pi*b.di_tube*b.nrow_inlet*b.ncol_tube*\
                   (b.T[t,x,b.r.first()]-b.tube.properties[t,x].temperature)

        # Heat to wall per length on shell side
        @self.Constraint(self.flowsheet().config.time, self.shell.length_domain, doc = "heat per length on shell side")
        def heat_shell_eqn(b, t, x):
            return b.heat_shell[t,x]*b.length_flow_shell == b.length_flow_tube*b.hconv_shell_foul[t,x]*\
                   const.pi*b.do_tube*b.nrow_inlet*b.ncol_tube*\
                   (b.T[t,x,b.r.last()]-b.shell.properties[t,x].temperature)

        ### Calculate mechanical and thermal stresses
        # ----------------------------------------------------------------------------------------------

        # Integer indexing for radius domain
        self.rindex = Param(self.r, initialize=1, mutable= True, doc="inter indexing for radius domain")

        # calculate integral point for mean temperature in the wall
        @self.Expression(self.flowsheet().config.time, self.tube.length_domain, doc="integral point used to estimate mean temperature")
        def int_mean_temp(b,t,x):
            return 2*(b.r[2]-b.r[1])*b.ri_scaling**2/(b.ro_tube**2-b.ri_tube**2)*(sum(0.5*(b.r[i-1]*b.T[t,x,b.r[i-1]]+\
                b.r[i]*b.T[t,x,b.r[i]]) for i in range(2,len(b.r)+1)))

        for index_r,value_r in enumerate(self.r,1):
            self.rindex[value_r] = index_r

        @self.Expression(self.flowsheet().config.time,self.tube.length_domain, self.r, doc="integral point at each element")
        def int_temp(b,t,x,r):
            if b.rindex[r].value == 1:
                return b.T[t,x,b.r.first()]
            else:
                return 2*(b.r[2]-b.r[1])*b.ri_scaling**2/((b.r[b.rindex[r].value]*b.ri_scaling)**2-b.ri_tube**2)\
                *(sum(0.5*(b.r[j-1]*b.T[t,x,b.r[j-1]]+b.r[j]*b.T[t,x,b.r[j]])\
                 for j in range(2,b.rindex[r].value+1)))

        @self.Expression(self.flowsheet().config.time,self.tube.length_domain, self.r, doc="thermal stress at radial direction")
        def therm_stress_radial(b,t,x,r):
            if r == b.r.first() or r == b.r.last():
                return 0
            else:
                return 0.5*b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)\
                *((1-b.ri_tube**2/(r*b.ri_scaling)**2)*(b.int_mean_temp[t,x]-b.int_temp[t,x,r]))

        @self.Expression(self.flowsheet().config.time, self.tube.length_domain, self.r, doc="thermal stress at circumferential direction")
        def therm_stress_circumferential(b,t,x,r):
            r_2 = (r*b.ri_scaling)**2
            return 0.5*b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)\
                *((1+b.ri_tube**2/r_2)*b.int_mean_temp[t,x]+(1-b.ri_tube**2/r_2)*b.int_temp[t,x,r]-2*b.T[t,x,r])

        @self.Expression(self.flowsheet().config.time, self.tube.length_domain, self.r, doc="thermal stress at axial direction")
        def therm_stress_axial(b,t,x,r):
            return b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)*(b.int_mean_temp[t,x]-b.T[t,x,r])

        @self.Expression(self.flowsheet().config.time, self.tube.length_domain, self.r, doc="mechanical stress at radial direction")
        def mech_stress_radial(b,t,x,r):
            if r == b.r.first():
                return 1e-6*(-b.tube.properties[t,x].pressure)
            elif r == b.r.last():
                return 1e-6*(-b.shell.properties[t,x].pressure)
            else:
                return 0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                    /(b.ro_tube**2-b.ri_tube**2)+(1E-5*(b.shell.properties[t,x].pressure-b.tube.properties[t,x].pressure)\
                    *b.ri_tube**2*b.ro_tube**2/((r*b.ri_scaling)**2*(b.ro_tube**2-b.ri_tube**2))))

        @self.Expression(self.flowsheet().config.time, self.tube.length_domain, self.r, doc="mechanical stress at circumferential direction")
        def mech_stress_circumferential(b,t,x,r):
            return 0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                /(b.ro_tube**2-b.ri_tube**2)-(1E-5*(b.shell.properties[t,x].pressure-b.tube.properties[t,x].pressure)\
                *b.ri_tube**2*b.ro_tube**2/((r*b.ri_scaling)**2*(b.ro_tube**2-b.ri_tube**2))))

        @self.Expression(self.flowsheet().config.time, self.tube.length_domain, self.r, doc="mechanical stress at axial direction")
        def mech_stress_axial(b,t,x,r):
            return 0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                /(b.ro_tube**2-b.ri_tube**2))

        @self.Expression(self.flowsheet().config.time, self.tube.length_domain, self.r, doc="principal structural stress at radial direction")
        def prin_struc_radial(b,t,x,r):
            if r == b.r.first():
                return 1e-6*(-b.tube.properties[t,x].pressure)
            elif r == b.r.last():
                return 1e-6*(-b.shell.properties[t,x].pressure)
            else:
                return 0.5*b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)\
                    *((1-b.ri_tube**2/(r*b.ri_scaling)**2)*(b.int_mean_temp[t,x]-b.int_temp[t,x,r])) +\
                    0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                    /(b.ro_tube**2-b.ri_tube**2)+(1E-5*(b.shell.properties[t,x].pressure-b.tube.properties[t,x].pressure)\
                    *b.ri_tube**2*b.ro_tube**2/((r*b.ri_scaling)**2*(b.ro_tube**2-b.ri_tube**2))))

        @self.Expression(self.flowsheet().config.time,self.tube.length_domain, self.r, doc = "principal structural stress at tangential direction")
        def prin_struc_circumferential(b,t,x,r):
            r_2 = (r*b.ri_scaling)**2
            return 0.5*b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)\
                *((1+b.ri_tube**2/r_2)*b.int_mean_temp[t,x]+(1-b.ri_tube**2/r_2)*b.int_temp[t,x,r]-2*b.T[t,x,r]) +\
                0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                /(b.ro_tube**2-b.ri_tube**2)-(1E-5*(b.shell.properties[t,x].pressure-b.tube.properties[t,x].pressure)\
                *b.ri_tube**2*b.ro_tube**2/(r_2*(b.ro_tube**2-b.ri_tube**2))))

        @self.Expression(self.flowsheet().config.time, self.tube.length_domain, self.r, doc = "principal structural stress at axial direction")
        def prin_struc_axial(b,t,x,r):
            return b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)*(b.int_mean_temp[t,x]-b.T[t,x,r]) +\
                0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                /(b.ro_tube**2-b.ri_tube**2))

        @self.Expression(self.flowsheet().config.time, self.tube.length_domain, self.r, doc='Von Mises equivalent stress')
        def Von_Mises_equi_stress(b,t,x,r):
            r_2 = (r*b.ri_scaling)**2
            if r == b.r.first():
                prin_struc_radial = 1e-6*(-b.tube.properties[t,x].pressure)
                prin_struc_circumferential = 0.5*b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)\
                *((1+b.ri_tube**2/r_2)*b.int_mean_temp[t,x]+(1-b.ri_tube**2/r_2)*b.int_temp[t,x,r]-2*b.T[t,x,r]) +\
                0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                /(b.ro_tube**2-b.ri_tube**2)-(1E-5*(b.shell.properties[t,x].pressure-b.tube.properties[t,x].pressure)\
                *b.ri_tube**2*b.ro_tube**2/(r_2*(b.ro_tube**2-b.ri_tube**2))))
                prin_struc_axial = b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)*(b.int_mean_temp[t,x]-b.T[t,x,r]) +\
                0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                /(b.ro_tube**2-b.ri_tube**2))
                return sqrt(prin_struc_radial**2+ prin_struc_circumferential**2+prin_struc_axial**2\
                    -(prin_struc_radial*prin_struc_circumferential+prin_struc_radial*prin_struc_axial+prin_struc_circumferential*prin_struc_axial))
            elif r == b.r.last():
                prin_struc_radial = 1e-6*(-b.shell.properties[t,x].pressure)
                prin_struc_circumferential = 0.5*b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)\
                *((1+b.ri_tube**2/r_2)*b.int_mean_temp[t,x]+(1-b.ri_tube**2/r_2)*b.int_temp[t,x,r]-2*b.T[t,x,r]) +\
                0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                /(b.ro_tube**2-b.ri_tube**2)-(1E-5*(b.shell.properties[t,x].pressure-b.tube.properties[t,x].pressure)\
                *b.ri_tube**2*b.ro_tube**2/(r_2*(b.ro_tube**2-b.ri_tube**2))))
                prin_struc_axial = b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)*(b.int_mean_temp[t,x]-b.T[t,x,r]) +\
                0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                /(b.ro_tube**2-b.ri_tube**2))
                return sqrt(prin_struc_radial**2+ prin_struc_circumferential**2+prin_struc_axial**2\
                    -(prin_struc_radial*prin_struc_circumferential+prin_struc_radial*prin_struc_axial+prin_struc_circumferential*prin_struc_axial))
            else:
                prin_struc_radial = 0.5*b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)\
                    *((1-b.ri_tube**2/r_2)*(b.int_mean_temp[t,x]-b.int_temp[t,x,r])) +\
                    0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                    /(b.ro_tube**2-b.ri_tube**2)+(1E-5*(b.shell.properties[t,x].pressure-b.tube.properties[t,x].pressure)\
                    *b.ri_tube**2*b.ro_tube**2/(r_2*(b.ro_tube**2-b.ri_tube**2))))
                prin_struc_circumferential = 0.5*b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)\
                *((1+b.ri_tube**2/r_2)*b.int_mean_temp[t,x]+(1-b.ri_tube**2/r_2)*b.int_temp[t,x,r]-2*b.T[t,x,r]) +\
                0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                /(b.ro_tube**2-b.ri_tube**2)-(1E-5*(b.shell.properties[t,x].pressure-b.tube.properties[t,x].pressure)\
                *b.ri_tube**2*b.ro_tube**2/(r_2*(b.ro_tube**2-b.ri_tube**2))))
                prin_struc_axial = b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)*(b.int_mean_temp[t,x]-b.T[t,x,r]) +\
                0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                /(b.ro_tube**2-b.ri_tube**2))
                return sqrt(prin_struc_radial**2+ prin_struc_circumferential**2+prin_struc_axial**2\
                    -(prin_struc_radial*prin_struc_circumferential+prin_struc_radial*prin_struc_axial+prin_struc_circumferential*prin_struc_axial))

        #
        # Calculate creep // Using property of Steel 12CrMoV6-2-2 (Low alloy ferritic steels) , page 29/156 book eccc_data_sheet_2017_i2r002
        # data were assessed using Minimum-Commitment parameter
        self.creep_a = Param(initialize=-39.7658730, mutable=True)
        self.creep_b = Param(initialize=-8.43513298, mutable=True)
        self.creep_c = Param(initialize=-0.00186616660, mutable=True)
        self.creep_d = Param(initialize=-2.91037377E-5, mutable=True)
        self.creep_e = Param(initialize=0.00935613085, mutable=True)
        self.creep_f = Param(initialize=49662.4102, mutable=True)

        @self.Expression(self.flowsheet().config.time, self.tube.length_domain, self.r, doc='Rupture time')
        def rupture_time(b,t,x,r):
            r_2 = (r*b.ri_scaling)**2
            if r == b.r.first():
                prin_struc_radial = 1e-6*(-b.tube.properties[t,x].pressure)
                prin_struc_circumferential = 0.5*b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)\
                *((1+b.ri_tube**2/r_2)*b.int_mean_temp[t,x]+(1-b.ri_tube**2/r_2)*b.int_temp[t,x,r]-2*b.T[t,x,r]) +\
                0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                /(b.ro_tube**2-b.ri_tube**2)-(1E-5*(b.shell.properties[t,x].pressure-b.tube.properties[t,x].pressure)\
                *b.ri_tube**2*b.ro_tube**2/(r_2*(b.ro_tube**2-b.ri_tube**2))))
                prin_struc_axial = b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)*(b.int_mean_temp[t,x]-b.T[t,x,r]) +\
                0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                /(b.ro_tube**2-b.ri_tube**2))
                Von_Mises_equi_stress = sqrt(prin_struc_radial**2+ prin_struc_circumferential**2+prin_struc_axial**2\
                    -(prin_struc_radial*prin_struc_circumferential+prin_struc_radial*prin_struc_axial+prin_struc_circumferential*prin_struc_axial))
                return exp(b.creep_a + b.creep_e*b.T[t,x,r]+ b.creep_f/b.T[t,x,r] + b.creep_b*log10(Von_Mises_equi_stress) + \
                            b.creep_c*Von_Mises_equi_stress+b.creep_d*Von_Mises_equi_stress**2)
            elif r == b.r.last():
                prin_struc_radial = 1e-6*(-b.shell.properties[t,x].pressure)
                prin_struc_circumferential = 0.5*b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)\
                *((1+b.ri_tube**2/r_2)*b.int_mean_temp[t,x]+(1-b.ri_tube**2/r_2)*b.int_temp[t,x,r]-2*b.T[t,x,r]) +\
                0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                /(b.ro_tube**2-b.ri_tube**2)-(1E-5*(b.shell.properties[t,x].pressure-b.tube.properties[t,x].pressure)\
                *b.ri_tube**2*b.ro_tube**2/(r_2*(b.ro_tube**2-b.ri_tube**2))))
                prin_struc_axial = b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)*(b.int_mean_temp[t,x]-b.T[t,x,r]) +\
                0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                /(b.ro_tube**2-b.ri_tube**2))
                Von_Mises_equi_stress = sqrt(prin_struc_radial**2+ prin_struc_circumferential**2+prin_struc_axial**2\
                    -(prin_struc_radial*prin_struc_circumferential+prin_struc_radial*prin_struc_axial+prin_struc_circumferential*prin_struc_axial))
                return exp(b.creep_a + b.creep_e*b.T[t,x,r]+ b.creep_f/b.T[t,x,r] + b.creep_b*log10(Von_Mises_equi_stress) + \
                            b.creep_c*Von_Mises_equi_stress+b.creep_d*Von_Mises_equi_stress**2)
            else:
                prin_struc_radial = 0.5*b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)\
                    *((1-b.ri_tube**2/r_2)*(b.int_mean_temp[t,x]-b.int_temp[t,x,r])) +\
                    0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                    /(b.ro_tube**2-b.ri_tube**2)+(1E-5*(b.shell.properties[t,x].pressure-b.tube.properties[t,x].pressure)\
                    *b.ri_tube**2*b.ro_tube**2/(r_2*(b.ro_tube**2-b.ri_tube**2))))
                prin_struc_circumferential = 0.5*b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)\
                *((1+b.ri_tube**2/r_2)*b.int_mean_temp[t,x]+(1-b.ri_tube**2/r_2)*b.int_temp[t,x,r]-2*b.T[t,x,r]) +\
                0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                /(b.ro_tube**2-b.ri_tube**2)-(1E-5*(b.shell.properties[t,x].pressure-b.tube.properties[t,x].pressure)\
                *b.ri_tube**2*b.ro_tube**2/(r_2*(b.ro_tube**2-b.ri_tube**2))))
                prin_struc_axial = b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)*(b.int_mean_temp[t,x]-b.T[t,x,r]) +\
                0.1*(1E-5*(b.tube.properties[t,x].pressure*b.ri_tube**2-b.shell.properties[t,x].pressure*b.ro_tube**2)\
                /(b.ro_tube**2-b.ri_tube**2))
                Von_Mises_equi_stress = sqrt(prin_struc_radial**2+ prin_struc_circumferential**2+prin_struc_axial**2\
                    -(prin_struc_radial*prin_struc_circumferential+prin_struc_radial*prin_struc_axial+prin_struc_circumferential*prin_struc_axial))
                return exp(b.creep_a + b.creep_e*b.T[t,x,r]+ b.creep_f/b.T[t,x,r] + b.creep_b*log10(Von_Mises_equi_stress) + \
                            b.creep_c*Von_Mises_equi_stress+b.creep_d*Von_Mises_equi_stress**2)


        # -----------------------------------------------------------------------------------------------------------------------------------

        # total heat released by shell side fluid assuming even discretization. shell side always in forward direction and the first point is skiped
        @self.Expression(self.flowsheet().config.time, doc = "Total heat released from shell side")
        def total_heat(b, t):
            return -(sum(b.heat_shell[t,x] for x in b.shell.length_domain)-b.heat_shell[t,b.shell.length_domain.first()])*\
            b.length_flow_shell/b.config.finite_elements

    def set_initial_condition(self):
        if self.config.dynamic is True:
            t0 = self.flowsheet().config.time.first()
            self.dTdt[:,:,:].value = 0
            self.dTdt[t0,:,:].fix(0)
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
            mcp_shell = value(sum(blk.shell_inlet.flow_component[0,j] \
                   for j in blk.shell.properties[0,0].component_list) * blk.shell.properties[0,0].heat_cap)
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
                    blk.T[t, z, :].fix(value(
                          0.05*((1-z)*blk.shell.properties[0, 0].temperature + z*temp_out_shell_guess) +
                          0.95*((1-z)*blk.tube.properties[0, 0].temperature + z*temp_out_tube_guess)))
                else:
                    blk.T[t, z, :].fix(value(
                          0.05*((1-z)*blk.shell.properties[0, 0].temperature + z*temp_out_shell_guess) +
                          0.95*((1-z)*temp_out_tube_guess + z*blk.tube.properties[0, 1].temperature)))

        for t in blk.flowsheet().config.time:
            for z in blk.tube.length_domain:
                blk.tube.properties[t,z].enth_mol.fix(value(blk.tube.properties[t,0].enth_mol))
                blk.tube.properties[t,z].pressure.fix(value(blk.tube.properties[t,0].pressure))

        for t in blk.flowsheet().config.time:
            for z in blk.shell.length_domain:
                blk.shell.properties[t,z].temperature.fix(value(blk.shell.properties[t,0].temperature))
                blk.shell.properties[t,z].pressure.fix(value(blk.shell.properties[t,0].pressure))

        blk.heat_conduction_eqn.deactivate()
        blk.inner_wall_bc_eqn.deactivate()
        blk.outer_wall_bc_eqn.deactivate()
        blk.extra_at_inner_wall_eqn.deactivate()
        blk.extra_at_outer_wall_eqn.deactivate()
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

        blk.T[:,:,:].unfix()
        blk.heat_conduction_eqn.activate()
        blk.inner_wall_bc_eqn.activate()
        blk.outer_wall_bc_eqn.activate()
        blk.extra_at_inner_wall_eqn.activate()
        blk.extra_at_outer_wall_eqn.activate()

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

        for i, c in self.tube.pressure_dx_disc_eq.items():
            sf = iscale.get_scaling_factor(
                self.tube.properties[i].pressure, default=1, warning=True)
            iscale.constraint_scaling_transform(c, sf)

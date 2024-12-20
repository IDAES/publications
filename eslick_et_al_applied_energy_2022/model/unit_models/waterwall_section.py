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
unit operation model for a section of waterwall
main equations:
    Heat is given by fire-side boiler model
    Calculate pressure change due to friction and gravity
    Calculate slag layer wall temperature
    Consider a layer of metal and a layer of slag

Created: February 2019 by Jinliang Ma
@JinliangMa: jinliang.ma@netl.doe.gov
"""
# Import Python libraries
import math

# Import Pyomo libraries
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.core.expr.current import Expr_if

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
import idaes.logger as idaeslog


# Additional import for the unit operation
from pyomo.environ import SolverFactory, value, Var, Param, asin, cos, sqrt, log10
from pyomo.opt import TerminationCondition
from pyomo.dae import DerivativeVar


__author__ = "Jinliang Ma <jinliang.ma.doe.gov>"
__version__ = "2.0.0"


@declare_process_block_class("WaterwallSection")
class WaterwallSectionData(UnitModelBlockData):
    """
    WaterwallSection Unit Class
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
    CONFIG.declare("has_equilibrium_reactions", ConfigValue(
        default=True,
        domain=In([True, False]),
        description="Equilibrium reaction construction flag",
        doc="""Indicates whether terms for equilibrium controlled reactions
should be constructed,
**default** - True.
**Valid values:** {
**True** - include equilibrium reaction terms,
**False** - exclude equilibrium reaction terms.}"""))
    CONFIG.declare("has_heat_of_reaction", ConfigValue(
        default=False,
        domain=In([True, False]),
        description="Heat of reaction term construction flag",
        doc="""Indicates whether terms for heat of reaction terms should be
constructed,
**default** - False.
**Valid values:** {
**True** - include heat of reaction terms,
**False** - exclude heat of reaction terms.}"""))
    CONFIG.declare("property_package", ConfigValue(
        default=useDefault,
        domain=is_physical_parameter_block,
        description="Property package to use for control volume",
        doc="""Property parameter object used to define property calculations,
**default** - useDefault.
**Valid values:** {
**useDefault** - use default package from parent model or flowsheet,
**PhysicalParameterObject** - a PhysicalParameterBlock object.}"""))
    CONFIG.declare("property_package_args", ConfigBlock(
        implicit=True,
        description="Arguments to use for constructing property packages",
        doc="""A ConfigBlock with arguments to be passed to a property block(s)
and used when constructing these,
**default** - None.
**Valid values:** {
see property package for documentation.}"""))
    CONFIG.declare("reaction_package", ConfigValue(
        default=None,
        domain=is_reaction_parameter_block,
        description="Reaction package to use for control volume",
        doc="""Reaction parameter object used to define reaction calculations,
**default** - None.
**Valid values:** {
**None** - no reaction package,
**ReactionParameterBlock** - a ReactionParameterBlock object.}"""))
    CONFIG.declare("reaction_package_args", ConfigBlock(
        implicit=True,
        description="Arguments to use for constructing reaction packages",
        doc="""A ConfigBlock with arguments to be passed to a reaction block(s)
and used when constructing these,
**default** - None.
**Valid values:** {
see reaction package for documentation.}"""))

    def build(self):
        """
        Begin building model (pre-DAE transformation)


        Args:
            None

        Returns:
            None
        """
        # Call UnitModel.build to setup dynamics
        super(WaterwallSectionData, self).build()

        # Build Control Volume
        self.control_volume = ControlVolume0DBlock(default={
                "dynamic": self.config.dynamic,
                "has_holdup": self.config.has_holdup,
                "property_package": self.config.property_package,
                "property_package_args": self.config.property_package_args})

        self.control_volume.add_geometry()

        self.control_volume.add_state_blocks(has_phase_equilibrium=False)

        self.control_volume.add_material_balances(
            balance_type=self.config.material_balance_type,
            has_rate_reactions=False,
            has_equilibrium_reactions=False)

        self.control_volume.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_heat_of_reaction=False,
            has_heat_transfer=self.config.has_heat_transfer)

        self.control_volume.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=True)

        # Add Ports
        self.add_inlet_port()
        self.add_outlet_port()

        # Add object references
        add_object_reference(self,
                             "volume",
                             self.control_volume.volume)

        # Set references to balance terms at unit level
        if (self.config.has_heat_transfer is True and
                self.config.energy_balance_type != EnergyBalanceType.none):
            add_object_reference(self, "heat_duty", self.control_volume.heat)

        if (self.config.has_pressure_change is True and
                self.config.momentum_balance_type != 'none'):
            add_object_reference(self, "deltaP", self.control_volume.deltaP)

        # Set Unit Geometry and Holdup Volume
        self._set_geometry()

        # Construct performance equations
        self._make_performance()


    def _set_geometry(self):
        """
        Define the geometry of the unit as necessary, and link to holdup volume
        """

        # Constant pi
        self.pi = Param(
                initialize=math.pi,
                doc="Pi")
        # Constant gravity
        self.gravity = Param(
                initialize=9.807,
                doc="Gravity")
        # Total projected wall area of waterwall section from fire-side model
        self.area_proj_total = Var(
                initialize=10,
                doc="Total projected wall area of waterwall section")
        # Number of waterwall tubes
        self.count = Var(
                initialize=4,
                doc="Number of waterwall tubes")
        # Height of waterwall section, given by boiler model
        self.height = Var(
                initialize=5.0,
                doc="Height of waterwall section")
        # Length of waterwall tubes calculated based on given total area and perimeter of waterwall
        self.length_tube = Var(
                initialize=5.0,
                doc="length waterwall tube")
       # Inside diameter of waterwall tubes
        self.diameter_in = Var(
                initialize=0.05,
                doc="Inside diameter of waterwall tube")
        # Inside radius of waterwall tube
        @self.Expression(doc="Inside radius of waterwall tube")
        def radius_in(b):
            return 0.5*b.diameter_in
        # Total cross section area of fluid flow
        @self.Expression(doc="Cross section area of fluid")
        def area_cross_fluid_total(b):
            return 0.25*b.pi*b.diameter_in**2*b.count
        # Tube thickness
        self.thk_tube = Var(
                initialize=0.005,
                doc="Thickness of waterwall tube")
        # Outside radius of waterwall tube
        @self.Expression(doc="Outside radius of waterwall tube")
        def radius_out(b):
            return b.radius_in + b.thk_tube
        # Thickness of waterwall fin
        self.thk_fin = Var(
                initialize=0.004,
                doc="Thickness of waterwall fin")
        # Half of the waterwall fin thickness
        @self.Expression(doc="Half of the waterwall fin thickness")
        def thk_fin_half(b):
            return 0.5*b.thk_fin
        # Length of waterwall fin
        self.length_fin = Var(
                initialize=0.005,
                doc="Length of waterwall fin")
        # Thickness of slag layer
        self.thk_slag = Var(
                self.flowsheet().config.time,
                bounds=(0.0001, 0.009),
                initialize=0.001,
                doc="Thickness of slag layer")
        @self.Expression(doc="Pitch of two neighboring tubes")
        def pitch(b):
            return b.length_fin + b.radius_out*2.0
        # Equivalent tube length (not neccesarily equal to height)
        @self.Constraint(doc="Equivalent length of tube")
        def length_tube_eqn(b):
            return b.length_tube * b.pitch * b.count == b.area_proj_total
        @self.Expression(doc="Angle at joint of tube and fin")
        def alpha_tube(b):
            return asin(b.thk_fin_half/b.radius_out)
        @self.Expression(self.flowsheet().config.time, doc="Angle at joint of tube and fin at outside slag layer")
        def alpha_slag(b, t):
            return asin((b.thk_fin_half+b.thk_slag[t])/(b.radius_out+b.thk_slag[t]))
        @self.Expression(doc="Perimeter of interface between slag and tube")
        def perimeter_if(b):
            return (b.pi-2*b.alpha_tube)*b.radius_out + b.pitch - 2*b.radius_out*cos(b.alpha_tube)
        @self.Expression(doc="Perimeter on the inner tube side")
        def perimeter_ts(b):
            return b.pi*b.diameter_in
        @self.Expression(self.flowsheet().config.time, doc="Perimeter on the outer slag side")
        def perimeter_ss(b, t):
            return (b.pi-2*b.alpha_slag[t])*(b.radius_out+b.thk_slag[t]) + \
                   b.pitch - 2*(b.radius_out+b.thk_slag[t])*cos(b.alpha_slag[t])
        # Cross section area of tube and fin metal
        @self.Expression(doc="Cross section area of tube and fin metal")
        def area_cross_metal(b):
            return b.pi*(b.radius_out**2-b.radius_in**2) + b.thk_fin*b.length_fin
        # Cross section area of slag layer
        @self.Expression(self.flowsheet().config.time, doc="Cross section area of slag layer per tube")
        def area_cross_slag(b, t):
            return b.perimeter_if*b.thk_slag[t]
        # Volume constraint
        @self.Constraint(self.flowsheet().config.time, doc="waterwall fluid volume of all tubes")
        def volume_eqn(b, t):
            return b.volume[t] == 0.25*b.pi*b.diameter_in**2*b.length_tube*b.count


    def _make_performance(self):
        """
        Define constraints which describe the behaviour of the unit model.
        """
        # Thermal conductivity of metal
        self.fcorrection_dp = Param(
                initialize=1.2,
                mutable=True,
                doc='correction factor for pressure drop due to acceleration and unsmooth tube applied to friction term')
        # Thermal conductivity of metal
        self.therm_cond_metal = Param(
                initialize=43.0,
                mutable=True,
                doc='Thermal conductivity of tube metal')
        # Thermal conductivity of slag
        self.therm_cond_slag = Param(
                initialize=1.3,
                mutable=True,
                doc='Thermal conductivity of slag')
        # Heat capacity of metal
        self.cp_metal = Param(
                initialize=500.0,
                mutable=True,
                doc='Heat capacity of tube metal')
        # Heat Capacity of slag
        self.cp_slag = Param(
                initialize=250,
                mutable=True,
                doc='Heat capacity of slag')
        # Density of metal
        self.dens_metal = Param(
                initialize=7800.0,
                mutable=True,
                doc='Density of tube metal')
        # Density of slag
        self.dens_slag = Param(
                initialize=2550,
                mutable=True,
                doc='Density of slag')
        # Shape factor of tube metal conduction resistance based on projected area
        self.fshape_metal = Param(
                initialize=0.7718,
                mutable=True,
                doc='Shape factor of tube metal conduction')
        # Shape factor of slag conduction resistance based on projected area
        self.fshape_slag = Param(
                initialize=0.6858,
                mutable=True,
                doc='Shape factor of slag conduction')
        # Shape factor of convection on tube side resistance based on projected area
        self.fshape_conv = Param(
                initialize=0.8496,
                mutable=True,
                doc='Shape factor of convection')

        # Heat conduction resistance of half of metal wall thickness based on interface perimeter
        @self.Expression(doc="half metal layer conduction resistance")
        def half_resistance_metal(b):
            return b.thk_tube/2/b.therm_cond_metal*b.fshape_metal*b.perimeter_if/b.pitch

        # Heat conduction resistance of half of slag thickness based on mid slag layer perimeter
        @self.Expression(self.flowsheet().config.time, doc="half slag layer conduction resistance")
        def half_resistance_slag(b, t):
            return b.thk_slag[t]/2/b.therm_cond_slag*b.fshape_slag*(b.perimeter_ss[t]+b.perimeter_if)/2/b.pitch

        # Add performance variables
        # Heat from fire side boiler model
        self.heat_fireside = Var(
                self.flowsheet().config.time,
                initialize=1e7,
                doc='total heat from fire side model for the section')
        # Tube boundary wall temperature
        self.temp_tube_boundary = Var(
                self.flowsheet().config.time,
                initialize=400.0,
                doc='Temperature of tube boundary wall')
        # Tube center point wall temperature
        self.temp_tube_center = Var(
                self.flowsheet().config.time,
                initialize=450.0,
                doc='Temperature of tube center wall')
        # Slag boundary wall temperature
        self.temp_slag_boundary = Var(
                self.flowsheet().config.time,
                initialize=600.0,
                doc='Temperature of slag boundary wall')
        # Slag center point wall temperature
        self.temp_slag_center = Var(
                self.flowsheet().config.time,
                initialize=500.0,
                doc='Temperature of slag layer center point')

        # Energy holdup for slag layer
        self.energy_holdup_slag = Var(
                self.flowsheet().config.time,
                initialize=1e4,
                doc='Energy holdup of slag layer')

        # Energy holdup for metal (tube + fin)
        self.energy_holdup_metal = Var(
                self.flowsheet().config.time,
                initialize=1e6,
                doc='Energy holdup of metal')

        # Energy accumulation for slag and metal
        if self.config.dynamic is True:
                self.energy_accumulation_slag = DerivativeVar(
                        self.energy_holdup_slag,
                        wrt=self.flowsheet().config.time,
                        doc='Energy accumulation of slag layer')
                self.energy_accumulation_metal = DerivativeVar(
                        self.energy_holdup_metal,
                        wrt=self.flowsheet().config.time,
                        doc='Energy accumulation of tube and fin metal')

        def energy_accumulation_term_slag(b, t):
            return b.energy_accumulation_slag[t] if b.config.dynamic else 0

        def energy_accumulation_term_metal(b, t):
            return b.energy_accumulation_metal[t] if b.config.dynamic else 0

        # Velocity of liquid only
        self.velocity_lo = Var(
                self.flowsheet().config.time,
                initialize=3.0,
                doc='Velocity of liquid only')

        # Reynolds number based on liquid only flow
        self.N_Re = Var(
                self.flowsheet().config.time,
                initialize=1.0e6,
                doc='Reynolds number')

        # Prandtl number of liquid phase
        self.N_Pr = Var(
                self.flowsheet().config.time,
                initialize=2.0,
                doc='Reynolds number')

        # Darcy friction factor
        self.friction_factor_darcy = Var(
                self.flowsheet().config.time,
                initialize=0.01,
                doc='Darcy friction factor')

        # Vapor fraction at inlet
        self.vapor_fraction = Var(
                self.flowsheet().config.time,
                initialize=0.0,
                doc='Vapor fractoin of vapor-liquid mixture')

        # Liquid fraction at inlet
        self.liquid_fraction = Var(
                self.flowsheet().config.time,
                initialize=1.0,
                doc='Liquid fractoin of vapor-liquid mixture')

        # Density ratio of liquid to vapor
        self.ratio_density = Var(
                self.flowsheet().config.time,
                initialize=1.0,
                doc='Liquid to vapor density ratio')

        # Void fraction at inlet
        self.void_fraction = Var(
                self.flowsheet().config.time,
                initialize=0.0,
                doc='void fraction at inlet')

        # Exponent n for gamma at inlet, typical range in (0.75,0.8294)
        self.n_exp = Var(
                self.flowsheet().config.time,
                initialize=0.82,
                doc='exponent for gamma at inlet')

        # Gamma for velocity slip at inlet
        self.gamma = Var(
                self.flowsheet().config.time,
                initialize=3.0,
                doc='gamma for velocity slip at inlet')

        # Mass flux
        self.mass_flux = Var(
                self.flowsheet().config.time,
                initialize=1000.0,
                doc='mass flux')

        # Reduced pressure
        self.reduced_pressure = Var(
                self.flowsheet().config.time,
                initialize=0.85,
                doc='reduced pressure')

        # Two-phase correction factor
        self.phi_correction = Var(
                self.flowsheet().config.time,
                initialize=1.01,
                doc='Two-phase flow correction factor')

        # Convective heat transfer coefficient on tube side, typically in range (1000, 5e5)
        self.hconv = Var(
                self.flowsheet().config.time,
                initialize=30000.0,
                doc='Convective heat transfer coefficient')

        # Convective heat transfer coefficient for liquid only, typically in range (1000.0, 1e5)
        self.hconv_lo = Var(
                self.flowsheet().config.time,
                initialize=20000.0,
                doc='Convective heat transfer coefficient of liquid only')

        # Pool boiling heat transfer coefficient, typically in range (1e4, 5e5)
        self.hpool = Var(
                self.flowsheet().config.time,
                initialize=1e5,
                doc='Pool boiling heat transfer coefficient')
        '''
        # Reciprocal of Martinelli parameter to the power of 0.86, typical range in (1e-3, 1.0)
        self.martinelli_reciprocal_p86 = Var(
                self.flowsheet().config.time,
                initialize=0.2,
                doc='Reciprocal of Martinelli parameter to the power of 0.86')
        '''
        # Boiling number, typical range in (1e-7, 5e-4) in original formula.  we define here as boiling_number_scaled == 1e6*boiling_number
        self.boiling_number_scaled = Var(
                self.flowsheet().config.time,
                initialize=1,
                doc='Scaled boiling number')

        # Enhancement factor, typical range in (1.0, 3.0)
        self.enhancement_factor = Var(
                self.flowsheet().config.time,
                initialize=1.3,
                doc='Enhancement factor')

        # Suppression factor, typical range in (0.005, 1.0)
        self.suppression_factor = Var(
                self.flowsheet().config.time,
                initialize=0.03,
                doc='Suppression factor')

        # Convective heat flux to fluid
        self.heat_flux_conv = Var(
                self.flowsheet().config.time,
                initialize=7e4,
                doc='Convective heat flux to fluid')

        # Slag-tube interface heat flux
        self.heat_flux_interface = Var(
                self.flowsheet().config.time,
                initialize=100000.0,
                doc='Slag-tube interface heat flux')

        # Pressure change due to friction
        self.deltaP_friction = Var(
                self.flowsheet().config.time,
                initialize=-1000.0,
                doc='Pressure change due to friction')

        # Pressure change due to gravity
        self.deltaP_gravity = Var(
                self.flowsheet().config.time,
                initialize=-1000.0,
                doc='Pressure change due to gravity')

        # Equation to calculate heat flux to slag boundary
        @self.Expression(self.flowsheet().config.time, doc="heat flux at slag outer layer")
        def heat_flux_fireside(b, t):
            return (b.heat_fireside[t] * b.pitch/
                (b.area_proj_total * b.perimeter_ss[t]))

        # Equation to calculate slag layer boundary temperature
        @self.Constraint(self.flowsheet().config.time, doc="slag layer boundary temperature")
        def slag_layer_boundary_temperature_eqn(b, t):
            return b.heat_flux_fireside[t] * b.half_resistance_slag[t] == \
                   (b.temp_slag_boundary[t] - b.temp_slag_center[t])

        # Equation to calculate heat flux at the slag-metal interface
        @self.Constraint(self.flowsheet().config.time, doc="heat flux at slag-tube interface")
        def heat_flux_interface_eqn(b, t):
            return b.heat_flux_interface[t] * (b.half_resistance_metal + b.half_resistance_slag[t]) == \
                   (b.temp_slag_center[t] - b.temp_tube_center[t])

        # Equation to calculate heat flux at tube boundary
        @self.Constraint(self.flowsheet().config.time, doc="convective heat flux at tube boundary")
        def heat_flux_conv_eqn(b, t):
            return b.heat_flux_conv[t] * b.fshape_conv * b.perimeter_ts == b.pitch * \
                   b.hconv[t] * (b.temp_tube_boundary[t] - b.control_volume.properties_in[t].temperature)

        # Equation to calculate tube boundary wall temperature
        @self.Constraint(self.flowsheet().config.time, doc="tube bounary wall temperature")
        def temperature_tube_boundary_eqn(b, t):
            return b.heat_flux_conv[t] * b.half_resistance_metal * b.perimeter_ts == \
                   b.perimeter_if * (b.temp_tube_center[t] - b.temp_tube_boundary[t])

        # Equation to calculate energy holdup for slag layer per tube length
        @self.Constraint(self.flowsheet().config.time, doc="energy holdup for slag layer")
        def energy_holdup_slag_eqn(b, t):
            return b.energy_holdup_slag[t] == \
                   b.temp_slag_center[t]*b.cp_slag*b.dens_slag*b.area_cross_slag[t]

        # Equation to calculate energy holdup for metal (tube + fin) per tube length
        @self.Constraint(self.flowsheet().config.time, doc="energy holdup for metal")
        def energy_holdup_metal_eqn(b, t):
            return b.energy_holdup_metal[t] == \
                   b.temp_tube_center[t]*b.cp_metal*b.dens_metal*b.area_cross_metal

        # Energy balance for slag layer
        @self.Constraint(self.flowsheet().config.time, doc="energy balance for slag layer")
        def energy_balance_slag_eqn(b, t):
            return energy_accumulation_term_slag(b,t) == \
                   b.heat_flux_fireside[t]*b.perimeter_ss[t] - \
                   b.heat_flux_interface[t]*b.perimeter_if

        # Energy balance for metal
        @self.Constraint(self.flowsheet().config.time, doc="energy balance for metal")
        def energy_balance_metal_eqn(b, t):
            return energy_accumulation_term_metal(b,t) == \
                   b.heat_flux_interface[t]*b.perimeter_if - \
                   b.heat_flux_conv[t]*b.perimeter_ts

        # Expression to calculate slag/tube metal interface wall temperature
        @self.Expression(self.flowsheet().config.time, doc="Slag tube interface wall temperature")
        def temp_interface(b, t):
            return b.temp_tube_center[t] + b.heat_flux_interface[t]*b.half_resistance_metal

        # Equations for calculate pressure drop and convective heat transfer coefficient for 2-phase flow
        # Equation to calculate liquid to vapor density ratio
        @self.Constraint(self.flowsheet().config.time, doc="liquid to vapor density ratio")
        def ratio_density_eqn(b, t):
            return 0.001*b.ratio_density[t] * b.control_volume.properties_in[t].dens_mol_phase["Vap"] == \
                   0.001*b.control_volume.properties_in[t].dens_mol_phase["Liq"]

        # Equation for calculating velocity if the flow is liquid only
        @self.Constraint(self.flowsheet().config.time, doc="Vecolity of fluid if liquid only")
        def velocity_lo_eqn(b, t):
            return 1e-4*b.velocity_lo[t]*b.area_cross_fluid_total * \
                   b.control_volume.properties_in[t].dens_mol_phase["Liq"] \
                   == 1e-4*b.control_volume.properties_in[t].flow_mol

        # Equation for calculating Reynolds number if liquid only
        @self.Constraint(self.flowsheet().config.time, doc="Reynolds number if liquid only")
        def Reynolds_number_eqn(b, t):
            return b.N_Re[t] * \
                   b.control_volume.properties_in[t].visc_d_phase["Liq"] == \
                   b.diameter_in * b.velocity_lo[t] * \
                   b.control_volume.properties_in[t].dens_mass_phase["Liq"]

        # Friction factor depending on laminar or turbulent flow, usually always turbulent (>1187.385)
        @self.Constraint(self.flowsheet().config.time, doc="Darcy friction factor")
        def friction_factor_darcy_eqn(b, t):
            return b.friction_factor_darcy[t]*b.N_Re[t]**0.25/0.3164 == 1.0

        # Vapor fractoin equation at inlet, add 1e-5 such that vapor fraction is always positive
        @self.Constraint(self.flowsheet().config.time, doc="Vapor fractoin at inlet")
        def vapor_fraction_eqn(b, t):
            return 100*b.vapor_fraction[t] == 100*(b.control_volume.properties_in[t].vapor_frac + 1e-5)

        # n-exponent equation for inlet
        @self.Constraint(self.flowsheet().config.time, doc="n-exponent")
        def n_exp_eqn(b, t):
            return 0.001*(0.8294 - b.n_exp[t]) * b.control_volume.properties_in[t].pressure == 8.0478

        # Gamma equation for inlet
        @self.Constraint(self.flowsheet().config.time, doc="Gamma at inlet")
        def gamma_eqn(b, t):
            return b.gamma[t] == b.ratio_density[t]**b.n_exp[t]

        # void faction at inlet equation
        @self.Constraint(self.flowsheet().config.time, doc="Void fractoin at inlet")
        def void_fraction_eqn(b, t):
            return b.void_fraction[t] * (1.0 + \
                   b.vapor_fraction[t] * (b.gamma[t] - 1.0)) == \
                   b.vapor_fraction[t] * b.gamma[t]

        # Two-phase flow correction factor equation
        @self.Constraint(self.flowsheet().config.time, doc="Correction factor")
        def correction_factor_eqn(b, t):
            return (b.phi_correction[t] - 0.027*b.liquid_fraction[t])**2 == \
                   (0.973*b.liquid_fraction[t] + b.vapor_fraction[t] * b.ratio_density[t]) * \
                   (0.973*b.liquid_fraction[t] + b.vapor_fraction[t])

        # Pressure change equation due to friction, -1/2*density*velocity^2*fD/diameter*length*phi^2
        @self.Constraint(self.flowsheet().config.time, doc="pressure change due to friction")
        def pressure_change_friction_eqn(b, t):
            return 0.01* b.deltaP_friction[t] * b.diameter_in == \
                   -0.01 * b.fcorrection_dp * 0.5 * b.control_volume.properties_in[t].dens_mass_phase["Liq"] * \
                   b.velocity_lo[t]**2 * b.friction_factor_darcy[t] * b.length_tube * \
                   b.phi_correction[t]**2

        # Pressure change equation due to gravity, density_mixture*gravity*height
        @self.Constraint(self.flowsheet().config.time, doc="pressure change due to gravity")
        def pressure_change_gravity_eqn(b, t):
            return 1e-3 * b.deltaP_gravity[t] == -1e-3 * b.gravity * b.height * \
                   (b.control_volume.properties_in[t].dens_mass_phase["Vap"] * b.void_fraction[t] + \
                   b.control_volume.properties_in[t].dens_mass_phase["Liq"] * (1.0 - b.void_fraction[t]))

        # Mass flux of vapor-liquid mixture (density*velocity or mass_flow/area)
        @self.Constraint(self.flowsheet().config.time, doc="mass flux")
        def mass_flux_eqn(b, t):
            return b.mass_flux[t] * b.area_cross_fluid_total == \
                   b.control_volume.properties_in[t].flow_mol * \
                   b.control_volume.properties_in[0].mw

        # Liquid fraction at inlet
        @self.Constraint(self.flowsheet().config.time, doc="liquid fraction")
        def liquid_fraction_eqn(b, t):
            return b.liquid_fraction[t] + b.vapor_fraction[t] == 1.0

        # Total pressure change equation
        @self.Constraint(self.flowsheet().config.time, doc="pressure drop")
        def pressure_change_total_eqn(b, t):
            return b.deltaP[t] == b.deltaP_friction[t] + b.deltaP_gravity[t]

        # Total heat added to control_volume
        @self.Constraint(self.flowsheet().config.time, doc="total heat added to fluid control_volume")
        def heat_eqn(b, t):
            return b.heat_duty[t] == b.count * b.heat_flux_conv[t] * b.length_tube * b.perimeter_ts

        # Reduced pressure
        @self.Constraint(self.flowsheet().config.time, doc="reduced pressure")
        def reduced_pressure_eqn(b, t):
            return b.reduced_pressure[t] * self.config.property_package.pressure_crit == \
                   b.control_volume.properties_in[t].pressure

        # Prandtl number of liquid
        @self.Constraint(self.flowsheet().config.time, doc="liquid Prandtl number")
        def N_Pr_eqn(b, t):
            return b.N_Pr[t]*b.control_volume.properties_in[t].therm_cond_phase["Liq"] \
                   * b.control_volume.properties_in[0].mw == \
                   b.control_volume.properties_in[t].cp_mol_phase["Liq"] * \
                   b.control_volume.properties_in[t].visc_d_phase["Liq"]

        # Forced convection heat transfer coefficient for liquid only
        @self.Constraint(self.flowsheet().config.time, doc="forced convection heat transfer coefficient for liquid only")
        def hconv_lo_eqn(b, t):
            return b.hconv_lo[t] * b.diameter_in == 0.023 * b.N_Re[t]**0.8 * b.N_Pr[t]**0.4 * \
                   b.control_volume.properties_in[t].therm_cond_phase["Liq"]

        # Pool boiling heat transfer coefficient
        @self.Constraint(self.flowsheet().config.time, doc="pool boiling heat transfer coefficient")
        def hpool_eqn(b, t):
            return 1e-4*b.hpool[t] * sqrt(b.control_volume.properties_in[0].mw*1000.0) * \
                   (-log10(b.reduced_pressure[t]))**(0.55) == 1e-4*\
                   55.0 * b.reduced_pressure[t]**0.12 * b.heat_flux_conv[t]**0.67

        # Boiling number scaled by a factor of 1e6
        @self.Constraint(self.flowsheet().config.time, doc="boiling number")
        def boiling_number_eqn(b, t):
            return 1e-10*b.boiling_number_scaled[t] * b.control_volume.properties_in[t].dh_vap_mol * b.mass_flux[t] == \
                   b.heat_flux_conv[t]*b.control_volume.properties_in[0].mw*1e-4
        '''
        # Reciprocal of Martinelli parameter to the power of 0.86
        @self.Constraint(self.flowsheet().config.time, doc="Reciprocal of Martineli parameter to the power of 0.86")
        def martinelli_reciprocal_p86_eqn(b, t):
            return b.martinelli_reciprocal_p86[t]*b.liquid_fraction[t]**0.774 * \
                   b.control_volume.properties_in[t].visc_d_phase["Liq"]**0.086 == \
                   Expr_if(b.vapor_fraction[t]>0, \
                   (b.vapor_fraction[t])**0.774 * b.ratio_density[t]**0.43 * \
                   b.control_volume.properties_in[t].visc_d_phase["Vap"]**0.086, 0.0)
        '''

        # Enhancement factor
        @self.Constraint(self.flowsheet().config.time, doc="Forced convection enhancement factor")
        def enhancement_factor_eqn(b, t):
            return b.enhancement_factor[t] == 1.0 + 24000.0*(b.boiling_number_scaled[t]/1e6)**1.16 \
                   #+ 1.37*b.martinelli_reciprocal_p86[t]

        # Suppression factor
        @self.Constraint(self.flowsheet().config.time, doc="Pool boiler suppression factor")
        def suppression_factor_eqn(b, t):
            return b.suppression_factor[t] * (1.0 + 1.15e-6*b.enhancement_factor[t]**2*b.N_Re[t]**1.17) == 1.0

        @self.Constraint(self.flowsheet().config.time, doc="convective heat transfer coefficient")
        def hconv_eqn(b, t):
            return 1e-3*b.hconv[t] == 1e-3*b.hconv_lo[t]*b.enhancement_factor[t] + 1e-3*b.hpool[t]*b.suppression_factor[t]

    def set_initial_condition(self):
        ''' currently ignore hold up for convegence issue '''

        if self.config.dynamic is True:
            self.control_volume.material_accumulation[:,:,:].value = 0
            self.control_volume.energy_accumulation[:,:].value = 0
            self.control_volume.material_accumulation[0,:,:].fix(0)
            self.control_volume.energy_accumulation[0,:].fix(0)

        if self.config.dynamic is True:
            self.energy_accumulation_slag[:].value = 0
            self.energy_accumulation_metal[:].value = 0
            self.energy_accumulation_slag[0].fix(0)
            self.energy_accumulation_metal[0].fix(0)

    def initialize(blk, state_args=None, outlvl=0, solver='ipopt', optarg={'tol': 1e-6}):
        '''
        Waterwall section initialization routine.

        Keyword Arguments:
            state_args : a dict of arguments to be passed to the property
                           package(s) for the control_volume of the model to
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

        flags = blk.control_volume.initialize(outlvl=outlvl+1,
                                       optarg=optarg,
                                       solver=solver,
                                       state_args=state_args)
        init_log.info_high("Initialization Step 1 Complete.")
        # Fix outlet enthalpy and pressure
        for t in blk.flowsheet().config.time:
            blk.control_volume.properties_out[t].enth_mol.fix(
                value(blk.control_volume.properties_in[t].enth_mol) +
                value(blk.heat_fireside[t])/
                value(blk.control_volume.properties_in[t].flow_mol)
            )
            blk.control_volume.properties_out[t].pressure.fix(
                value(blk.control_volume.properties_in[t].pressure) - 1.0
            )
        blk.heat_eqn.deactivate()
        blk.pressure_change_total_eqn.deactivate()

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info_high(
                "Initialization Step 2 {}.".format(idaeslog.condition(res))
            )

        # Unfix outlet enthalpy and pressure
        for t in blk.flowsheet().config.time:
            blk.control_volume.properties_out[t].enth_mol.unfix()
            blk.control_volume.properties_out[t].pressure.unfix()
        blk.heat_eqn.activate()
        blk.pressure_change_total_eqn.activate()

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info_high(
                "Initialization Step 3 {}.".format(idaeslog.condition(res))
            )

        blk.control_volume.release_state(flags, outlvl+1)
        init_log.info("Initialization Complete.")

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()
        for t, c in self.energy_holdup_slag_eqn.items():
            s = iscale.get_scaling_factor(
                self.energy_holdup_slag[t], default=1, warning=True)
            iscale.constraint_scaling_transform(c, s)
        for t, c in self.friction_factor_darcy_eqn.items():
            s = iscale.get_scaling_factor(self.N_Re[t], default=1, warning=True)
            iscale.constraint_scaling_transform(c, s)
        for t, c in self.volume_eqn.items():
            s = iscale.get_scaling_factor(
                self.volume[t], default=1, warning=True)
            iscale.constraint_scaling_transform(c, s)
        for t, c in self.heat_flux_conv_eqn.items():
            s = iscale.get_scaling_factor(
                self.heat_flux_conv[t], default=1, warning=True)
            s *= iscale.get_scaling_factor(
                self.diameter_in, default=1, warning=True)
            iscale.constraint_scaling_transform(c, s/10.0)
        for t, c in self.energy_holdup_slag_eqn.items():
            s = iscale.get_scaling_factor(
                self.energy_holdup_slag[t], default=1, warning=True)
            iscale.constraint_scaling_transform(c, s)
        for t, c in self.energy_holdup_metal_eqn.items():
            s = iscale.get_scaling_factor(
                self.energy_holdup_metal[t], default=1, warning=True)
            iscale.constraint_scaling_transform(c, s)

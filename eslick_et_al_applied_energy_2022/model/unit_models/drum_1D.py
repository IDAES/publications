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
1-D drum model
main equations:
    Heat loss is a variable given by the user (zero heat loss can be specified if adiabatic)
    Calculate pressure change due to gravity based on water level and contraction to downcomer
    Water level is either fixed for steady-state model or calculated for dynamic model
    Assume enthalpy_in == enthalpy_out + heat loss
    Subcooled water from economizer and staturated water from waterwall are mixed before enerting drum
    Drum1D has only one inlet and one outlet

Created: February 2019 by Jinliang Ma
@JinliangMa: jinliang.ma@netl.doe.gov
"""
from __future__ import division

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

from idaes.core.util.config import (is_physical_parameter_block,
                                    is_reaction_parameter_block)
from idaes.core.util.misc import add_object_reference


# Additional import for the unit operation
from pyomo.environ import SolverFactory, value, Var, Param, asin, cos, log10, log, sqrt, Constraint, TransformationFactory
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.opt import TerminationCondition
import idaes.core.util.scaling as iscale

import idaes.logger as idaeslog

__author__ = "Jinliang Ma <jinliang.ma.doe.gov>"
__version__ = "2.0.0"


@declare_process_block_class("Drum1D")
class Drum1DData(UnitModelBlockData):
    """
    1-D Boiler Drum Unit Operation Class
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
    CONFIG.declare("inside_diameter",ConfigValue(
        default= 1.0,
        description='inside diameter of drum',
        doc='define inside diameter of drum'))

    CONFIG.declare("thickness",ConfigValue(
        default= 0.1,
        description='drum wall thickness',
        doc='define drum wall thickness'))

    def build(self):
        """
        Begin building model (pre-DAE transformation).
        Args:
            None
        Returns:
            None
        """

        # Call UnitModel.build to setup dynamics
        super(Drum1DData, self).build()

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
        Define the geometry of the unit as necessary
        """

        # Constant pi
        self.pi = Param(
                initialize=math.pi,
                doc="Pi")
        # Constant gravity
        self.gravity = Param(
                initialize=9.806,
                doc="Gravity")

        di = self.config.inside_diameter
        thk = self.config.thickness

        # Inside diameter of drum
        self.diameter_drum = Param(
                initialize=di,
                doc="Inside diameter of drum")
        # Thickness of drum wall
        self.thickness_drum = Param(
                initialize=thk,
                doc="wall thickness of drum")
        # Thickness of insulation layer
        self.thickness_insulation = Var(
                initialize = 0.2,
                doc="Insulation layer thickness")
        # Length of drum
        self.length_drum = Var(
                initialize=10,
                doc="Horizontal length of drum")
        # Number of downcomers connected at the bottom of drum, used to calculate contrac
        self.count_downcomer = Var(
                initialize=4,
                doc="Number of downcomers connected to drum")
        # Inside diameter of downcomer
        self.diameter_downcomer = Var(
                initialize=0.6,
                doc="Inside diameter of downcomer")

        # Inside Radius expression
        @self.Expression(doc="Inside radius of drum")
        def rin_drum(b):
            return 0.5*b.diameter_drum

        # Outside Radius expression
        @self.Expression(doc="Outside radius of drum")
        def rout_drum(b):
            return b.rin_drum+b.thickness_drum

        # Outside diameter expression
        @self.Expression(doc="Outside radius of drum")
        def dout_drum(b):
            return b.diameter_drum+2*b.thickness_drum

        # Inner surface area (ignore two hemispheres at ends)
        @self.Expression(doc="Inner surface area")
        def area_drum(b):
            return b.pi*b.diameter_drum*b.length_drum

    def _make_performance(self):
        """
        Define constraints which describe the behaviour of the unit model.
        """

        # thermal conductivity of drum
        self.cond_therm_metal = Param(initialize = 40, mutable=True)

        # thermal conductivity of insulation
        self.cond_therm_insulation = Param(initialize = 0.08, mutable=True)

        # thermal conductivity of air
        self.cond_therm_air = Param(initialize = 0.03, mutable=True)

        # thermal diffusivity of drum
        #self.diff_therm_metal = Param(initialize = 9.5E-6, mutable=True)

        # material density of drum
        self.density_thermal_metal = Param(initialize = 7753, mutable=True)

        # heat capacity of drum
        self.heat_capacity_thermal_drum = Param(initialize = 486, mutable=True)

        # Young modulus
        self.Young_modulus = Param(initialize = 2.07E5, mutable=True)

        # Poisson's ratio
        self.Poisson_ratio = Param(initialize = 0.292, mutable=True)

        # Coefficient of thermal expansion
        self.coefficient_therm_expansion = Param(initialize = 1.4E-5, mutable=True)

        # constant related to Ra, (gravity*expansion_coefficient*density/viscosity/thermal_diffusivity)^(1/6)
        # use properties at 50 C and 1 atm, 6.84e7^(1/6)=20.223
        self.const_Ra_root6 = Param(initialize = 20.223, mutable=True)

        # constant related to Nu for free convection, 0.387/(1+0.721*Pr^(-9/16))^(8/27)
        # use properties at 50 C and 1 atm
        self.const_Nu = Param(initialize = 0.322, mutable=True)

        # ambient pressure
        self.pres_amb = Param(initialize = 0.81E5, doc="ambient pressure")

        # ambient temperature
        self.temp_amb = Var(self.flowsheet().config.time, initialize = 300, doc="ambient temperature")

        # inside heat transfer coefficient
        self.h_in = Var(self.flowsheet().config.time, initialize = 1, doc="inside heat transfer coefficient")

        # outside heat transfer coefficient
        self.h_out = Var(self.flowsheet().config.time, initialize = 1, doc="outside heat transfer coefficient")

        # insulation free convection heat transfer coefficient
        self.h_free_conv = Var(self.flowsheet().config.time, initialize = 1, doc="insulation free convection heat transfer coefficient")

        # Ra number of free convection
        self.N_Ra_root6 = Var(self.flowsheet().config.time, initialize = 80, doc="1/6 power of Ra number of free convection of air")

        # Nu number  of free convection
        self.N_Nu = Var(self.flowsheet().config.time, initialize = 1, doc="Nu number of free convection of air")

        # Define the continuous domains for model
        self.r = ContinuousSet(bounds=(self.rin_drum, self.rout_drum))

        # Temperature across wall thickness
        self.T = Var(self.flowsheet().config.time, self.r, bounds=(280,800), initialize = 550)

        # Declare derivatives in the model
        if self.config.dynamic is True:
            self.dTdt = DerivativeVar(self.T, wrt = self.flowsheet().config.time)
        self.dTdr = DerivativeVar(self.T, wrt = self.r)
        self.d2Tdr2 = DerivativeVar(self.T, wrt = (self.r, self.r))

        discretizer = TransformationFactory('dae.finite_difference')
        discretizer.apply_to(self, nfe=self.config.finite_elements, wrt=self.r, scheme='CENTRAL')

        # Add performance variables
        self.level = Var(
                self.flowsheet().config.time,
                initialize=1.0,
                doc='Water level from the bottom of the drum')

        # Velocity of fluid inside downcomer pipe
        self.velocity_downcomer = Var(
                self.flowsheet().config.time,
                initialize=10.0,
                doc='Liquid water velocity at the top of downcomer')

        # Pressure change due to contraction
        self.deltaP_contraction = Var(
                self.flowsheet().config.time,
                initialize=-1.0,
                doc='Pressure change due to contraction')

        # Pressure change due to gravity
        self.deltaP_gravity = Var(
                self.flowsheet().config.time,
                initialize=1.0,
                doc='Pressure change due to gravity')

        # thermal diffusivity of drum
        @self.Expression(doc="Thermal diffusivity of drum")
        def diff_therm_metal(b):
            return b.cond_therm_metal/(b.density_thermal_metal*b.heat_capacity_thermal_drum)

        # Expressure for the angle from the drum center to the circumference point at water level
        @self.Expression(self.flowsheet().config.time, doc="angle of water level")
        def alpha_drum(b, t):
            return asin((b.level[t]-b.rin_drum)/b.rin_drum)

        # Expressure for the fraction of wet area
        @self.Expression(self.flowsheet().config.time, doc="fraction of wet area")
        def frac_wet_area(b, t):
            return (b.alpha_drum[t]+b.pi/2)/b.pi

        # Constraint for volume liquid in drum
        @self.Constraint(self.flowsheet().config.time, doc="volume of liquid in drum")
        def volume_eqn(b, t):
            return b.volume[t] == \
                   ((b.alpha_drum[t]+0.5*b.pi)*b.rin_drum**2 + \
                   b.rin_drum*cos(b.alpha_drum[t])*(b.level[t]-b.rin_drum))* \
                   b.length_drum

        # Equation for velocity at the entrance of downcomer
        @self.Constraint(self.flowsheet().config.time, doc="Vecolity at entrance of downcomer")
        def velocity_eqn(b, t):
            return b.velocity_downcomer[t]*0.25*b.pi*b.diameter_downcomer**2*b.count_downcomer \
                   == b.control_volume.properties_out[t].flow_vol

        # Pressure change equation for contraction, -0.5*1/2*density*velocity^2 for stagnation head loss
        # plus 1/2*density*velocity^2 dynamic head (acceleration pressure change)
        @self.Constraint(self.flowsheet().config.time, doc="pressure change due to contraction")
        def pressure_change_contraction_eqn(b, t):
            return b.deltaP_contraction[t] == \
                   -0.75 * b.control_volume.properties_out[t].dens_mass_phase["Liq"] * \
                   b.velocity_downcomer[t]**2

        # Pressure change equation for gravity, density*gravity*height
        @self.Constraint(self.flowsheet().config.time, doc="pressure change due to gravity")
        def pressure_change_gravity_eqn(b, t):
            return b.deltaP_gravity[t] == \
                   b.control_volume.properties_out[t].dens_mass_phase["Liq"] * b.gravity * b.level[t]

        # Total pressure change equation
        @self.Constraint(self.flowsheet().config.time, doc="pressure drop")
        def pressure_change_total_eqn(b, t):
            return b.deltaP[t] == b.deltaP_contraction[t] + b.deltaP_gravity[t]

        # Constraint for heat conduction equation
        @self.Constraint(self.flowsheet().config.time, self.r, doc="1-D heat conduction equation through radius")
        def heat_conduction_eqn(b, t, r):
            if r == b.r.first() or r == b.r.last():
                return Constraint.Skip
            if self.config.dynamic is True:
                return b.dTdt[t,r] == b.diff_therm_metal * b.d2Tdr2[t,r] + b.diff_therm_metal*(1/r)*b.dTdr[t,r]
            else:
                return 0 == b.diff_therm_metal * b.d2Tdr2[t,r] + b.diff_therm_metal*(1/r)*b.dTdr[t,r]

        @self.Constraint(self.flowsheet().config.time, doc="inner wall BC")
        def inner_wall_bc_eqn(b, t):
            return b.h_in[t]*(b.control_volume.properties_out[t].temperature - b.T[t,b.r.first()]) == \
                   -b.dTdr[t,b.r.first()]*b.cond_therm_metal

        @self.Constraint(self.flowsheet().config.time, doc="outer wall BC")
        def outer_wall_bc_eqn(b, t):
            return b.h_out[t]*(b.T[t,b.r.last()] - b.temp_amb[t]) == \
                   -b.dTdr[t, b.r.last()]*b.cond_therm_metal

        # Inner wall BC for dTdt
        @self.Constraint(self.flowsheet().config.time, doc="extra inner wall temperature derivative")
        def extra_at_inner_wall_eqn(b, t):
            if self.config.dynamic is True:
                term = b.dTdt[t, b.r.first()]
            else:
                term = 0
            return term == 4*b.diff_therm_metal*(b.r.first()+b.r[2])/ \
                (b.r[2]-b.r.first())**2/(3*b.r.first()+b.r[2])*(b.T[t,b.r[2]] \
                -b.T[t,b.r.first()]) + 8*b.diff_therm_metal/b.cond_therm_metal*b.h_in[t]*b.r.first()/ \
                (b.r[2]-b.r.first())/(3*b.r.first()+b.r[2])*(b.control_volume.properties_out[t].temperature \
                -b.T[t,b.r.first()])

        @self.Constraint(self.flowsheet().config.time, doc="extra outer wall temperature derivative")
        def extra_at_outer_wall_eqn(b, t):
            if self.config.dynamic is True:
                term = b.dTdt[t, b.r.last()]
            else:
                term = 0
            return term == 4*b.diff_therm_metal*(b.r.last()+b.r[-2])/ \
                (b.r.last()-b.r[-2])**2/(3*b.r.last()+b.r[-2])*(b.T[t,b.r[-2]] \
                -b.T[t,b.r.last()]) + 8*b.diff_therm_metal/b.cond_therm_metal*b.h_out[t]*b.r.last()/ \
                (b.r.last()-b.r[-2])/(3*b.r.last()+b.r[-2])*(b.temp_amb[t] \
                -b.T[t,b.r.last()])

        # reduced pressure expression
        @self.Expression(self.flowsheet().config.time, doc="reduced pressure")
        def pres_reduced(b,t):
            return b.control_volume.properties_out[t].pressure/2.2048e7

        # calculate inner side heat transfer coefficient with minimum temperature difference set to sqrt(0.1)
        # multipling wet area fraction to convert it to the value based on total circumference
        @self.Constraint(self.flowsheet().config.time, doc="inner side heat transfer coefficient")
        def h_in_eqn(b, t):
            return b.h_in[t] == 2178.6*(b.control_volume.properties_out[t].pressure/2.2048e7)**0.36\
                   /(-log10(b.control_volume.properties_out[t].pressure/2.2048e7))**1.65 *\
                   (0.1+(b.control_volume.properties_out[t].temperature-b.T[t,b.r.first()])**2)*b.frac_wet_area[t]

        # Expressure for insulation heat transfer (conduction) resistance based on drum metal outside diameter
        @self.Expression(doc="heat transfer resistance of insulation layer")
        def resistance_insulation(b):
            return b.rout_drum*log((b.rout_drum+b.thickness_insulation)/b.rout_drum)/b.cond_therm_insulation

        # h_out equation considering conduction through insulation and free convection between insulation and ambient
        @self.Constraint(self.flowsheet().config.time, doc="outer side heat transfer coefficient")
        def h_out_eqn(b, t):
            return b.h_out[t]*(b.resistance_insulation + 1/b.h_free_conv[t]) == 1.0

        # Expressure for outside insulation wall temperature (skin temperature)
        @self.Expression(self.flowsheet().config.time, doc="outside insulation wall temperature")
        def temp_insulation_outside(b, t):
            return b.temp_amb[t] + (b.T[t,b.r.last()]-b.temp_amb[t])*b.h_out[t]/b.h_free_conv[t]

        # Ra number equation
        @self.Constraint(self.flowsheet().config.time, doc="Ra number of free convection")
        def Ra_number_eqn(b, t):
            return b.N_Ra_root6[t] == b.const_Ra_root6*sqrt(b.dout_drum+2*b.thickness_insulation)*(b.T[t,b.r.last()]-b.temp_amb[t])**0.166667

        # Nu number equation
        @self.Constraint(self.flowsheet().config.time, doc=" Nu number of free convection")
        def Nu_number_eqn(b, t):
            return b.N_Nu[t] == (0.6+b.const_Nu*b.N_Ra_root6[t])**2

        # free convection coefficient based on the drum metal outside diameter
        @self.Constraint(self.flowsheet().config.time, doc="free convection heat transfer coefficient between insulation wall and ambient")
        def h_free_conv_eqn(b, t):
            return b.h_free_conv[t] == b.N_Nu[t]*b.cond_therm_air/b.dout_drum

        @self.Constraint(self.flowsheet().config.time, doc="heat loss of water")
        def heat_loss_eqn(b, t):
            return b.heat_duty[t] == b.area_drum*b.h_in[t]*(b.T[t,b.r.first()]\
                   -b.control_volume.properties_out[t].temperature)

        ### Calculate mechanical and thermal stresses
        # ----------------------------------------------------------------------------------------------

        # Integer indexing for radius domain
        self.rindex = Param(self.r, initialize=1, mutable= True, doc="inter indexing for radius domain")

        # calculate integral point for mean temperature in the wall
        @self.Expression(self.flowsheet().config.time, doc="integral point used to estimate mean temperature")
        def int_mean_temp(b,t):
            return 2*(b.r[2]-b.r[1])/(b.rout_drum**2-b.rin_drum**2)*(sum(0.5*(b.r[i-1]*b.T[t,b.r[i-1]]+\
                b.r[i]*b.T[t,b.r[i]]) for i in range(2,len(b.r)+1)))

        for index_r,value_r in enumerate(self.r,1):
            self.rindex[value_r] = index_r

        @self.Expression(self.flowsheet().config.time, self.r, doc="integral point at each element")
        def int_temp(b,t,r):
            if b.rindex[r].value == 1:
                return b.T[t,b.r.first()]
            else:
                return 2*(b.r[2]-b.r[1])/(b.r[b.rindex[r].value]**2-b.rin_drum**2)\
                *(sum(0.5*(b.r[j-1]*b.T[t,b.r[j-1]]+b.r[j]*b.T[t,b.r[j]])\
                 for j in range(2,b.rindex[r].value+1)))

        @self.Expression(self.flowsheet().config.time, self.r, doc="thermal stress at radial direction")
        def therm_stress_radial(b,t,r):
            return 0.5*b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)\
            *((1-b.rin_drum**2/r**2)*(b.int_mean_temp[t]-b.int_temp[t,r]))

        @self.Expression(self.flowsheet().config.time, self.r, doc="thermal stress at circumferential direction")
        def therm_stress_circumferential(b,t,r):
            return 0.5*b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)\
                *((1+b.rin_drum**2/r**2)*b.int_mean_temp[t]+(1-b.rin_drum**2/r**2)*b.int_temp[t,r]-2*b.T[t,r])

        @self.Expression(self.flowsheet().config.time, self.r, doc="thermal stress at axial direction")
        def therm_stress_axial(b,t,r):
            return b.Young_modulus*b.coefficient_therm_expansion/(1-b.Poisson_ratio)*(b.int_mean_temp[t]-b.T[t,r])

        @self.Expression(self.flowsheet().config.time, self.r, doc="mechanical stress at radial direction")
        def mech_stress_radial(b,t,r):
            return 0.1*(1E-5*(b.control_volume.properties_out[t].pressure*b.rin_drum**2-b.pres_amb*b.rout_drum**2)\
                /(b.rout_drum**2-b.rin_drum**2)+(1E-5*(b.pres_amb-b.control_volume.properties_out[t].pressure)\
                *b.rin_drum**2*b.rout_drum**2/(r**2*(b.rout_drum**2-b.rin_drum**2))))

        @self.Expression(self.flowsheet().config.time, self.r, doc="mechanical stress at circumferential direction")
        def mech_stress_circumferential(b,t,r):
            return 0.1*(1E-5*(b.control_volume.properties_out[t].pressure*b.rin_drum**2-b.pres_amb*b.rout_drum**2)\
                /(b.rout_drum**2-b.rin_drum**2)-(1E-5*(b.pres_amb-b.control_volume.properties_out[t].pressure)\
                    *b.rin_drum**2*b.rout_drum**2/(r**2*(b.rout_drum**2-b.rin_drum**2))))

        @self.Expression(self.flowsheet().config.time, self.r, doc="mechanical stress at axial direction")
        def mech_stress_axial(b,t,r):
            return 0.1*(1E-5*(b.control_volume.properties_out[t].pressure*b.rin_drum**2-b.pres_amb*b.rout_drum**2)\
                /(b.rout_drum**2-b.rin_drum**2))

    def set_initial_condition(self):
        if self.config.dynamic is True:
            self.control_volume.material_accumulation[:,:,:].value = 0
            self.control_volume.energy_accumulation[:,:].value = 0
            self.dTdt[:,:].value = 0
            self.control_volume.material_accumulation[0,:,:].fix(0)
            self.control_volume.energy_accumulation[0,:].fix(0)
            self.dTdt[0,:].fix(0)

    def initialize(blk, state_args={}, outlvl=0, solver='ipopt', optarg={'tol': 1e-6}):
        '''
        Drum1D initialization routine.

        Keyword Arguments:
            state_args : a dict of arguments to be passed to the property
                           package(s) for the control_volume of the model to
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

        flags = blk.control_volume.initialize(outlvl=outlvl+1,
                                       optarg=optarg,
                                       solver=solver,
                                       state_args=state_args)
        init_log.info_high("Initialization Step 1 Complete.")

        # set initial values for T
        r_mid = value((blk.r.first()+blk.r.last())/2)
        # assume outside wall temperature is 10 K lower than fluid temperature
        T_out = value(blk.control_volume.properties_in[0].temperature-1)
        T_mid = value((T_out+blk.control_volume.properties_in[0].temperature)/2)
        slope = value((T_out-blk.control_volume.properties_in[0].temperature)/ \
                      (blk.r.last()-blk.r.first())/3)
        for x in blk.r:
            blk.T[:,x].fix(T_mid + slope*(x-r_mid))
        blk.T[:,:].unfix()

        # Fix outlet enthalpy and pressure
        for t in blk.flowsheet().config.time:
            blk.control_volume.properties_out[t].pressure.fix(value(blk.control_volume.properties_in[0].pressure) \
            - 5000.0)
            blk.control_volume.properties_out[t].enth_mol.fix(value(blk.control_volume.properties_in[0].enth_mol))
        blk.pressure_change_total_eqn.deactivate()
        blk.heat_loss_eqn.deactivate()

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info_high(
                "Initialization Step 2 {}.".format(idaeslog.condition(res))
            )

        # Unfix outlet enthalpy and pressure
        for t in blk.flowsheet().config.time:
            blk.control_volume.properties_out[t].pressure.unfix()
            blk.control_volume.properties_out[t].enth_mol.unfix()
        blk.pressure_change_total_eqn.activate()
        blk.heat_loss_eqn.activate()

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info_high(
                "Initialization Step 2 {}.".format(idaeslog.condition(res))
            )

        blk.control_volume.release_state(flags, outlvl-1)
        init_log.info("Initialization Complete.")

    def calculate_scaling_factors(self):
        for v in self.deltaP_gravity.values():
            if iscale.get_scaling_factor(v, warning=True) is None:
                iscale.set_scaling_factor(v, 1e-3)

        for v in self.deltaP_contraction.values():
            if iscale.get_scaling_factor(v, warning=True) is None:
                iscale.set_scaling_factor(v, 1e-3)

        for t, c in self.heat_loss_eqn.items():
            sf = iscale.get_scaling_factor(
                self.heat_duty[t], default=1, warning=True)
            iscale.constraint_scaling_transform(c, sf)

        for t, c in self.pressure_change_contraction_eqn.items():
            sf = iscale.get_scaling_factor(
                self.deltaP_contraction[t], default=1, warning=True)
            iscale.constraint_scaling_transform(c, sf)

        for t, c in self.pressure_change_gravity_eqn.items():
            sf = iscale.get_scaling_factor(
                self.deltaP_gravity[t], default=1, warning=True)
            iscale.constraint_scaling_transform(c, sf)

        for t, c in self.pressure_change_total_eqn.items():
            sf = iscale.get_scaling_factor(
                self.deltaP[t], default=1, warning=True)
            iscale.constraint_scaling_transform(c, sf)

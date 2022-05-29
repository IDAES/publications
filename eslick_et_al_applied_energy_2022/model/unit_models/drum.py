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
drum model
main equations:
    Heat loss is a variable given by the user (zero heat loss can be specified if adiabatic)
    Calculate pressure change due to gravity based on water level and contraction to downcomer
    Water level is either fixed for steady-state model or calculated for dynamic model
    Assume enthalpy_in == enthalpy_out + heat loss
    Subcooled water from economizer and staturated water from waterwall are mixed before enerting drum
    Drum has only one inlet and one outlet

Created: February 2019 by Jinliang Ma
@JinliangMa: jinliang.ma@netl.doe.gov
"""
from __future__ import division

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
    useDefault
)
import idaes.logger as idaeslog

from idaes.core.util.config import (is_physical_parameter_block,
                                    is_reaction_parameter_block)
from idaes.core.util.misc import add_object_reference


# Additional import for the unit operation
from pyomo.environ import SolverFactory, value, Var, Param, asin, cos
import idaes.core.util.scaling as iscale

from pyomo.opt import TerminationCondition

__author__ = "Jinliang Ma <jinliang.ma.doe.gov>"
__version__ = "2.0.0"

@declare_process_block_class("Drum")
class DrumData(UnitModelBlockData):
    """
    Boiler Drum Unit Operation Class
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


    def build(self):
        """
        Begin building model (pre-DAE transformation).
        Args:
            None
        Returns:
            None
        """

        # Call UnitModel.build to setup dynamics
        super(DrumData, self).build()

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
        # Inside diameter of drum
        self.diameter_drum = Var(
                initialize=1.0,
                doc="Inside diameter of drum")
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


    def _make_performance(self):
        """
        Define constraints which describe the behaviour of the unit model.
        """

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

        # Radius expression
        @self.Expression(doc="Radius of drum")
        def radius_drum(b):
            return 0.5*b.diameter_drum

        # Expressure for the angle from the drum center to the circumference point at water level
        @self.Expression(self.flowsheet().config.time, doc="angle of water level")
        def alpha_drum(b, t):
            return asin((b.level[t]-b.radius_drum)/b.radius_drum)

        # Constraint for volume liquid in drum
        @self.Constraint(self.flowsheet().config.time, doc="volume of liquid in drum")
        def volume_eqn(b, t):
            return b.volume[t] == \
                   ((b.alpha_drum[t]+0.5*b.pi)*b.radius_drum**2 + \
                   b.radius_drum*cos(b.alpha_drum[t])*(b.level[t]-b.radius_drum))* \
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
            return 1e-3*b.deltaP_contraction[t] == \
                   -1e-3*0.75 * b.control_volume.properties_out[t].dens_mass_phase["Liq"] * \
                   b.velocity_downcomer[t]**2

        # Pressure change equation for gravity, density*gravity*height
        @self.Constraint(self.flowsheet().config.time, doc="pressure change due to gravity")
        def pressure_change_gravity_eqn(b, t):
            return 1e-3*b.deltaP_gravity[t] == 1e-3*\
                   b.control_volume.properties_out[t].dens_mass_phase["Liq"] * b.gravity * b.level[t]

        # Total pressure change equation
        @self.Constraint(self.flowsheet().config.time, doc="pressure drop")
        def pressure_change_total_eqn(b, t):
            return 1e-3*b.deltaP[t] == 1e-3*(b.deltaP_contraction[t] + b.deltaP_gravity[t])

    def set_initial_condition(self):
        if self.config.dynamic is True:
            self.control_volume.material_accumulation[:,:,:].value = 0
            self.control_volume.energy_accumulation[:,:].value = 0
            self.control_volume.material_accumulation[0,:,:].fix(0)
            self.control_volume.energy_accumulation[0,:].fix(0)

    def initialize(blk, state_args=None, outlvl=0, solver='ipopt', optarg={'tol': 1e-6}):
        '''
        Drum initialization routine.

        Keyword Arguments:
            state_args : a dict of arguments to be passed to the property
                           package(s) for the control_volume of the model to
                           provide an initial state for initialization
                           (see documentation of the specific property package)
                           (default = None).
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

        flags = blk.control_volume.initialize(
            outlvl=outlvl+1,
            optarg=optarg,
            solver=solver,
            state_args=state_args
        )
        init_log.info_high("Initialization Step 1 Complete.")

        # Fix outlet enthalpy and pressure
        for t in blk.flowsheet().config.time:
            blk.control_volume.properties_out[t].pressure.fix(
                value(blk.control_volume.properties_in[t].pressure) - 1000.0
            )
        blk.pressure_change_total_eqn.deactivate()

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info_high(
                "Initialization Step 2 {}.".format(idaeslog.condition(res))
            )

        # Unfix outlet enthalpy and pressure
        for t in blk.flowsheet().config.time:
            blk.control_volume.properties_out[t].pressure.unfix()
        blk.pressure_change_total_eqn.activate()

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info_high(
                "Initialization Step 3 {}.".format(idaeslog.condition(res))
            )

        blk.control_volume.release_state(flags, outlvl+1)
        init_log.info("Initialization Complete.")

    def calculate_scaling_factors(self):
        for v in self.deltaP_gravity.values():
            if iscale.get_scaling_factor(v, warning=True) is None:
                iscale.set_scaling_factor(v, 1e-3)

        for v in self.deltaP_contraction.values():
            if iscale.get_scaling_factor(v, warning=True) is None:
                iscale.set_scaling_factor(v, 1e-3)

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

# Import moving bed from workspace as unit changes in 1.13-dev broke
# some of my code
from common.unit_models.moving_bed import MBRData
from idaes.core import (
        declare_process_block_class,
        ControlVolume1DBlock,
        FlowDirection,
        EnergyBalanceType,
        MomentumBalanceType,
        MaterialBalanceType,
        )
from idaes.core.control_volume1d import DistributedVars

from pyomo.core.base.var import Var
from pyomo.core.base.block import _BlockData
from pyomo.dae import ContinuousSet
from pyomo.environ import TransformationFactory, Reals

@declare_process_block_class("MBR")
class BidirectionalMBData(MBRData):
    """
    All this subclass does is override the build method
    to let me construct gas and solid control volumes
    with different length domains.
    """

    def build(self):
        """
        Begin building model (pre-DAE transformation).

        Args:
            None

        Returns:
            None
        """
        # Call UnitModel.build to build default attributes
        super(MBRData, self).build()

        # Set flow directions for the control volume blocks
        # Gas flows from 0 to 1, solid flows from 1 to 0
        # An if statement is used here despite only one option to allow for
        # future extensions to other flow configurations
        if self.config.flow_type == "counter_current":
            set_direction_gas = FlowDirection.forward
            set_direction_solid = FlowDirection.backward

            # Set transformation scheme to be in the "opposite
            # direction" as flow.
            self.GAS_TRANSFORM_SCHEME = "BACKWARD"
            self.SOLID_TRANSFORM_SCHEME = "FORWARD"
        else:
            raise BurntToast(
                    "{} encountered unrecognized argument "
                    "for flow type. Please contact the IDAES"
                    " developers with this bug.".format(self.name))
        # Set arguments for gas sides if homoogeneous reaction block
        if self.config.gas_phase_config.reaction_package is not None:
            has_rate_reaction_gas_phase = True
        else:
            has_rate_reaction_gas_phase = False

        # Set arguments for gas and solid sides if heterogeneous reaction block
        if self.config.solid_phase_config.reaction_package is not None:
            has_rate_reaction_solid_phase = True
            has_mass_transfer_gas_phase = True
        else:
            has_rate_reaction_solid_phase = False
            has_mass_transfer_gas_phase = False

        # Set heat transfer terms
        if self.config.energy_balance_type != EnergyBalanceType.none:
            has_heat_transfer = True
        else:
            has_heat_transfer = False

        # Set heat of reaction terms
        if (self.config.energy_balance_type != EnergyBalanceType.none
                and self.config.gas_phase_config.reaction_package is not None):
            has_heat_of_reaction_gas_phase = True
        else:
            has_heat_of_reaction_gas_phase = False

        if (self.config.energy_balance_type != EnergyBalanceType.none
                and self.config.solid_phase_config.
                reaction_package is not None):
            has_heat_of_reaction_solid_phase = True
        else:
            has_heat_of_reaction_solid_phase = False

        # Create two different length domains; one for each phase.
        # Add them to this block so I can use one of them to index
        # all the variables on this block.
        # I can then call discretization on this block, which will
        # discretize the variables on the control volume blocks.
        self.solid_length_domain = ContinuousSet(
                                bounds=(0.0, 1.0),
                                initialize=self.config.length_domain_set,
                                doc="Normalized length domain",
                                )
        self.gas_length_domain = ContinuousSet(
                                bounds=(0.0, 1.0),
                                initialize=self.config.length_domain_set,
                                doc="Normalized length domain",
                                )
        self.bed_height = Var(domain=Reals, initialize=1, doc="Bed length [m]")
        super(_BlockData, self).__setattr__(
                'length_domain',
                self.solid_length_domain,
                )

    # =========================================================================
        """ Build Control volume 1D for gas phase and
            populate gas control volume"""

        self.gas_phase = ControlVolume1DBlock(default={
            "transformation_method": self.config.transformation_method,
            #"transformation_scheme": self.config.transformation_scheme,
            "transformation_scheme": self.GAS_TRANSFORM_SCHEME,
            "finite_elements": self.config.finite_elements,
            "collocation_points": self.config.collocation_points,
            "dynamic": self.config.dynamic,
            "has_holdup": self.config.has_holdup,
            "area_definition": DistributedVars.variant,
            "property_package": self.config.gas_phase_config.property_package,
            "property_package_args":
                self.config.gas_phase_config.property_package_args,
            "reaction_package": self.config.gas_phase_config.reaction_package,
            "reaction_package_args":
                self.config.gas_phase_config.reaction_package_args})

        # Pass gas_length_domain to the gas phase control volume
        # Note that length_domain_set is redundant as the set is
        # already initialized.
        self.gas_phase.add_geometry(
                length_domain=self.gas_length_domain,
                length_domain_set=self.config.length_domain_set,
                flow_direction=set_direction_gas,
                )

        self.gas_phase.add_state_blocks(
            information_flow=set_direction_gas,
            has_phase_equilibrium=False)

        if self.config.gas_phase_config.reaction_package is not None:
            self.gas_phase.add_reaction_blocks(
                    has_equilibrium=self.config.gas_phase_config.
                    has_equilibrium_reactions)

        self.gas_phase.add_material_balances(
            balance_type=self.config.material_balance_type,
            has_phase_equilibrium=False,
            has_mass_transfer=has_mass_transfer_gas_phase,
            has_rate_reactions=has_rate_reaction_gas_phase)

        self.gas_phase.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_heat_transfer=has_heat_transfer,
            has_heat_of_reaction=has_heat_of_reaction_gas_phase)

        self.gas_phase.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=self.config.has_pressure_change)

    # =========================================================================
        """ Build Control volume 1D for solid phase and
            populate solid control volume"""

        # Set argument for heterogeneous reaction block
        self.solid_phase = ControlVolume1DBlock(default={
            "transformation_method": self.config.transformation_method,
            "transformation_scheme": self.SOLID_TRANSFORM_SCHEME,
            "finite_elements": self.config.finite_elements,
            "collocation_points": self.config.collocation_points,
            # ^ These arguments have no effect as the transformation
            # is applied in this class.
            "dynamic": self.config.dynamic,
            "has_holdup": self.config.has_holdup,
            "area_definition": DistributedVars.variant,
            "property_package":
                self.config.solid_phase_config.property_package,
            "property_package_args":
                self.config.solid_phase_config.property_package_args,
            "reaction_package":
                self.config.solid_phase_config.reaction_package,
            "reaction_package_args":
                self.config.solid_phase_config.reaction_package_args})

        # Same comment as made for the gas phase.
        # Pass in the set we've constructed for this purpose.
        self.solid_phase.add_geometry(
                length_domain=self.solid_length_domain,
                length_domain_set=self.config.length_domain_set,
                flow_direction=set_direction_solid)

        # These constraints no longer are created by make_performance,
        # So I make them here...
        # Length of gas side, and solid side
        @self.Constraint(doc="Gas side length")
        def gas_phase_length(b):
            return (b.gas_phase.length == b.bed_height)

        @self.Constraint(doc="Solid side length")
        def solid_phase_length(b):
            return (b.solid_phase.length == b.bed_height)

        # Many other methods of the MBR base class actually rely on this
        # attribute, so here I slap on a reference. The particular set
        # I use shouldn't matter, as long as the derivatives are constructed
        # wrt the correct set.

        self.solid_phase.add_state_blocks(
            information_flow=set_direction_solid,
            has_phase_equilibrium=False)

        if self.config.solid_phase_config.reaction_package is not None:
            # TODO - a generalization of the heterogeneous reaction block
            # The heterogeneous reaction block does not use the
            # add_reaction_blocks in control volumes as control volumes are
            # currently setup to handle only homogeneous reaction properties.
            # Thus appending the heterogeneous reaction block to the
            # solid state block is currently hard coded here.

            tmp_dict = dict(**self.config.solid_phase_config.
                            reaction_package_args)
            tmp_dict["gas_state_block"] = self.gas_phase.properties
            tmp_dict["solid_state_block"] = self.solid_phase.properties
            tmp_dict["has_equilibrium"] = (self.config.solid_phase_config.
                                           has_equilibrium_reactions)
            tmp_dict["parameters"] = (self.config.solid_phase_config.
                                      reaction_package)
            self.solid_phase.reactions = (
                    self.config.solid_phase_config.reaction_package.
                    reaction_block_class(
                        self.flowsheet().config.time,
                        self.length_domain,
                        doc="Reaction properties in control volume",
                        default=tmp_dict))

        self.solid_phase.add_material_balances(
            balance_type=self.config.material_balance_type,
            has_phase_equilibrium=False,
            has_mass_transfer=False,
            has_rate_reactions=has_rate_reaction_solid_phase)

        self.solid_phase.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_heat_transfer=has_heat_transfer,
            has_heat_of_reaction=has_heat_of_reaction_solid_phase)

        self.solid_phase.add_momentum_balances(
            balance_type=MomentumBalanceType.none,
            has_pressure_change=False)

    # =========================================================================
        """ Add ports"""
        # Add Ports for gas side
        self.add_inlet_port(name="gas_inlet", block=self.gas_phase)
        self.add_outlet_port(name="gas_outlet", block=self.gas_phase)

        # Add Ports for solid side
        self.add_inlet_port(name="solid_inlet", block=self.solid_phase)
        self.add_outlet_port(name="solid_outlet", block=self.solid_phase)

    # =========================================================================
        """ Add performace equation method"""
        self._apply_transformation()
        self._make_performance()

    def _apply_transformation(self):
        if self.config.finite_elements is None:
            raise ConfigurationError(
                    'PROVIDE FINITE_ELEMENTS!!!'
                    )
        if self.config.transformation_method == 'dae.finite_difference':
            self.discretizer = TransformationFactory(
                    self.config.transformation_method)
            # Apply discretization to gas and solid phases separately
            self.discretizer.apply_to(
                    self,
                    wrt=self.gas_length_domain,
                    nfe=self.config.finite_elements,
                    scheme=self.GAS_TRANSFORM_SCHEME,
                    )
            self.discretizer.apply_to(
                    self,
                    wrt=self.solid_length_domain,
                    nfe=self.config.finite_elements,
                    scheme=self.SOLID_TRANSFORM_SCHEME,
                    )
        else:
            raise ConfigurationError(
                    'THIS SUBCLASS ONLY SUPPORTS FINITE DIFFERENCE!!!'
                    )


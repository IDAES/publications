import sympy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from dataclasses import dataclass 

# PowerSpecs dataclass - holds specifications for power system
@dataclass
class PowerSpecs:
    def __init__(self, up_bound_nameplate, min_plant_power, turndown_limit, ramp_rate = None):
        # check that everything is a float
        if type(up_bound_nameplate) is not float:
            raise TypeError('up_bound_nameplate must be a positive float')
        if type(turndown_limit) is not float:
            raise TypeError('turndown_limit must be a positive float, between 0 and 1')
        # check everything is positive
        if up_bound_nameplate < 0.0:
            raise ValueError('up_bound_nameplate must be a positive float')
        if turndown_limit < 0.0 or turndown_limit > 1.0:
            raise ValueError('turndown_limit must be a positive float, between 0 and 1')
        # upper bound of nameplate capacity
        self.up_bound_nameplate = up_bound_nameplate
        # lower bound of nameplate capacity
        self.min_plant_power = min_plant_power
        # turndown limit of plant - fraction of maximum capacity system can be operated at
        self.turndown_limit = turndown_limit
        if ramp_rate is not None:
            # check the ramp rate is a float
            if type(ramp_rate) is not float:
                raise TypeError('ramp_rate must be a positive float')
            if ramp_rate < 0:
                raise ValueError('ramp_rate must be a positive float')
            # ramping rate
            self.ramp_rate = ramp_rate

# HydrogenSpecs dataclass - holds specifications for hydrogen production system
@dataclass
class HydrogenSpecs:
    def __init__(self, up_bound_nameplate, min_plant_h2, turndown_limit, ramp_rate = None):
        # check that everything is a float
        if type(up_bound_nameplate) is not float:
            raise TypeError('up_bound_nameplate must be a positive float')
        if type(min_plant_h2) is not float:
            raise TypeError('min_plant_h2 must be a positive float')
        if type(turndown_limit) is not float:
            raise TypeError('turndown_limit must be a positive float, between 0 and 1')
        # check everything is positive
        if up_bound_nameplate < 0.0:
            raise ValueError('up_bound_nameplate must be a positive float')
        if min_plant_h2 < 0.0:
            raise ValueError('min_plant_h2 must be a positive float')
        if turndown_limit < 0.0 or turndown_limit > 1.0:
            raise ValueError('turndown_limit must be a positive float, between 0 and 1')
        # upperbound of nameplate capacity
        self.up_bound_nameplate = up_bound_nameplate
        # lower bound of nameplate capacity
        self.min_plant_h2 = min_plant_h2
        # turndown limit - fraction of maximum hydrogen output system can be operated at
        self.turndown_limit = turndown_limit
        if ramp_rate is not None:
            # check the ramp rate is a float
            if type(ramp_rate) is not float:
                raise TypeError('ramp_rate must be a positive float')
            if ramp_rate < 0:
                raise ValueError('ramp_rate must be a positive float')
            # ramping rate
            self.ramp_rate = ramp_rate

@dataclass
class StorageSpecs:
    def __init__(self, up_bound_capacity, lower_bound_capacity, sqrt_round_trip_efficiency):
        if type(up_bound_capacity) is not float:
            raise TypeError('up_bound_capacity must be a positive float')
        if type(lower_bound_capacity) is not float:
            raise TypeError('lower_bound_capacity must be a postive float')
        if type(sqrt_round_trip_efficiency) is not float:
            raise TypeError('sqrt_round_trip_efficiency must be a positive float')
        # upper bound on storage capacity
        self.up_bound_capacity = up_bound_capacity
        # lower bound on storage capacity
        self.lower_bound_capacity = lower_bound_capacity
        # round trip efficiency
        self.sqrt_round_trip_efficiency = sqrt_round_trip_efficiency


class IESModel:

    def var_cost_power_pyomo(self, m, time):
        '''
        Converts variable cost functions from standard variables to pyomo variables

        Arguments:
            self: self
            m: pyomo model for IES
            time: time-step for time series model components (m.pt/m.ht)

        Returns:
            Variable cost function in terms of pyomo model components
        '''

        return self.var_cost_power(m.pt_scaled[time]*m.reference_power)

    def var_cost_hydrogen_pyomo(self, m, time):
            '''
            Converts variable cost functions from standard variables to pyomo variables

            Arguments:
                self: self
                m: pyomo model for IES
                time: time-step for time series model components (m.pt/m.ht)

            Returns:
                Variable cost function in terms of pyomo model components
            '''
            if m.POWER:
                return self.var_cost_hydrogen(m.pt_scaled[time]*m.reference_power, m.ht[time])
            else:
                return self.var_cost_hydrogen(m.ht[time])

    def var_cost_coproduction_pyomo(self, m, time):
            '''
            Converts variable cost functions from standard variables to pyomo variables

            Arguments:
                self: self
                m: pyomo model for IES
                time: time-step for time series model components (m.pt/m.ht)

            Returns:
                Variable cost function in terms of pyomo model components
            '''

            return self.var_cost_coproduction(m.pt_scaled[time]*m.reference_power, m.ht[time])

    def fixed_cost_pyomo(self, m):
        '''
        Converts fixed cost functions from standard variables to pyomo variables

        Arguments:
            self: self
            m: pyomo model for IES

        Returns:
            Fixed cost function in terms of pyomo model components
        '''
        if self.POWER_mode and not self.HYDROGEN_mode and not self.CHARGE_mode:
            return self.fixed_cost(m.P)
        elif self.POWER_mode and self.HYDROGEN_mode and not self.CHARGE_mode:
            return self.fixed_cost(m.P, m.H)
        elif self.POWER_mode and self.CHARGE_mode and not self.HYDROGEN_mode:
            return self.fixed_cost(m.P, m.E)
        elif self.HYDROGEN_mode and not self.POWER_mode:
            return self.fixed_cost(m.H)
        else:
            raise NotImplementedError()

    def h2_power_pyomo(self, m, time):
        '''
        Converts h2_power functions from standard variables to pyomo variables

        Arguments:
            self: self
            m: pyomo model for IES
            time: timestep for time series variables

        Returns:
            pyomo function for h2_power
        '''
        assert self.COPRODUCTION_mode, 'System must include hydrogen to create h2_power constraint'

        return self.h2_power(m.pt_scaled[time]*m.reference_power, m.ht[time])

    def h2_power_bigm_pyomo(self, m, time):
        '''
        Converts h2_power functions from standard variables to pyomo variables

        Arguments:
            self: self
            m: pyomo model for IES
            time: timestep for time series variables

        Returns:
            pyomo function for h2_power
        '''
        assert self.COPRODUCTION_mode, 'System must include hydrogen to create h2_power constraint'

        return self.h2_power_bigm(m.pt_scaled[time]*m.reference_power, m.ht[time], m.POWER_mode[time], m.OFF_mode[time])

    def fuel_cost_power_pyomo(self, m, time):
        '''
        converts fuel_cost function from standard variables to pyomo variables

        Arguments:
            self: self
            m: pyomo model for IES
            time: timestep for time series variables

        Returns:
            pyomo function for fuel_cost
        '''

        return self.fuel_cost_power(m.pt_scaled[time]*m.reference_power)

    def fuel_cost_hydrogen_pyomo(self, m, time):
        '''
        converts fuel_cost function from standard variables to pyomo variables

        Arguments:
            self: self
            m: pyomo model for IES
            time: timestep for time series variables

        Returns:
            pyomo function for fuel_cost
        '''

        return self.fuel_cost_hydrogen(m.pt_scaled[time]*m.reference_power, m.ht[time])

    def fuel_cost_coproduction_pyomo(self, m, time):
        '''
        converts fuel_cost function from standard variables to pyomo variables

        Arguments:
            self: self
            m: pyomo model for IES
            time: timestep for time series variables

        Returns:
            pyomo function for fuel_cost
        '''

        return self.fuel_cost_coproduction(m.pt_scaled[time]*m.reference_power, m.ht[time])

    def electricity_cost_pyomo(self, m, time):
        '''
        converts electricity_cost function from standard variables to pyomo variables

        Arguments:
            self: self
            m: pyomo model for IES
            time: timestep for time series variables
        '''

        return self.electricity_cost_hydrogen(m.ht[time])

    def plot_marginal_cost(self, fdest = None, fdest_h2 = None):
        '''
        Generates marginal cost plot using surrogates. For power only cases, marginal cost vs power output.
        For coproduction, marginal cost surface for varying net power and hydrogen output

        Arguments:
            fdest: string, destination of saved file
                default = None, no file saved, only displayed
            fdest_h2: string, destination of saved file (derivative wrt hydrogen)
                default = None, no file saved or power only

        Returns:
            None
        '''

        if self.HYDROGEN_mode and self.COPRODUCTION_mode:
            # plot coproduction contour plots

            # generate pt and ht values
            pt_vals = np.linspace(self.PowerSpecs.min_plant_power, self.PowerSpecs.up_bound_nameplate, num = 100)
            ht_vals = np.linspace(self.HydrogenSpecs.min_plant_h2, self.HydrogenSpecs.up_bound_nameplate, num = 100)

            # generate x and y tick labels for pt and ht
            tix = np.linspace(0, 100, num = 10)
            ytix = np.linspace(min(ht_vals), max(ht_vals), num = 10)
            xtix = np.linspace(min(pt_vals), max(pt_vals), num = 10)

            xtix_ = np.around(xtix, 1)
            ytix_ = np.around(ytix, 1)

            xlabels = [str(i) for i in xtix_]
            ylabels = [str(j) for j in ytix_]

            # compute derivatives
            pt, ht = sympy.symbols('pt,ht')

            # compute fuel cost derivatives
            fuel_cost_diff_p = sympy.diff(self.fuel_cost_coproduction(pt, ht), pt)
            fuel_cost_diff_h = sympy.diff(self.fuel_cost_coproduction(pt, ht), ht)

            # compute variable cost derivatives
            var_cost_diff_p = sympy.diff(self.var_cost_coproduction(pt, ht), pt)
            var_cost_diff_h = sympy.diff(self.var_cost_coproduction(pt, ht), ht)

            # lambdify functions
            marg_cost_p = sympy.lambdify((pt,ht), (fuel_cost_diff_p + var_cost_diff_p))
            marg_cost_h = sympy.lambdify((pt,ht), (fuel_cost_diff_h + var_cost_diff_h))

            # compute derivatives
            marg_cost_p_ = np.empty((100,100))
            marg_cost_h_ = np.empty((100,100))

            for p in range(len(pt_vals)):
                for h in range(len(ht_vals)):
                    marg_cost_p_[p,h] = marg_cost_p(pt_vals[p], ht_vals[h])*1E6 #M$/MWh
                    marg_cost_h_[p,h] = marg_cost_h(pt_vals[p], ht_vals[h])*(1E6/3600) #M$/kg

            # create mesh grid for plotting
            pt_mesh, ht_mesh = np.meshgrid(pt_vals, ht_vals)

            # generate plot
            fig,ax = plt.subplots(figsize = (6,6))


            plt.imshow(marg_cost_p_, cmap = 'gist_rainbow', interpolation = 'nearest', origin = 'lower', alpha = 0.5)
            contours = plt.contour(marg_cost_p_, levels = 10, colors = 'black', linewidths = 5)
            plt.clabel(contours, inline = True, fontsize = 15)


            # titles and labels
            plt.title('Marginal Cost of Power ($/MWh)', fontsize = 20, fontweight = 'bold')
            ax.set_xticks(tix, xlabels, fontsize = 15, rotation = 45, ha = 'right')
            ax.set_yticks(tix, ylabels, fontsize = 15)
            plt.xlabel('Net Power (MW)', fontsize = 20, fontweight = 'bold')
            plt.ylabel('H$_{2}$ Output (kg/s)', fontsize = 20, fontweight = 'bold')

            ht_1 = lambda ht: ht >= 1.0
            ht_5 = lambda ht: ht <= 5.0
            # plot feasible region
            if len(self.h2_power(pt_mesh, ht_mesh)) == 3:
                con = self.h2_power(pt_mesh, ht_mesh)[0] & self.h2_power(pt_mesh, ht_mesh)[1] & self.h2_power(pt_mesh, ht_mesh)[2] * ht_1(ht_mesh).astype(int)
                plt.imshow(con, origin = 'lower', interpolation = 'nearest', cmap = 'Greys_r', alpha = 0.2, zorder = 2)
            else:
                con = self.h2_power(pt_mesh, ht_mesh)[0] & self.h2_power(pt_mesh, ht_mesh)[1] & ht_1(ht_mesh) & ht_5(ht_mesh).astype(int)
                plt.imshow(con, origin = 'lower', interpolation = 'nearest', cmap = 'Greys_r', alpha = 0.2, zorder = 2)

            # manually plot feasible region based on corners 
            # define the vertices 
            if self.number == 3:
                y = np.array([[-73.8, 3.64], [-49.79, 5.0], [514.37, 1.0], [304.46, 1.0]])

            elif self.number == 5:
                y = np.array([[569.74, 1], [99.85, 1], [-381.99, 5], [1.5, 5.0]])

            else:
                raise NotImplementedError()
            
            # convert points to points on xticks and y ticks 
            for row in y:
                row[0] = (((row[0]-min(pt_vals))*(ax.get_xlim()[1]-ax.get_xlim()[0]))/(max(pt_vals)-min(pt_vals))) + ax.get_xlim()[0]

                row[1] = (((row[1]-min(ht_vals))*(ax.get_ylim()[1]-ax.get_ylim()[0]))/(max(ht_vals)-min(ht_vals))) + ax.get_ylim()[0]


            poly = Polygon(y, edgecolor = 'dimgray', linewidth = 4, fill = False, alpha = 1, zorder = 3)
            ax.add_patch(poly)

            # major and minor ticks
            plt.tick_params(direction = 'in', top = True, right = True)
            plt.minorticks_on()
            plt.tick_params(which = 'minor', direction = 'in', top = True, right = True)

            # save figure
            if fdest is not None:
                plt.savefig((fdest + '.png'), dpi = 300, bbox_inches = 'tight')
                plt.savefig((fdest + '.eps'), dpi = 300, bbox_inches = 'tight')

            plt.show()

            # plot hydrogen graph
            # generate plot
            fig, ax = plt.subplots(figsize = (6,6))
            plt.imshow(marg_cost_h_, cmap = 'gist_rainbow', interpolation = 'nearest', origin = 'lower', alpha = 0.5)
            contours = plt.contour(marg_cost_h_, levels = 10, colors = 'black', linewidths = 5)
            plt.clabel(contours, inline = True, fontsize = 15)

            # titles and labels
            plt.title('Marginal Cost of H$_{2}$ (\$/MWh)', fontsize = 20, fontweight = 'bold')
            ax.set_xticks(tix, xlabels, fontsize = 15, rotation = 45)
            ax.set_yticks(tix, ylabels, fontsize = 15)
            plt.xlabel('Net Power (MW)', fontsize = 20, fontweight = 'bold')
            plt.ylabel('H$_{2}$ Output (kg/s)', fontsize = 20, fontweight = 'bold')

            # plot feasible region
            if self.number == 3:
                con = self.h2_power(pt_mesh, ht_mesh)[0] & self.h2_power(pt_mesh, ht_mesh)[1] & self.h2_power(pt_mesh, ht_mesh)[2] * ht_1(ht_mesh).astype(int)
                plt.imshow(con, origin = 'lower', interpolation = 'nearest', cmap = 'Greys_r', alpha = 0.2, zorder = 2)
            elif self.number == 5:
                con = self.h2_power(pt_mesh, ht_mesh)[0] & self.h2_power(pt_mesh, ht_mesh)[1] & ht_1(ht_mesh) & ht_5(ht_mesh).astype(int)
                plt.imshow(con, origin = 'lower', interpolation = 'nearest', cmap = 'Greys_r', alpha = 0.2, zorder = 2)

            else:
                raise NotImplementedError()

            # manually plot feasible region based on corners 
            # define the vertices 
            if self.number == 3:
                y = np.array([[-73.8, 3.64], [-49.79, 5.0], [514.37, 1.0], [304.46, 1.0]])

            elif self.number == 5:
                y = np.array([[569.74, 1], [99.85, 1], [-381.99, 5], [1.5, 5.0]])

            else:
                raise NotImplementedError()
            
            # convert points to points on xticks and y ticks 
            for row in y:
                row[0] = (((row[0]-min(pt_vals))*(ax.get_xlim()[1]-ax.get_xlim()[0]))/(max(pt_vals)-min(pt_vals))) + ax.get_xlim()[0]

                row[1] = (((row[1]-min(ht_vals))*(ax.get_ylim()[1]-ax.get_ylim()[0]))/(max(ht_vals)-min(ht_vals))) + ax.get_ylim()[0]


            poly = Polygon(y, edgecolor = 'dimgray', linewidth = 4, fill = False, alpha = 1, zorder = 3)
            ax.add_patch(poly)


            # major and minor ticks
            plt.tick_params(direction = 'in', top = True, right = True)
            plt.minorticks_on()
            plt.tick_params(which = 'minor', direction = 'in', top = True, right = True)

            # save figure
            if fdest_h2 is not None:
                plt.savefig((fdest_h2 + '.png'), dpi = 300, bbox_inches = 'tight')
                plt.savefig((fdest_h2 + '.eps'), dpi = 300, bbox_inches = 'tight')

            plt.show()


        elif self.HYDROGEN_mode and not self.COPRODUCTION_mode:
            # produce two plots for cost of hydrogen and cost of power (not contours)

            # generate pt and ht values
            pt_vals = np.linspace(self.PowerSpecs.up_bound_nameplate*self.PowerSpecs.turndown_limit, self.PowerSpecs.up_bound_nameplate, num = 100)
            ht_vals = np.linspace(self.HydrogenSpecs.min_plant_h2, self.HydrogenSpecs.up_bound_nameplate, num = 100)

            pt, ht = sympy.symbols('pt, ht')

            # compute fuel cost derivatives
            fuel_cost_diff_p = sympy.diff(self.fuel_cost_power(pt), pt)
            fuel_cost_diff_h = sympy.diff(self.fuel_cost_hydrogen(pt, ht), ht)

            # electricity cost derivative
            power_cost_diff_h = sympy.diff(self.electricity_cost_hydrogen(ht),ht)

            # variable cost derivatives
            var_cost_diff_p = sympy.diff(self.var_cost_power(pt), pt)
            var_cost_diff_h = sympy.diff(self.var_cost_hydrogen(pt, ht),ht)

            # lambdify functions
            marg_cost_p = sympy.lambdify(pt, (fuel_cost_diff_p + var_cost_diff_p))
            marg_cost_h_30 = sympy.lambdify((pt,ht), (power_cost_diff_h + var_cost_diff_h + fuel_cost_diff_h))
            marg_cost_h_60 = sympy.lambdify((pt,ht), (power_cost_diff_h*(60/30) + var_cost_diff_h + fuel_cost_diff_h))
            marg_cost_h_100 = sympy.lambdify((pt,ht), (power_cost_diff_h*(100/30) + var_cost_diff_h + fuel_cost_diff_h))

            # plot
            fig, ax = plt.subplots(figsize = (6,6))
            plt.plot(pt_vals, marg_cost_p(pt_vals)*1E6, color = 'black', linewidth = 3)
            plt.title('Marginal Cost of Power', fontsize = 20, fontweight = 'bold')
            plt.xlabel('Net Power (MW)', fontsize = 20, fontweight = 'bold')
            plt.ylabel('Marginal Cost ($/MWh)', fontsize = 20, fontweight = 'bold')
            plt.grid(True)

            # major and minor ticks
            plt.tick_params(direction = 'in', top = True, right = True)
            plt.minorticks_on()
            plt.tick_params(which = 'minor', direction = 'in', top = True, right = True)

            # axes labels
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(15)

            # save figure
            if fdest is not None:
                plt.savefig(fdest + '.png', dpi = 300, bbox_inches = 'tight')
                plt.savefig(fdest + '.eps', dpi = 300, bbox_inches = 'tight')

            plt.show()

            fig, ax = plt.subplots(figsize = (6,6))
            plt.plot(ht_vals, np.ones_like(ht_vals)*(marg_cost_h_30(0.0, ht_vals)*(1E6/3600)), color = 'black', linestyle = 'solid', linewidth = 3, label = 'LMP = \$30/MWh')
            plt.plot(ht_vals, np.ones_like(ht_vals)*(marg_cost_h_60(0.0, ht_vals)*(1E6/3600)), color = 'black', linestyle = 'dashed', linewidth = 3, label = 'LMP = \$60/MWh')
            plt.plot(ht_vals, np.ones_like(ht_vals)*(marg_cost_h_100(0.0, ht_vals)*(1E6/3600)), color = 'black', linestyle = 'dotted', linewidth = 3, label = 'LMP = \$100/MWh')
            plt.title('Marginal Cost of H$_{2}$', fontsize = 20, fontweight = 'bold')
            plt.xlabel('H$_{2}$ Output (kg/s)', fontsize = 20, fontweight = 'bold')
            plt.ylabel('Marginal Cost ($/kg)', fontsize = 20, fontweight = 'bold')
            plt.grid(True)
            fig.legend(loc = 'upper right', bbox_to_anchor = (0.9, 0.7), fontsize = 15)

            # major and minor ticks
            plt.tick_params(direction = 'in', top = True, right = True)
            plt.minorticks_on()
            plt.tick_params(which = 'minor', direction = 'in', top = True, right = True)

            # axes labels
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(15)

            # save figure
            if fdest_h2 is not None:
                plt.savefig(fdest_h2 + '.png', dpi = 300, bbox_inches = 'tight')
                plt.savefig(fdest_h2 + '.eps', dpi = 300, bbox_inches = 'tight')

            plt.show()

        elif not self.HYDROGEN_mode:
            # generate pt
            pt_vals = np.linspace(self.PowerSpecs.up_bound_nameplate*self.PowerSpecs.turndown_limit, self.PowerSpecs.up_bound_nameplate, num = 100)

            pt, ht = sympy.symbols('pt,ht')

            # take derivatives
            diff_p = sympy.diff(self.fuel_cost_power(pt), pt) + sympy.diff(self.var_cost_power(pt), pt)

            # lambdify function
            marg_cost = sympy.lambdify((pt), diff_p)

            # generate plot
            fig, ax = plt.subplots(figsize = (6,6))
            plt.plot(pt_vals, marg_cost(pt_vals)*1E6, color = 'black', linewidth = 3)
            plt.title('Marginal Cost of Power', fontsize = 20, fontweight = 'bold')
            plt.xlabel('Net Power (MW)', fontsize = 20, fontweight = 'bold')
            plt.ylabel('Marginal Cost ($/MWh)', fontsize = 20, fontweight = 'bold')
            plt.grid(True)

            # major and minor ticks
            plt.tick_params(direction = 'in', top = True, right = True)
            plt.minorticks_on()
            plt.tick_params(which = 'minor', direction = 'in', top = True, right = True)

            # axes labels
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(15)

            # save figure
            if fdest is not None:
                plt.savefig((fdest + '.png'), format = 'png', dpi = 300, bbox_inches = 'tight')
                plt.savefig((fdest + '.eps'), format = 'eps', dpi = 300, bbox_inches = 'tight')

            plt.show()

        return


class Case0(IESModel):

    def __init__(self):
        '''
        Initialize case 0 model with model modes and power system specs

        Arguments:
            self: self

        Returns:
            none
        '''
        # case number
        self.number = 0

        # model modes
        self.OFF_mode = True
        self.POWER_mode = True
        self.HYDROGEN_mode = False
        self.COPRODUCTION_mode = False
        self.CHARGE_mode = False
        self.DISCHARGE_mode = False

        # power system specs
        self.PowerSpecs = PowerSpecs(650.0, 0.0, 0.5, ramp_rate = 70*60.0)

        return

    def var_cost_power(self, pt):
        '''
        Variable cost function for case 0 - NGCC w/ carbon capture
        Note: in surrogate model report this is "other" variable cost to include price correction for NG

        Arguments:
            pt: power output at time t (MW)

        Returns:
            variable cost (M$/hr)
        '''
        return ((5.391 * pt) - (0.11624023e-2 * pt**2) + (0.78853095302e-6 * pt**3) + 437)/1e6 #MM$/hr

    def fixed_cost(self, P):
        '''
        Fixed cost function for case 0 - NGCC w/ carbon capture

        Arguments:
            P: plant capacity (MW)

        Returns:
            fixed cost (M$/hr)
        '''
        # fixed cost expression
        term_1 = 172.54 * ((P/650)**0.8) #MM$/yr

        return term_1/8760 #$/hr

    def startup_cost(self):
        return 2898.0/1e6 #M$

    def shutdown_cost(self):
        return 176.0/1e6 #M$

    def startup_time(self):
        return 1 #hr

    def shutdown_time(self):
        return 1 #hr

    def fuel_cost_power(self, pt):
        '''
        Variable cost contribution from natural gas with assumed price $4.42/MMBTU

        Arguments:
            pt: net power output at timestep (MW)

        Returns:
            fuel cost (M$/hr)
        '''

        return ((30.285 * pt) - (0.65303505e-2 * pt**2) + (0.44299491743e-5 * pt**3) + 2453)/1e6 #MM$/hr

class Case1(IESModel):

    def __init__(self):
        '''
        Initialize case 1 model with model modes and power system specs

        Arguments:
            self: self

        Returns:
            none
        '''
        # case number
        self.number = 1

        # model modes
        self.OFF_mode = True
        self.POWER_mode = True
        self.HYDROGEN_mode = False
        self.COPRODUCTION_mode = False
        self.CHARGE_mode = False
        self.DISCHARGE_mode = False

        # power system specs
        self.PowerSpecs = PowerSpecs(650.0, 0.0, .25, ramp_rate = 100*60.0)

        return

    def startup_cost(self):
        return 162015.0/1e6 #M$

    def shutdown_cost(self):
        return 89330.0/1e6 #M$

    def startup_time(self):
        return 24 #hrs

    def shutdown_time(self):
        return 36 #hrs

    def var_cost_power(self, pt):
        '''
        Variable cost function for case 1 - SOFC

        Arguments:
            pt: power output at time t (MW)

        Returns:
            variable cost (M$/hr)
        '''
        return ((0.795309 * pt) + (0.1610535e-4 * pt**2) + (0.820813703e-7 * pt**3) + 10.60)/1e6 #MM$/hr

    def fixed_cost(self, P):
        '''
        Fixed cost function for case 1 - SOFC

        Arguments:
            P: plant capacity (MW)

        Returns:
            fixed cost (M$/hr)
        '''
        term_1 = 49.53*(P/650)**0.779 #MM$/yr
        term_2 = 70.37*(P/650)**0.77 #MM$/yr

        #combine to calculate costs
        c_fixed = (term_1 + term_2) #MM$/yr
        return c_fixed/8760 #MM$/hr

    def fuel_cost_power(self, pt):
        '''
        Variable cost contribution from natural gas with assumed price $4.42/MMBTU

        Arguments:
            pt: net power output at timestep (MW)

        Returns:
            fuel cost (M$/hr)
        '''

        return ((22.4980965 * pt) + (0.2714661e-3 * pt**2) + (0.107382231e-5 * pt**3) + 38.617)/1e6 #MM$/hr

class Case2(IESModel):

    def __init__(self):
        '''
        Initializes case 2 model with model modes, power system specs, and storage system specs

        Arguments:
            self: self

        Returns:
            None
        '''
        # case number
        self.number = 2

        # model modes
        self.OFF_mode = True
        self.POWER_mode = False
        self.HYDROGEN_mode = False
        self.COPRODUCTION_mode = False
        self.CHARGE_mode = True
        self.DISCHARGE_mode = True

        # power system specs
        self.PowerSpecs = PowerSpecs(650.0, 190.0, 190/650, ramp_rate = 650.0)

        # storage system specs
        self.StorageSpecs = StorageSpecs(100.0, 0.0, 0.7)

        return

    def var_cost_power(self, P, grid_power, charge):
        '''
        Variable cost function for case 2 - SOFC w/ CAES

        Arguments:
            P:
            pt:
            p_excess:
            soc:

        Returns:
            variable cost ($/hr)
        '''

        return (24.04108 * grid_power) + (31.5294*charge)

    def fixed_cost(self, P, E):
        '''
        Fixed cost function for case 2 - SOFC w/ CAES

        Arguments:
            P: nameplate capacity of SOFC
            E: CAES capacity

        Returns:
            fixed cost ($/hr)
        '''

        term_1 = 7441.8*((P/650)**0.77) #$/hr
        term_2 = 1030.3*(E/100) #$/hr
        term_3 = 5124.4*((P/650)**0.779) #$/hr
        term_4 = 184*(E/100) #$/hr

        return term_1 + term_2 + term_3 + term_4

    def fuel_cost(self, pt):
        '''
        Variable cost contribution from natural gas with assumed price $4.42/MMBTU
        '''

        pass

class Case3(IESModel):

    def __init__(self):
        '''
        Initializes case 3 model with model modes, power system specs, and hydrogen system specs

        Arguments:
            self: self

        Returns:
            none
        '''
        # case number
        self.number = 3

        # model modes
        self.OFF_mode = True
        self.POWER_mode = True
        self.HYDROGEN_mode = True
        self.COPRODUCTION_mode = True
        self.CHARGE_mode = False
        self.DISCHARGE_mode = False

        # power system specs
        self.PowerSpecs = PowerSpecs(650.0, -100.0, 0.25, ramp_rate = 70*60.0)

        # hydrogen system specs
        self.HydrogenSpecs = HydrogenSpecs(5.0, 0.0, 0.25)
        return


    def var_cost_power(self, pt):
        '''
        Variable cost function for case 3 - NGCC w/ SOEC
        Note: power surrogates match NGCC only case

        Arguments:
            pt: power output at time t (MW)

        Returns:
            variable cost (M$/hr)
        '''

        return ((5.391 * pt) - (0.11624023e-2 * pt**2) + (0.78853095302e-6 * pt**3) + 437)/1e6 #MM$/hr

    def var_cost_hydrogen(self, pt, ht):
        '''
        Variable cost function for case 3 - NGCC w/ SOEC

        Arguments:
            pt: power output at time t (MW)
            ht: hydrogen produced at time t (kg/s)

        Returns:
            variable cost (M$/hr)
        '''

        return ((2.0023 * pt) + (350.73 * ht) + (0.3782337e-2 * pt**2) + (56.55358 * ht**2) - (0.181734434e-5 * pt**3) - (2.382281948 * ht **3) + (0.5692029 * pt * ht) + 1229.53)/1e6 #MM$/hr
    def var_cost_coproduction(self, pt, ht):
        '''
        Variable cost function for case 3 - NGCC w/ SOEC

        Arguments:
            pt: power output at time t (MW)
            ht: hydrogen produced at time t (kg/s)

        Returns:
            variable cost (M$/hr)
        '''

        return ((2.0023 * pt) + (350.73 * ht) + (0.3782337e-2 * pt**2) + (56.55358 * ht**2) - (0.181734434e-5 * pt**3) - (2.382281948 * ht **3) + (0.5692029 * pt * ht) + 1229.53)/1e6 #MM$/hr

    def fixed_cost(self, P, H):
        '''
        Fixed cost function for case 3 - NGCC w/ SOEC

        Arguments:
            P: plant capacity (MW)
            H: hydrogen production capacity (kg/hr)

        Returns:
            fixed cost (M$/hr)
        '''
        # calculate terms separately
        term_1 = 145.61*((H/5)**0.8) #MM$/yr
        term_2 = 72.73*((H/5)**0.8) #$MM/yr

        return (term_1 + term_2)/8760 #MM$/hr

    def h2_power(self, pt, ht):
        '''
        Coproduction feasibility constraints for NGCC + SOEC case

        Arguments:
            pt: net power production at timestep (MW)
            ht: hydrogen production at timestep (kg/s)

        Returns:
            List of expressions
        '''


        # define each constraint
        expression1 = (pt >= (15.549 * ht) - 130.94)
        expression2 = (pt >= (-143.31 * ht) + 447.77)
        expression3 = (pt <= (-141.04 * ht) + 655.41)

        return [expression1, expression2, expression3]
    
    def h2_power_values(self, ht):
        '''
        Coproduction feasibility constraints, returns values instead of expressions
        '''

        

        pt1 = (15.549*ht) - 130.94
        pt2 = (-143.31*ht) + 447.77
        pt3 = (-141.04*ht) + 655.41

        return pt1, pt2, pt3
        
        
    def h2_power_bigm(self, pt, ht, ypow, yoff):
        '''
        Coproduction feasibility constraints for NGCC + SOEC case

        Arguments:
            pt: net power production at timestep (MW)
            ht: hydrogen production at timestep (kg/s)

        Returns:
            List of expressions
        '''


        # define each constraint
        expression1 = (pt >= (15.549 * ht) - 130.94 + 54*(ypow + yoff))
        expression2 = (pt >= (-143.31 * ht) + 447.77 - 305*(ypow + yoff))
        expression3 = (pt <= (-141.04 * ht) + 655.41 - 515*(ypow + yoff))

        return [expression1, expression2, expression3]

    def startup_cost(self):
        return 181404.0/1e6 # M$

    def shutdown_cost(self):
        return 98457.0/1e6 # M$

    def startup_time(self):
        return 24 #hrs

    def shutdown_time(self):
        return 36 #hrs

    def fuel_cost_power(self, pt):
        '''
        Variable cost contribution from natural gas with assumed price $4.42/MMBTU

        Arguments:
            pt: net power output at timestep (MW)

        Returns:
            fuel cost (M$/hr)
        '''

        return ((30.285 * pt) - (0.65303505e-2 * pt**2) + (0.44299491743e-5 * pt**3) + 2453)/1e6 #MM$/hr

    def fuel_cost_hydrogen(self, pt, ht):
        '''
        Variable cost contribution from natural gas with assumed price $4.42/MMBTU

        Arguments:
            pt: net power output at timestep (MW)
            ht: hydrogen output at timestep (kg/s)

        Returns:
            fuel cost (M$/hr)
        '''

        return ((11.249 * pt) + (1759.1 * ht) + (0.02124908 * pt**2) + (317.7167 * ht**2) - (0.102097997e-4 * pt**3) - (13.38360646 * ht**3) + (3.197769 * pt * ht) + 6907.51)/1e6 #MM$/hr

    def fuel_cost_coproduction(self, pt, ht):
        '''
        Variable cost contribution from natural gas with assumed price $4.42/MMBTU

        Arguments:
            pt: net power output at timestep (MW)
            ht: hydrogen output at timestep (kg/s)

        Returns:
            fuel cost (M$/hr)
        '''

        return ((11.249 * pt) + (1759.1 * ht) + (0.02124908 * pt**2) + (317.7167 * ht**2) - (0.102097997e-4 * pt**3) - (13.38360646 * ht**3) + (3.197769 * pt * ht) + 6907.51)/1e6 #MM$/hr

class Case4(IESModel):

    def __init__(self):
        '''
        Initialize case 4 model with model modes, power system specs and hydrogen system specs

        Arguments:
            self: self

        Returns:
            none
        '''
        # case number
        self.number = 4

        # model modes
        self.OFF_mode = True
        self.POWER_mode = True
        self.HYDROGEN_mode = True
        self.COPRODUCTION_mode = False
        self.CHARGE_mode = False
        self.DISCHARGE_mode = False

        # power system specs
        self.PowerSpecs = PowerSpecs(650.0, 0.0, 0.25, ramp_rate = 100*60.0)

        # hydrogen system specs
        self.HydrogenSpecs = HydrogenSpecs(5.0, 0.0, 0.25, ramp_rate = 1*60.0)

        return

    def startup_cost(self):
        return 162015.0/1e6 # M$

    def shutdown_cost(self):
        return 89330.0/1e6 # M$

    def startup_time(self):
        return 24 #hrs

    def shutdown_time(self):
        return 36 #hrs

    def fixed_cost(self, P, H):
        '''
        Fixed cost function for case 4 - rSOFC

        Arguments:
            P: plant capacity (MW)
            H: hydrogen capacity (kg/s)

        Returns:
            fixed cost (M$/hr)
        '''
        term_1 = 73.65*(P/650)**0.77 #MM$/yr
        term_2 = 51.69*(P/650)**0.779 #MM$/yr

        #combine to calculate costs
        c_fixed = (term_1 + term_2) #MM$/yr
        return c_fixed/8760 #MM$/hr

    def var_cost_power(self, pt):
        '''
        Variable cost function for case 4 - SOFC mode
        Note: same equations as sofc only

        Arguments:
            pt: power output at time t (MW)

        Returns:
            variable cost (M$/hr)
        '''
        return ((0.795309 * pt) + (0.1610535e-4 * pt**2) + (0.820813703e-7 * pt**3) + 10.60)/1e6 #MM$/hr

    def var_cost_hydrogen(self, pt, ht):
        '''
        Variable cost function for case 4 - SOEC mode

        Arguments:
            pt: power output at time t (MW)
            ht: hydrogen produced at time t (kg/s)

        Returns:
            variable cost (M$/hr)
        '''
        return (23.366 * ht - (0.19545612e-8 * ht**2) + (0.2203012e-9 * ht**3) - 0.4938047e-8)/1e6 #MM$/hr

    def fuel_cost_power(self, pt):
        '''
        Variable cost contribution from natural gas with assumed price $4.42/MMBTU

        Arguments:
            pt: net power output at timestep (MW)

        Returns:
            fuel cost (M$/hr)
        '''

        return ((22.4980965 * pt) + (0.2714661e-3 * pt**2) + (0.107382231e-5 * pt**3) + 38.617)/1e6 #MM$/hr

    def fuel_cost_hydrogen(self, pt, ht):
        '''
        Variable cost contribution from natural gas with assumed price $4.42/MMBTU

        Arguments:
            pt: net power output at timestep (MW)
            ht: hydrogen output at timestep (kg/s)
        '''

        return ((506.88 * ht) + (41.73346 * ht**2) - (4.208053* ht**3) + 86.32)/1e6 #MM$/hr

    def electricity_cost_hydrogen(self, ht):
        '''
        Variable cost contribution from electricity cost with assumed price $30/MWh

        Arguments:
            ht: hydrogen output at timestep (kg/s)

        Returns:
            electricity cost (M$/hr)
        '''

        return (3957.83 * ht)/1e6 #MM$/hr

class Case5(IESModel):

    def __init__(self, safe_divide = False):
        '''
        Initialize case 5 model with model modes, power system specs and hydrogen system specs

        Arguments:
            self: self
            safe_divide: boolean, states whether or not to reformulate the divide by variable
            default = False: do not reformulate, True: reformulate

        Returns:
            none
        '''
        # case number
        self.number = 5

        # model modes
        self.OFF_mode = True
        self.POWER_mode = True
        self.HYDROGEN_mode = True
        self.COPRODUCTION_mode = True
        self.CHARGE_mode = False
        self.DISCHARGE_mode = False

        # power system specs
        self.PowerSpecs = PowerSpecs(712.0, -400, 0.25, ramp_rate = 100*60.0)

        # hydrogen system specs
        self.HydrogenSpecs = HydrogenSpecs(5.0, 0.0, 0.25, ramp_rate = 1*60.0)

        self.safe_divide = safe_divide

        return

    def startup_cost(self):
        return 285146.0/1e6 # M$

    def shutdown_cost(self):
        return 157221.0/1e6 #M$

    def startup_time(self):
        return 24 #hrs

    def shutdown_time(self):
        return 36 #hrs

    def var_cost_coproduction(self, pt, ht):
        '''
        Variable cost function for case 5 - SOFC + SOEC Mode

        Arguments:
            pt: power output at time t (MW)
            ht: hydrogen produced at time t (kg/s)

        Returns:
            variable cost (M$/hr)
        '''
        return ((-1.076292 * pt) + (85.88541 * ht) + (0.562227e-2 * pt**2) - (0.551328e-5 * pt**3) + (1.126261 * ht**3) + (0.427743 * pt*ht) - (0.190403e-3 * (pt*ht)**2) + 205.4984)/1e6 #MM$/hr

    def fixed_cost(self, P, H):
        '''
        Fixed cost function for case 5 - SOFC + SOEC

        Arguments:
            P: plant capacity (MW)
            H: hydrogen capacity (kg/s)

        Returns:
            fixed cost (M$/hr)
        '''
        term_1 = 85.24*(H/5)**0.8 #MM$/yr
        term_2 = 62.65*(H/5)**0.8 #MM$/yr

        #combine to calculate costs
        c_fixed = (term_1 + term_2) #MM$/yr
        return c_fixed/8760 #MM$/hr

    def var_cost_power(self, pt):
        '''
        Variable cost function for case 5 - SOFC Mode

        Arguments:
            pt: power output at time t (MW)
            ht: hydrogen produced at time t (kg/s)

        Returns:
            variable cost (M$/hr)
        '''
        return ((0.795309 * pt) + (0.1610535e-4 * pt**2) + (0.820813703e-7 * pt**3) + 10.60)/1e6 #MM$/hr

    def var_cost_hydrogen(self, pt, ht):
        '''
        Variable cost function for case 5 - SOEC Mode

        Arguments:
            pt: power output at time t (MW)
            ht: hydrogen produced at time t (kg/s)

        Returns:
            variable cost (M$/hr)
        '''
        return ((-1.076292 * pt) + (85.88541 * ht) + (0.562227e-2 * pt**2) - (0.551328e-5 * pt**3) + (1.126261 * ht**3) + (0.427743 * pt*ht) - (0.190403e-3 * (pt*ht)**2) + 205.4984)/1e6 #MM$/hr

    def h2_power(self, pt, ht):
        '''
        Coproduction feasability constraints for SOFC + SOEC case

        Arguments:
            pt: net power production at timestep (MW)
            ht: hydrogen production at timestep (kg/s)

        Returns:
            List of expressions
        '''

        expression1 = (pt <= (-142.06*ht) + 711.8)
        expression2 = (pt >= (-120.46*ht) + 220.31)

        return [expression1, expression2]
    
    def h2_power_values(self, ht):
        '''
        Coproduction feasibility, returns values instead of expressions 

        '''

        pt = np.zeros(2)

        pt[0] = (-142.06*ht) + 711.8
        pt[1] = (-120.46*ht) + 220.31

        return pt

    def h2_power_bigm(self, pt, ht, ypow, yoff):
        '''
        Coproduction feasability constraints for SOFC + SOEC case

        Arguments:
            pt: net power production at timestep (MW)
            ht: hydrogen production at timestep (kg/s)

        Returns:
            List of expressions
        '''

        expression1 = (pt <= (-142.06*ht) + 711.8 - 535*(ypow + yoff))
        expression2 = (pt >= (-120.46*ht) + 220.31 - 70*(ypow + yoff))

        return [expression1, expression2]

    def fuel_cost_power(self, pt):
        '''
        Variable cost contribution from natural gas with assumed price $4.42/MMBTU

        Arguments:
            pt: net power output at timestep (MW)

        Returns:
            fuel cost (M$/hr)
        '''

        return ((22.4980965 * pt) + (0.2714661e-3 * pt**2) + (0.107382231e-5 * pt**3) + 38.617)/1e6 #MM$/hr

    def fuel_cost_coproduction(self, pt, ht):
        '''
        Variable cost contribution from natural ggas with assumed price $4.42/MMBTU

        Arguments:
            pt: net power output at timestep (MW)
            ht: hydrogen output at timestep (kg/s)

        Returns:
            fuel cost (M$/hr)
        '''

        # define safe divide expression
        hmin = (self.HydrogenSpecs.up_bound_nameplate*self.HydrogenSpecs.turndown_limit)
        eps = 1e-5
        if self.safe_divide:
            ht_safe_divide = (0.5*((ht - hmin)**2 + eps**2)**(1/2)) + (0.5*(ht - hmin)) + hmin
        else:
            ht_safe_divide = ht

        # if safe divide selected, replace last ht with safe divide
        return ((4.578016 * pt) + (2685.456 * ht) + (0.526439e-1 * pt**2) - (0.515940e-4 * pt**3) + (12.73327 * ht**3) + (4.057687 * pt*ht)- (0.178612e-2 * (pt*ht)**2) + 1570.231)/1e6 #MM$/hr

    def fuel_cost_hydrogen(self, pt, ht):
        '''
        Variable cost contribution from natural gas with assumed price $4.42/MMBTU
        CHANGE TO LOOK LIKE ABOVE

        Arguments:
            pt: net power output at timestep (MW)
            ht: hydrogen output at timestep (kg/s)

        Returns:
            fuel cost (M$/hr)
        '''

        # define safe divide expression
        hmin = (self.HydrogenSpecs.up_bound_nameplate*self.HydrogenSpecs.turndown_limit)
        eps = 1e-5
        if self.safe_divide:
            ht_safe_divide = (0.5*((ht - hmin)**2 + eps**2)**(1/2)) + (0.5*(ht - hmin)) + hmin
        else:
            ht_save_divide = ht

        return ((4.578016 * pt) + (2685.456 * ht) + (0.526439e-1 * pt**2) - (0.515940e-4 * pt**3) + (12.73327 * ht**3) + (4.057687 * pt*ht)- (0.178612e-2 * (pt*ht)**2) + 1570.231)/1e6 #MM$/hr

class Case6(IESModel):

    def __init__(self):
        '''
        Initialize case 6 model with HYDROGEN specs

        Arguments:
            self: self

        Returns:
            none
        '''
        # case number
        self.number = 6

        # model modes
        self.OFF_mode = True
        self.POWER_mode = False
        self.HYDROGEN_mode = True
        self.COPRODUCTION_mode = False
        self.CHARGE_mode = False
        self.DISCHARGE_mode = False

        # power system specs
        self.HydrogenSpecs = HydrogenSpecs(5.0, 0.0, .25, ramp_rate = 1*60.0)

        return

    def var_cost_hydrogen(self, ht):
        '''
        Variable cost function for case 6 - standalone SOEC
        Note: in surrogate model report this is "other" variable cost to include price correction for NG

        Arguments:
            ht: hydrogen output at time t (kg/s)

        Returns:
            variable cost (M$/hr)
        '''
        return ((421.36905537 * ht) - (80.61056621 * (ht**2)))/1e6 #MM$/hr

    def fixed_cost(self, H):
        '''
        Fixed cost function for case 6 - standalone SOEC

        Arguments:
            H: plant capacity (kg/s)

        Returns:
            fixed cost (M$/hr)
        '''
        # fixed cost expression
        term_1 = 49.38 * ((H/5)**0.8) #MM$/yr
        term_2 = 29.92 * ((H/5)**0.8) #MM$/yr

        return (term_1 + term_2)/8760 #MM$/hr

    def startup_cost(self):
        return 162015.0/1e6 # M$

    def shutdown_cost(self):
        return 89330.0/1e6 # M$

    def startup_time(self):
        return 24 #hrs

    def shutdown_time(self):
        return 36 #hrs

    def electricity_cost_hydrogen(self, ht):
        '''
        Variable cost contribution from natural gas with assumed price $4.42/MMBTU

        Arguments:
            pt: net power output at timestep (MW)

        Returns:
            fuel cost (M$/hr)
        '''

        return ((3998.91869783 * ht) + (120.867811622 * (ht**2)) - (8.86129082 * (ht**3)) + 237.30716883)/1e6 #MM$/hr

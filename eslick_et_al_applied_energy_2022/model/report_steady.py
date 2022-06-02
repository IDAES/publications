import pandas
import pyomo.environ as pyo
import idaes.core.util.tables as tables
from idaes.core.util.misc import svg_tag

def create_tags(m, stream_states):
    """ Create a set of tags with formatting.  These tags can be used to report
    output.  Since a lot of tags are expressions and/or have units that don't
    match the base units of the model.  These tags are only for reporting outputs

    Args:
        m: the plant model
        stream_states: a dictionary of stream states to auto-tag

    Returns:
        (tag dict, tag format dict)
    """
    tags = {} # dict of tags and data to insert into SVG
    tag_formats = {}
    for i, s in stream_states.items(): # Create entires for streams
        tags[i + "_Fmass"] = s.flow_mass
        tag_formats[i + "_Fmass"] = \
            lambda x : "{:.1f} kg/s" if x >= 10 else "{:.2f} kg/s"
        tags[i + "_F"] = s.flow_mol
        tag_formats[i + "_F"] = "{:,.0f} mol/s"
        tags[i + "_T"] = s.temperature
        tag_formats[i + "_T"] = "{:,.0f} K"
        tags[i + "_P_kPa"] = s.pressure/1000
        tag_formats[i + "_P_kPa"] = \
            lambda x : "{:,.0f} kPa" if x >= 100 else "{:.2f} kPa"
        tags[i + "_P"] = s.pressure/1000
        tag_formats[i + "_P"] = "{:,.0f} Pa"
        try:
            tags[i + "_hmass"] = s.enth_mass/1000.0
            tag_formats[i + "_hmass"] = "{:,.0f} kJ/kg"
            tags[i + "_h"] = s.enth_mol
            tag_formats[i + "_h"] = "{:,.0f} J/mol"
        except AttributeError: # flue gas doesn't have enthalpy
            pass
        try:
            tags[i + "_x"] = s.vapor_frac
            tag_formats[i + "_x"] = "{:.3f}"
        except AttributeError: # flue gas doesn't have vapor fraction
            pass
        try:
            tags[i + "_yN2"] = s.mole_frac_comp["N2"]
            tags[i + "_yO2"] =  s.mole_frac_comp["O2"]
            tags[i + "_yNO"] =  s.mole_frac_comp["NO"]
            tags[i + "_yCO2"] =  s.mole_frac_comp["CO2"]
            tags[i + "_yH2O"] =  s.mole_frac_comp["H2O"]
            tags[i + "_ySO2"] = s.mole_frac_comp["SO2"]
        except AttributeError: # Steam doesn't have these
            pass
    # Add some addtional quntities from the model to report
    tags["gross_power"] = -m.fs_main.fs_stc.turb.power[0]
    tags["gross_power_mw"] = -m.fs_main.fs_stc.turb.power[0] * 1e-6
    tag_formats["gross_power_mw"] = "{:.2f} MW"

    tags["thrtl_opening"] = \
        100 * m.fs_main.fs_stc.turb.throttle_valve[1].valve_opening[0]
    tag_formats["thrtl_opening"] = "{:.1f}%"

    tags["coal_hhv"] = m.fs_main.fs_blr.aBoiler.hhv_coal_dry
    tags["coal_hhv_MJ_per_kg"] = m.fs_main.fs_blr.aBoiler.hhv_coal_dry/1e6
    tag_formats["coal_hhv_MJ_per_kg"] = "{:.2f} MJ/kg"
    tags["raw_coal_mass_flow_rate"] = m.fs_main.fs_blr.aBoiler.flowrate_coal_raw[0]
    tag_formats["raw_coal_mass_flow_rate"] = "{:.2f} kg/s"
    tags["coal_moisture_pct"] = m.fs_main.fs_blr.aBoiler.mf_H2O_coal_raw[0]*100
    tag_formats["coal_moisture_pct"] = "{:.2f} %"
    tags["overall_efficiency"] = m.fs_main.overall_efficiency[0]
    tag_formats["overall_efficiency"] = "{:.2f}%"
    tags["overall_heatrate"] = m.fs_main.plant_heat_rate[0]
    tag_formats["overall_heatrate"] = "{:,.0f} BTU/kWh"
    tags["steam_cycle_eff"] = m.fs_main.steam_cycle_eff[0]
    tag_formats["steam_cycle_eff"] = "{:.2f}%"
    tags["boiler_eff"] = 100 * m.fs_main.boiler_heat[0]/(
        m.fs_main.fs_blr.aBoiler.flowrate_coal_raw[0]
        * (1 - m.fs_main.fs_blr.aBoiler.mf_H2O_coal_raw[0])
        * m.fs_main.fs_blr.aBoiler.hhv_coal_dry)
    tag_formats["boiler_eff"] = "{:.2f}%"

    # Pump tags/aux turbines/fans mills...
    tags["bfp_power"] = m.fs_main.fs_stc.bfp.control_volume.work[0]/1000.0
    tag_formats["bfp_power"] = "{:,.0f} kW"
    tags["bfp_eff"] = m.fs_main.fs_stc.bfp.efficiency_isentropic[0]*100
    tag_formats["bfp_eff"] = "{:.2f}%"

    tags["booster_power"] = \
        m.fs_main.fs_stc.booster.control_volume.work[0]/1000.0
    tag_formats["booster_power"] = "{:,.0f} kW"
    tags["booster_eff"] = m.fs_main.fs_stc.booster.efficiency_isentropic[0]*100
    tag_formats["booster_eff"] = "{:.2f}%"

    tags["fwh1_drain_pump_power"] = \
        m.fs_main.fs_stc.fwh1_drain_pump.control_volume.work[0]/1000.0
    tag_formats["fwh1_drain_pump_power"] = "{:,.0f} kW"
    tags["fwh1_drain_pump_eff"] = \
        m.fs_main.fs_stc.fwh1_drain_pump.efficiency_isentropic[0]*100
    tag_formats["fwh1_drain_pump_eff"] = "{:.2f}%"

    tags["cond_pump_power"] = \
        m.fs_main.fs_stc.cond_pump.control_volume.work[0]/1000.0
    tag_formats["cond_pump_power"] = "{:,.0f} kW"
    tags["cond_pump_eff"] = \
        m.fs_main.fs_stc.cond_pump.efficiency_isentropic[0]*100
    tag_formats["cond_pump_eff"] = "{:.2f}%"

    tags["bfp_turb_power"] = \
        m.fs_main.fs_stc.bfp_turb.control_volume.work[0]/1000.0
    tag_formats["bfp_turb_power"] = "{:,.0f} kW"
    tags["bfp_turb_eff"] = \
        m.fs_main.fs_stc.bfp_turb.efficiency_isentropic[0]*100
    tag_formats["bfp_turb_eff"] = "{:.2f}%"

    tags["bfp_turb_os_power"] = \
        m.fs_main.fs_stc.bfp_turb_os.control_volume.work[0]/1000.0
    tag_formats["bfp_turb_os_power"] = "{:,.0f} kW"
    tags["bfp_turb_os_eff"] = \
        m.fs_main.fs_stc.bfp_turb_os.efficiency_isentropic[0]*100
    tag_formats["bfp_turb_os_eff"] = "{:.2f}%"

    tags["bfpt_valve_opening"] = \
        m.fs_main.fs_stc.bfp_turb_valve.valve_opening[0]*100
    tag_formats["bfpt_valve_opening"] = "{:.2f}%"

    return tags, tag_formats


def stream_table(sdict, fname=None):
    """Make a stream table as a Pandas Dataframe, and optionally save it to a
    CSV file.

    Args:
        m: steady state plant model to make a stream table for
        fname: If not none the file to save the CSV file to

    Returns:
        Dataframe
    """
    df = tables.generate_table(
        blocks=sdict,
        attributes=[
            "flow_mass",
            "flow_mol",
            "temperature",
            "pressure",
            "enth_mol",
            "vapor_frac",
            ("flow_mol_comp", "O2"),
            ("flow_mol_comp", "N2"),
            ("flow_mol_comp", "NO"),
            ("flow_mol_comp", "CO2"),
            ("flow_mol_comp", "H2O"),
            ("flow_mol_comp", "SO2"),
        ],
        exception=False, # since there are two property packs with differnt
                         # components, ignore missing indexes
    )
    df.sort_index(inplace=True)
    df.to_csv(fname)
    return df

def create_stream_dict(m):
    """Create a dictionary of stream states.  This mostly uses Arc names,
    but inlet and outlet streams don't have associated Arcs, so those need to
    be set manually.

    """
    streams = tables.arcs_to_stream_dict(
        m.fs_main,
        descend_into=False,
        additional={
            "S002": m.fs_main.fs_stc.turb.inlet_stage[1].inlet,
            "S007": m.fs_main.fs_stc.turb.hp_split[14].outlet_3,
            "S012": m.fs_main.fs_stc.turb.lp_stages[1].inlet,
            "S050b": m.fs_main.fs_stc.condenser_hotwell.makeup,
            "S050": m.fs_main.fs_stc.makeup_valve.inlet,
            "S018": m.fs_main.fs_stc.condenser.tube.properties_in,
            "S020": m.fs_main.fs_stc.condenser.tube.properties_out,
            "S048": m.fs_main.fs_stc.aux_condenser.tube.properties_in,
            "S049": m.fs_main.fs_stc.aux_condenser.tube.properties_out,
            "SA02": m.fs_main.fs_blr.aAPH.side_3_inlet,
            "PA03": m.fs_main.fs_blr.aAPH.side_2_inlet,
            "TA02": m.fs_main.fs_blr.Mixer_PA.TA_inlet,
            "FG06": m.fs_main.fs_blr.aAPH.side_1_outlet,
            "B020": m.fs_main.fs_blr.blowdown_split.FW_Blowdown,
        }
    )
    tables.arcs_to_stream_dict(m.fs_main.fs_blr, s=streams)
    tables.arcs_to_stream_dict(m.fs_main.fs_stc, s=streams, descend_into=False)
    tables.arcs_to_stream_dict(m.fs_main.fs_stc.turb, s=streams)
    tables.arcs_to_stream_dict(m.fs_main.fs_stc.fwh1, s=streams, prepend="fwh1")
    tables.arcs_to_stream_dict(m.fs_main.fs_stc.fwh2, s=streams, prepend="fwh2")
    tables.arcs_to_stream_dict(m.fs_main.fs_stc.fwh3, s=streams, prepend="fwh3")
    tables.arcs_to_stream_dict(m.fs_main.fs_stc.fwh5, s=streams, prepend="fwh5")
    tables.arcs_to_stream_dict(m.fs_main.fs_stc.fwh6, s=streams, prepend="fwh6")

    return tables.stream_states_dict(streams, 0)

def svg_output(tags, tag_format, n=""):
    with open("pfd/plant_template.svg", "r") as f:
        svg_tag(
            svg=f,
            tags=tags,
            outfile=f"results/plant_pfd{n}.svg",
            tag_format=tag_format)
    with open("pfd/boiler_template.svg", "r") as f:
        svg_tag(
            svg=f,
            tags=tags,
            outfile=f"results/boiler_pfd{n}.svg",
            tag_format=tag_format)
    with open("pfd/fwh_hp_template.svg", "r") as f:
        svg_tag(
            svg=f,
            tags=tags,
            outfile=f"results/fwh_hp_pfd{n}.svg",
            tag_format=tag_format)
    with open("pfd/fwh_lp_template.svg", "r") as f:
        svg_tag(
            svg=f,
            tags=tags,
            outfile=f"results/fwh_lp_pfd{n}.svg",
            tag_format=tag_format)

def turbine_table(m, fname):
    stages = [m.fs_main.fs_stc.turb.inlet_stage[1]]
    stages += [x for x in m.fs_main.fs_stc.turb.hp_stages.values()]
    stages += [x for x in m.fs_main.fs_stc.turb.ip_stages.values()]
    stages += [x for x in m.fs_main.fs_stc.turb.lp_stages.values()]
    stages.append(m.fs_main.fs_stc.turb.outlet_stage)

    df = pandas.DataFrame(
        columns=[
            "F (kg/s)",
            "Efficency (%)",
            "Pressure Ratio",
            "Power (MW)",
            "T in (K)",
            "T out (K)",
            "P in (kPa)",
            "P out (kPa)",
            "h in (kJ/kg)",
            "h out (kJ/kg)",
            "x in (%)",
            "x out (%)"
        ],
        index=[x.getname() for x in stages]
    )

    for x in stages:
        n = x.getname()
        row = df.loc[n]
        row["F (kg/s)"] = pyo.value(x.control_volume.properties_in[0].flow_mass)
        row["Efficency (%)"] = pyo.value(x.efficiency_isentropic[0])
        row["Pressure Ratio"] = pyo.value(x.ratioP[0])
        row["Power (MW)"] = pyo.value(x.power_shaft[0]/1e6)
        row["T in (K)"] = pyo.value(x.control_volume.properties_in[0].temperature)
        row["P in (kPa)"] = pyo.value(x.control_volume.properties_in[0].pressure/1000)
        row["h in (kJ/kg)"] = pyo.value(x.control_volume.properties_in[0].enth_mass/1000)
        row["x in (%)"] = pyo.value(x.control_volume.properties_in[0].vapor_frac*100)
        row["T out (K)"] = pyo.value(x.control_volume.properties_out[0].temperature)
        row["P out (kPa)"] = pyo.value(x.control_volume.properties_out[0].pressure/1000)
        row["h out (kJ/kg)"] = pyo.value(x.control_volume.properties_out[0].enth_mass/1000)
        row["x out (%)"] = pyo.value(x.control_volume.properties_out[0].vapor_frac*100)

    df.to_csv(fname)
    return df

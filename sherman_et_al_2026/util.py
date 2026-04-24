from pyomo.common.collections import ComponentMap

def fix_vars(var_list):
    flags = ComponentMap()
    for var in var_list:
        if var.is_indexed():
            for subvar in var.values():
                flags[subvar] = subvar.fixed
                subvar.fix()
        else:
            flags[var] = var.fixed
            var.fix()
    return flags

def restore_fixedness(flags):
    for var, fix in flags.items():
        if fix:
            var.fix()
        else:
            var.unfix()
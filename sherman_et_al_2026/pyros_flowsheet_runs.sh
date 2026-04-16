#!/bin/bash
#
# defaults
process_solve_results="True"
co2_capture_targets="90 92.5 95 96 97 98 99 99.3 99.5"
pyros_conf_lvls="90 95 99"
include_heatmaps="False"

function show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run flowsheet model optimization studies."
    echo ""
    echo "Options:"
    echo "  -h  Display this help message"
    echo "  -d  Output directory. If the directory itself does not exist,"
    echo "      then it is automatically created, provided all parent"
    echo "      directories exist."
    echo "  -t  [Optional] CO2 capture targets of interest (wrap in quotations marks)"
    echo "      (default: '$co2_capture_targets')"
    echo "  -l  [Optional] PyROS confidence levels of interest (wrap in quotation marks)"
    echo "      (default: '$pyros_conf_lvls')"
    echo "  -p  [Optional] True to include heatmap (i.e., operability assessment)"
    echo "      runs in studies, False otherwise (default: $include_heatmaps)"
    echo "  -s  [Optional] True to compile solve results after all solves are complete,"
    echo "      False otherwise (default: $process_solve_results)"
    echo ""
    echo "NOTE: IPOPT and GAMS distributions must be installed and added "
    echo "to your system PATH, or else an error is raised."
    echo ""
}

# get options
while getopts d:l:p:s:t:h flag
do
    case $flag in
        # (mandatory) output directory name suffix
        # recommended: ISO format date
        d) outdir=${OPTARG};;
        l) pyros_conf_lvls=${OPTARG};;
        p) include_heatmaps=${OPTARG};;
        t) co2_capture_targets=${OPTARG};;
        s) process_solve_results=${OPTARG};;
        h) show_help; exit 0;;
        \?) exit 1;;
    esac
done

script_name="run_combined_flowsheet_pyros_econ_obj.py"

if [ "$outdir" == "" ]; then
    echo "Option -d (output directory) not provided"
    exit 1
fi

if [ "$pyros_conf_lvls" == "" ]; then
    echo "No PyROS uncertainty set confidence levels were provided"
    exit 1
fi

if [ "$co2_capture_targets" == "" ]; then
    echo "No CO2 capture targets were provided"
    exit 1
fi

if [ "$include_heatmaps" != "True" ] && [ "$include_heatmaps" != "False" ]; then
    echo "Argument -p must be 'True' or 'False'; instead got '$include_heatmaps'"
    exit 1
fi

if [ "$process_solve_results" != "True" ] && [ "$process_solve_results" != "False" ]; then
    echo "Argument -s must be 'True' or 'False'; instead got '$process_solve_results'"
    exit 1
fi

# check GAMS executable on path
if command -v gams &> /dev/null; then
  echo "Command 'gams' exists at path '$(command -v gams)'"
  # Execute code that depends on the command
else
  echo "Command 'gams' does not exist."
  echo "Ensure GAMS is installed and added to your system path."
  exit 1
fi

# check IPOPT executable on path
if command -v ipopt &> /dev/null; then
  echo "Command 'ipopt' exists at path '$(command -v ipopt)'"
  # Execute code that depends on the command
else
  echo "Command 'ipopt' does not exist."
  echo "Ensure IPOPT is installed and added to your system path."
  exit 1
fi

echo "Summary of resolved options:"
echo "  Outdir: $outdir"
echo "  CO2 capture targets: $co2_capture_targets"
echo "  PyROS uncertainty set confidence levels: $pyros_conf_lvls"
echo "  Include heatmaps: $include_heatmaps"
echo "  Process solve results when all done: $process_solve_results"

read -rp "Continue? (Y/N): " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1

# run script to get nominally optimal solutions
deterministic_cmd=(
    "python $script_name"
    "--workflow deterministic"
    "--co2-capture-targets $co2_capture_targets"
    "--results-dir $outdir"
    "--with-postsolve-costing True"
)
screen -S pyros_mea_deterministic -d -m bash -c "${deterministic_cmd[*]}"

# prevent simultaneous startups so that first screen
# is essentially guaranteed to create output directory
sleep 2

# MONTE CARLO (HEATMAP) WORKFLOWS
# evaluate nominally optimal solutions
if [ "$include_heatmaps" == "True" ]; then
    for targ in $co2_capture_targets; do
        thecmd=(
            "python $script_name"
            "--workflow deterministic_heatmaps"
            "--co2-capture-targets $targ"
            "--results-dir $outdir"
            "--export-nonoptimal-heatmap-models False"
            "--with-postsolve-costing True"
        )
        screen -S "pyros_mea_heatmap_${targ}" -d -m bash -c "${thecmd[*]}"
    done

    # wait until these are done before running PyROS
    while screen -list | grep -q "pyros_mea_heatmap_.*"
    do
        sleep 1
    done
fi

# solve with PyROS
for targ in $co2_capture_targets; do
    thecmd=(
        "python $script_name"
        "--workflow pyros"
        "--co2-capture-targets $targ"
        "--pyros-confidence-levels $pyros_conf_lvls"
        "--results-dir $outdir"
        "--include-pyros-heatmaps $include_heatmaps"
        "--with-postsolve-costing True"
        "--export-nonoptimal-heatmap-models False"
        "--pyros-dr-order 0"
    )
    screen -S "pyros_mea${targ}" -d -m bash -c "${thecmd[*]}"
done

# wait until all solves are complete before processing the results
while screen -list | grep -q "pyros_mea.*"
do
    sleep 1
done

# process solve results
if [ "$process_solve_results" == "True" ]; then
    thecmd=(
        "python $script_name"
        "--workflow solve_results"
        "--results-dir $outdir"
    )
    screen -S "pyros_mea_solve_results" -d -m bash -c "${thecmd[*]}"
fi

# process heatmap results
if [ "$process_solve_results" == "True" ] && [ "$include_heatmaps" == "True" ]; then
    thecmd=(
        "python $script_name"
        "--workflow all_heatmap_results"
        "--results-dir $outdir"
    )
    screen -S "pyros_meat_heatmap_results" -d -m bash -c "${thecmd[*]}"
fi

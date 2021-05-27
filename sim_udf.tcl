open_project udf_op_hls
set_top krnl_udf_selection
add_files src/krnl_udf_selection.cpp
# add_files src/krnl_udf_selection.h
# add_files src/types.h
# add_files src/hbm_column.hpp
add_files -tb src/udf_selection.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1" -flow_target vitis
set_part {xcu280-fsvh2892-2l-e}
# create_clock -period 10 -name default
csim_design -argv {16384 100 10000}
# csynth_design
# cosim_design
# export_design -format ip_catalog
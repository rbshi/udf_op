import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import csv
import math
import colorsys
import sys

# main
def main():
    with open('krnl_udf_selection.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader)
        for row in spamreader:
            [idx, num_kernel, hbm_grouping, auto_alloc] = map(int, row)
            cfg_filename = "krnl_udf_selection_" + str(idx) + ".cfg"
            cfgfile = open(cfg_filename, 'w')
            cfgfile.write("[connectivity]\n")
            for idx_kernel in range(num_kernel):
                wr_line = "sp=krnl_udf_selection_" + str(idx_kernel+1) + ".p_hbm:HBM[0:31]\n"
                cfgfile.write(wr_line)
            cfgfile.write("nk=krnl_udf_selection:" + str(num_kernel) + "\n")
            if auto_alloc != '0':
                print("add SLR allocation here")

            cfgfile.write("[vivado]\n")
            vivado_cfg_lines = \
                ["param=route.enableGlobalHoldIter=true\n", \
                 "param=project.writeIntermediateCheckpoints=false\n", \
                 "prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=SSI_SpreadLogic_high\n", \
                 "prop=run.impl_1.{STEPS.PHYS_OPT_DESIGN.IS_ENABLED}=true\n", \
                 "prop=run.impl_1.{STEPS.PHYS_OPT_DESIGN.ARGS.MORE OPTIONS}={-slr_crossing_opt -tns_cleanup}\n", \
                 "prop=run.impl_1.{STEPS.ROUTE_DESIGN.ARGS.MORE OPTIONS}={-ultrathreads}\n", \
                 "prop=run.impl_1.{STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED}=true\n", \
                 "prop=run.impl_1.{STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.MORE OPTIONS}={-critical_cell_opt -rewire -hold_fix -sll_reg_hold_fix -retime}"]
            cfgfile.writelines(vivado_cfg_lines)

            cfgfile.close()


if __name__ == '__main__':
	main()

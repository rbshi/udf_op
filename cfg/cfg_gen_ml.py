import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import csv
import math
import colorsys
import numpy as np
import csv
from eda import kernel_alloc
import math
import sys

# main
def main():
    kernel_name = "krnl_udf_ml"
    with open(kernel_name + '.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader)
        for row in spamreader:
            [idx, num_kernel, hbm_grouping, auto_alloc] = map(int, row[0:4])
            cfg_filename = kernel_name + "_" + str(idx) + ".cfg"
            cfgfile = open(cfg_filename, 'w')
            cfgfile.write("[connectivity]\n")
            for idx_kernel in range(num_kernel):
                cfgfile.write("sp=" + kernel_name + "_" + str(idx_kernel + 1) + ".p_hbm:HBM[" + str(2 * idx_kernel) + ":" + str(2 * idx_kernel + 1) + "]\n")
            cfgfile.write("nk=" + kernel_name + ":" + str(num_kernel) + "\n")
            if auto_alloc:
                slr_dense = float(row[4])
                n_slr = kernel_alloc(kernel_name, num_kernel, slr_dense)
                n_slr_presum = np.cumsum(n_slr)

                for idx_kernel in range(num_kernel):
                    compare_flag = np.tile(idx_kernel, n_slr_presum.shape) < n_slr_presum
                    i_slr = np.where(compare_flag == True)[0][0]
                    wr_line = "slr=" + kernel_name + "_" + str(idx_kernel+1) + ":SLR" + str(i_slr) + "\n"
                    cfgfile.write(wr_line)

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

import numpy as np
import sys

# kernel auto allocation
def kernel_alloc(kernel_name, num_kernel, slr_dense, hmss_path_per_kernel=1):

    # const, resource of each SLR (CLB_LUT, CLB_REG, BRAM, URAM, DSP)
    slr_resource = np.array([
        [439076, 879484, 488, 320, 2733],
        [431922, 863836, 487, 320, 2877],
        [431869, 864105, 512, 320, 2880]
    ])

    hmss_path_unit_resource = np.array([
        # CLB_LUT
        [[2057, 790, 1200],
         [0, 2315,0],
         [0, 0, 2577]],
        # CLB_REG
        [[3731, 2950, 2950],
         [0, 5427, 2500],
         [0, 0, 5421]]]
    )

    hmss_path_switch = np.array([
        [0, 0],
        [7438, 17034],
        [0, 0]
    ])

    # const, single kernel resource (e.g., from HLS, or vivado->kernel.rpt)
    kernel_resource={
        "krnl_udf_selection":[8043,15352,15,16,0],
        "krnl_udf_olap": [45828,48938,63,16,0],
        "krnl_udf_ml": [44901,54673,23,40,205]
    }

    # initial value - assume all kernels are allocated in slr0
    n_slr = np.array([num_kernel, 0, 0])
    slr_resource_meet = [False, False, False]

    # relax the slr_dense of SLR1/2 to N times of SLR0
    slr_dense_factor = 1.6

    # scheme: move a kernel to the next slr, if resource is NOT meet
    for i_slr in range(3):
        if i_slr > 0:
            slr_dense_multi = slr_dense * slr_dense_factor
        else:
            slr_dense_multi = slr_dense

        while ~slr_resource_meet[i_slr]:
            hmss_path_clb = np.transpose(np.dot(hmss_path_unit_resource, n_slr * hmss_path_per_kernel))
            hmss_path_all = hmss_path_clb + hmss_path_switch
            kernel_all = np.transpose(np.outer(np.array(kernel_resource[kernel_name]), n_slr))
            dynamic_all = kernel_all + np.concatenate((hmss_path_all, np.zeros([3, 3], dtype=int)), axis=1)
            static_per_slr = np.array([31093, 42921, 0, 0, 0])
            resource_all = dynamic_all + np.tile(static_per_slr, [3,1])

            slr_resource_meet[i_slr] = (resource_all[i_slr, 0] < slr_resource[i_slr, 0] * slr_dense_multi) & \
                                       (resource_all[i_slr, 1] < slr_resource[i_slr, 1] * slr_dense_multi) & \
                                       (resource_all[i_slr, 2] <= slr_resource[i_slr, 2]) & \
                                       (resource_all[i_slr, 3] <= slr_resource[i_slr, 3]) & \
                                       (resource_all[i_slr, 4] <= slr_resource[i_slr, 4])

            # last slr, but CANNOT meet
            if (i_slr == 2) & ~slr_resource_meet[i_slr]:
                print("[Error] Kernel allocation CANNOT meet the hardware resoruce constraint.")
                sys.exit(1)
                return n_slr

            if ~slr_resource_meet[i_slr]:
                n_slr[i_slr] -= 1; n_slr[i_slr+1] += 1
    return n_slr
#ifndef _KERNEL_UDF_ML_H_
#define _KERNEL_UDF_ML_H_

#include <hls_stream.h>
#include <ap_int.h>
#include "types.h"

#define MODEL_BITS 48
#define PARALLELISM 16
#define LOG2_FIXED_SCALE 35
const ap_uint<MODEL_BITS> FIXED_SCALE = ((ap_uint<64>)1 << LOG2_FIXED_SCALE);
#define MAX_DIMENSIONALITY 2048/PARALLELISM
#define LOG2_MAX_PARTITION_SIZE 16
#define MAX_PARTITION_SIZE (1 << LOG2_MAX_PARTITION_SIZE)

#define LOG2_BYTES_IN_LINE              6
#define LOG2_BYTES_IN_HBM_LINE          6
#define LOG2_BYTES_IN_FLOAT             2
#define LOG2_BYTES_IN_INT64             3

#define BITS_IN_LINE        (1 << LOG2_BYTES_IN_LINE)*8
#define BITS_IN_HBM_LINE    (1 << LOG2_BYTES_IN_HBM_LINE)*8
#define BITS_IN_FLOAT       (1 << LOG2_BYTES_IN_FLOAT)*8
#define BITS_IN_INT64       (1 << LOG2_BYTES_IN_INT64)*8

#define BYTES_IN_LINE       (1 << LOG2_BYTES_IN_LINE)
#define BYTES_IN_HBM_LINE   (1 << LOG2_BYTES_IN_HBM_LINE)
#define BYTES_IN_FLOAT      (1 << LOG2_BYTES_IN_FLOAT)
#define BYTES_IN_INT64      (1 << LOG2_BYTES_IN_INT64)

#define LOG2_FLOATS_IN_LINE     (LOG2_BYTES_IN_LINE - LOG2_BYTES_IN_FLOAT)
#define FLOATS_IN_LINE          (1 << LOG2_FLOATS_IN_LINE)

#define LOG2_FLOATS_IN_HBM_LINE (LOG2_BYTES_IN_HBM_LINE - LOG2_BYTES_IN_FLOAT)
#define FLOATS_IN_HBM_LINE      (1 << LOG2_FLOATS_IN_HBM_LINE)

#define LOG2_INT64_IN_LINE      (LOG2_BYTES_IN_LINE - LOG2_BYTES_IN_INT64)
#define INT64_IN_LINE           (1 << LOG2_INT64_IN_LINE)

#define LOG2_INT64_IN_HBM_LINE  (LOG2_BYTES_IN_HBM_LINE - LOG2_BYTES_IN_INT64)
#define INT64_IN_HBM_LINE       (1 << LOG2_INT64_IN_HBM_LINE)


void krnl_udf_ml(
    hbm_t* p_hbm,
    addr_t samples_addr,
    addr_t labels_addr,
    addr_t model_addr,
    float step_size,
    float lambda,
    bool do_logreg,
    unsigned minibatch_size,
    unsigned num_epochs,
    unsigned num_features_in_lines,
    unsigned num_samples);


#endif
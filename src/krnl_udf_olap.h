#define AP_INT_MAX_W 2048

#include <hls_stream.h>
#include <ap_int.h>
#include "types.h"

#define LOG2_HASH_TABLE_SIZE    12
#define HASH_TABLE_SIZE         (1 << LOG2_HASH_TABLE_SIZE)
#define PROBE_PARALLELISM       16

#define LOG2_MAX_PARTITIONING_FANOUT    9
#define MAX_PARTITIONING_FANOUT         (1 << LOG2_MAX_PARTITIONING_FANOUT)

#define LOG2_BYTES_IN_LINE              6
#define LOG2_BYTES_IN_HBM_LINE          6
#define LOG2_BYTES_IN_WORD              2
#define LOG2_BYTES_IN_TUPLE             3

#define BITS_IN_LINE        (1 << LOG2_BYTES_IN_LINE)*8
#define BITS_IN_HBM_LINE    (1 << LOG2_BYTES_IN_HBM_LINE)*8
#define BITS_IN_WORD        (1 << LOG2_BYTES_IN_WORD)*8
#define BITS_IN_TUPLE       (1 << LOG2_BYTES_IN_TUPLE)*8

#define BYTES_IN_LINE       (1 << LOG2_BYTES_IN_LINE)
#define BYTES_IN_HBM_LINE   (1 << LOG2_BYTES_IN_HBM_LINE)
#define BYTES_IN_WORD       (1 << LOG2_BYTES_IN_WORD)
#define BYTES_IN_TUPLE      (1 << LOG2_BYTES_IN_TUPLE)

#define LOG2_WORDS_IN_LINE      (LOG2_BYTES_IN_LINE - LOG2_BYTES_IN_WORD)
#define LOG2_TUPLES_IN_LINE     (LOG2_BYTES_IN_LINE - LOG2_BYTES_IN_TUPLE)
#define WORDS_IN_LINE           (1 << LOG2_WORDS_IN_LINE)
#define TUPLES_IN_LINE          (1 << LOG2_TUPLES_IN_LINE)

#define LOG2_WORDS_IN_HBM_LINE      (LOG2_BYTES_IN_HBM_LINE - LOG2_BYTES_IN_WORD)
#define LOG2_TUPLES_IN_HBM_LINE     (LOG2_BYTES_IN_HBM_LINE - LOG2_BYTES_IN_TUPLE)
#define WORDS_IN_HBM_LINE           (1 << LOG2_WORDS_IN_HBM_LINE)
#define TUPLES_IN_HBM_LINE          (1 << LOG2_TUPLES_IN_HBM_LINE)

void krnl_udf_olap(
    hbm_t *p_hbm_in, hbm_t *p_hbm_out,
    const addr_t status_addr,
    const addr_t l_addr, unsigned l_num_lines,
    const addr_t r_addr, unsigned r_num_lines,
    const addr_t out_l_addr, const addr_t out_r_addr, unsigned max_out_lines,
    const addr_t out_repeat_addr,
    unsigned r_index_offset,
    bool build_r, bool handle_collisions);
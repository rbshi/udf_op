
#include "krnl_udf_selection.h"

inline void select_write_hbm_line(hbm_t *p_hbm, addr_t hbm_addr,
                                  ap_uint<BITS_IN_LINE> line) {
  p_hbm[hbm_addr] = line(BITS_IN_HBM_LINE - 1, 0);
}

inline void select_write_dram_line(dram_t *p_dram, addr_t dram_addr,
                                  ap_uint<BITS_IN_LINE> line) {
  p_dram[dram_addr] = line(BITS_IN_HBM_LINE - 1, 0);
}


void select_read_hbm_to_stream(hls::stream<ap_uint<BITS_IN_LINE>> &out_stream,
                               const hbm_t *p_hbm, addr_t hbm_addr,
                               unsigned size_in_lines) {
  ap_uint<BITS_IN_LINE> temp = 0;
  unsigned k = 0;
  while (k < size_in_lines) {
#pragma HLS PIPELINE II = 1
    temp(BITS_IN_HBM_LINE - 1, 0) = p_hbm[hbm_addr + k];
    out_stream.write(temp);
    k++;
  }
}

void select_write_stream_to_hbm(hls::stream<ap_uint<BITS_IN_LINE>> &in_stream,
                                hbm_t *p_hbm, addr_t hbm_addr,
                                unsigned size_in_lines) {
  ap_uint<BITS_IN_LINE> temp = 0;
  unsigned k = 0;
  while (k < size_in_lines) {
#pragma HLS PIPELINE II = 1
    temp = in_stream.read();
    p_hbm[hbm_addr + k] = temp(BITS_IN_HBM_LINE - 1, 0);
    k++;
  }
}

void select_write_stream_to_dram(hls::stream<ap_uint<BITS_IN_LINE>> &in_stream,
                                dram_t *p_dram, addr_t dram_addr,
                                unsigned size_in_lines) {
  ap_uint<BITS_IN_LINE> temp = 0;
  unsigned k = 0;
  while (k < size_in_lines) {
#pragma HLS PIPELINE II = 1
    temp = in_stream.read();
    p_dram[dram_addr + k] = temp(BITS_IN_HBM_LINE - 1, 0);
    k++;
  }
}

void select_core(hls::stream<ap_uint<BITS_IN_LINE>> &in_stream,
                 unsigned indexes[PARALLELISM][BUFFER_SIZE],
                 unsigned fill_state[PARALLELISM], unsigned &offset,
                 unsigned num_lines, unsigned &positives, int lower,
                 int upper) {
  unsigned fill_state_here[PARALLELISM];

  for (unsigned p = 0; p < PARALLELISM; p++) {
#pragma HLS UNROLL
    fill_state_here[p] = 0;
  }

  for (unsigned i = 0; i < num_lines; i++) {
#pragma HLS PIPELINE II = 1
    ap_uint<BITS_IN_LINE> temp = in_stream.read();

    for (unsigned p = 0; p < PARALLELISM; p++) {
#pragma HLS UNROLL
      int value = (int)temp((p + 1) * BITS_IN_INT - 1, p * BITS_IN_INT);
      if (value > lower && value < upper) {
        // cout << "value match: " << value << endl;
        indexes[p][fill_state_here[p]] = offset + p;
        fill_state_here[p]++;
      }
    }

    offset += PARALLELISM;
  }

  for (unsigned p = 0; p < PARALLELISM; p++) {
#pragma HLS UNROLL
    // cout << "fill_state_here[" << p << "]: " << fill_state_here[p] << endl;
    positives += fill_state_here[p];
    fill_state[p] = fill_state_here[p];
  }
}

void select_pipeline(const hbm_t *p_hbm, addr_t input_addr,
                     unsigned num_to_process_lines,
                     unsigned indexes[PARALLELISM][BUFFER_SIZE],
                     unsigned fill_state[PARALLELISM], unsigned &offset,
                     unsigned &positives, int lower, int upper) {
#pragma HLS DATAFLOW
  hls::stream<ap_uint<BITS_IN_LINE>> in_stream("in_stream");
  select_read_hbm_to_stream(in_stream, p_hbm, input_addr, num_to_process_lines);
  select_core(in_stream, indexes, fill_state, offset, num_to_process_lines,
              positives, lower, upper);
}

void gather_all(hls::stream<ap_uint<BITS_IN_LINE>> &out_stream,
                unsigned indexes[PARALLELISM][BUFFER_SIZE],
                unsigned fill_state[PARALLELISM], unsigned max_fill_state) {
  ap_uint<BITS_IN_LINE> result = -1;

  for (unsigned i = 0; i < max_fill_state; i++) {
#pragma HLS PIPELINE II = 1
    for (unsigned p = 0; p < PARALLELISM; p++) {
#pragma HLS UNROLL
      if (i < fill_state[p]) {
        result((p + 1) * BITS_IN_INT - 1, p * BITS_IN_INT) = indexes[p][i];
      } else {
        result((p + 1) * BITS_IN_INT - 1, p * BITS_IN_INT) = 0xFFFFFFFF;
      }
    }
    out_stream.write(result);
  }
}

void write_pipeline(dram_t *p_dram, addr_t output_addr,
                    unsigned indexes[PARALLELISM][BUFFER_SIZE],
                    unsigned fill_state[PARALLELISM], unsigned max_fill_state) {
#pragma HLS DATAFLOW
  hls::stream<ap_uint<BITS_IN_LINE>> out_stream("out_stream");
  gather_all(out_stream, indexes, fill_state, max_fill_state);
  select_write_stream_to_dram(out_stream, p_dram, output_addr, max_fill_state);
}

inline unsigned CEILING(unsigned value, unsigned log2_divider) {
  return (value >> log2_divider) +
         (((value & ((1 << log2_divider) - 1)) > 0) ? 1 : 0);
}


void krnl_udf_selection(const hbm_t *p_hbm, //read-only hbm (can save datapath?)
                        dram_t* p_dram,
                        const addr_t input_addr, 
                        const addr_t output_addr,
                        const addr_t status_addr, 
                        const unsigned num_in_lines, 
                        const int lower,
                        const int upper) {

#pragma HLS INTERFACE m_axi port = p_hbm offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = p_dram offset = slave bundle = gmem1

#pragma HLS INTERFACE s_axilite port = p_hbm
#pragma HLS INTERFACE s_axilite port = p_dram  
#pragma HLS INTERFACE s_axilite port = input_addr
#pragma HLS INTERFACE s_axilite port = output_addr
#pragma HLS INTERFACE s_axilite port = status_addr
#pragma HLS INTERFACE s_axilite port = num_in_lines
#pragma HLS INTERFACE s_axilite port = lower
#pragma HLS INTERFACE s_axilite port = upper
#pragma HLS INTERFACE s_axilite port = return

  unsigned positives = 0;
  unsigned num_out_lines = 0;

  // FIXME: user uram as index buffer, now BUFFER_SIZE=1024, x32=32Kb, << uram
  static unsigned indexes[PARALLELISM][BUFFER_SIZE];
#pragma HLS RESOURCE variable = indexes core = XPM_MEMORY uram
#pragma HLS ARRAY_PARTITION variable = indexes dim = 0 complete

  // fill_state will be registers
  static unsigned fill_state[PARALLELISM];
#pragma HLS ARRAY_PARTITION variable = fill_state dim = 0 complete
  unsigned max_fill_state = 0;

  unsigned num_iterations = CEILING(num_in_lines, LOG2_BUFFER_SIZE);

  // cout << "select_main, num_iterations: " << num_iterations << endl;

  unsigned offset = 0;
  unsigned num_processed_lines = 0;
  for (unsigned i = 0; i < num_iterations; i++) {

    unsigned num_to_process_lines = BUFFER_SIZE;
    if (i == num_iterations - 1) {
      num_to_process_lines = num_in_lines - num_processed_lines;
    }

    // cout << "select_main, num_to_process_lines: " << num_to_process_lines <<
    // endl;

    for (unsigned p = 0; p < PARALLELISM; p++) {
#pragma HLS UNROLL
      fill_state[p] = 0;
    }

    select_pipeline(p_hbm, input_addr + num_processed_lines,
                    num_to_process_lines, indexes, fill_state, offset,
                    positives, lower, upper);

    // cout << "positives: " << positives << endl;

    for (unsigned p = 0; p < PARALLELISM; p++) {
      if (fill_state[p] > max_fill_state) {
        max_fill_state = fill_state[p];
      }
    }

    write_pipeline(p_dram, output_addr + num_out_lines, indexes, fill_state,
                   max_fill_state);

    num_out_lines += max_fill_state;
    num_processed_lines += num_to_process_lines;
  }

  ap_uint<BITS_IN_LINE> status_line = 0;
  status_line(31, 0) = positives;
  status_line(63, 32) = num_out_lines;

  select_write_dram_line(p_dram, status_addr, status_line);
}
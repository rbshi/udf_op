
#include "krnl_udf_olap.h"

#define HASH_BIT_MODULO(K, MASK, NBITS) ((K & MASK) >> NBITS)

inline ocxflit_t read_hbm_line(
    // hbm_gmem_t& hbm_gmem,
    hbm_t *p_hbm_in, addr_t hbm_addr) {
  ocxflit_t temp = 0;
  temp(BITS_IN_LINE - 1, 0) = p_hbm_in[hbm_addr];
  return temp;
}

inline void write_hbm_line(
    // hbm_gmem_t& hbm_gmem,
    hbm_t *p_hbm_out, addr_t hbm_addr, ocxflit_t line) {
  p_hbm_out[hbm_addr] = line(BITS_IN_LINE - 1, 0);
}

void olap_read_hbm_to_stream(hls::stream<ocxflit_t> &out_stream,
                             // read channel
                             // hbm_gmem_t& hbm_gmem,
                             hbm_t *p_hbm_in, addr_t hbm_addr,
                             unsigned size_in_lines) {
  ocxflit_t temp = 0;
  unsigned k = 0;
  while (k < size_in_lines) {
#pragma HLS PIPELINE II = 1
    temp(BITS_IN_LINE - 1, 0) = p_hbm_in[hbm_addr + k];
    out_stream.write(temp);
    k++;
  }
}

void olap_write_stream_to_hbm(hls::stream<ocxflit_t> &in_stream,
                              // write channel
                              // hbm_gmem_t& hbm_gmem,
                              hbm_t *p_hbm_out, addr_t hbm_addr,
                              unsigned size_in_lines) {
  ocxflit_t temp = 0;
  unsigned k = 0;
  while (k < size_in_lines) {
#pragma HLS PIPELINE II = 1
    temp = in_stream.read();
    p_hbm_out[hbm_addr + k] = temp(BITS_IN_LINE - 1, 0);
    k++;
  }
}

void assign_ids(hls::stream<ap_uint<BITS_IN_LINE>> &out_stream,
                hls::stream<ap_uint<BITS_IN_LINE>> &in_stream,
                unsigned num_lines) {
  ap_uint<BITS_IN_LINE> output = 0;

  for (unsigned i = 0; i < num_lines; i++) {
#pragma HLS PIPELINE II = 1

    ap_uint<BITS_IN_LINE> temp = in_stream.read();

    for (unsigned p = 0; p < WORDS_IN_LINE; p++) {
#pragma HLS UNROLL
      ap_uint<BITS_IN_TUPLE> temp_tuple = 0;
      temp_tuple(BITS_IN_WORD - 1, 0) = (i << LOG2_WORDS_IN_LINE) + p;
      temp_tuple(BITS_IN_TUPLE - 1, BITS_IN_WORD) =
          temp((p + 1) * BITS_IN_WORD - 1, p * BITS_IN_WORD);

#ifdef DEBUG2
      cout << "temp: " << hex
           << temp((p + 1) * BITS_IN_WORD - 1, p * BITS_IN_WORD) << dec << ", ";
      cout << "temp_tuple: " << hex << temp_tuple << dec << endl;
#endif
      unsigned _p = p & (TUPLES_IN_LINE - 1);
      output((_p + 1) * BITS_IN_TUPLE - 1, _p * BITS_IN_TUPLE) = temp_tuple;
      if (_p == TUPLES_IN_LINE - 1) {
        out_stream.write(output);
        output = 0;
      }
    }
  }
}

#ifdef DEBUG
// Monitor a stream
template <unsigned WORD_WIDTH, unsigned LINE_WIDTH>
void monitor(hls::stream<ap_uint<LINE_WIDTH>> &in_stream) {
  for (unsigned i = 0; i < in_stream.size(); i++) {
    ap_uint<LINE_WIDTH> temp = in_stream.read();
    cout << "monitor, line " << i << " ---- ";
    for (unsigned p = 0; p < LINE_WIDTH / WORD_WIDTH; p++) {
      cout << hex << temp((p + 1) * WORD_WIDTH - 1, p * WORD_WIDTH) << " "
           << dec;
    }
    if (LINE_WIDTH % WORD_WIDTH > 0) {
      cout << hex
           << temp(LINE_WIDTH - 1, WORD_WIDTH * (LINE_WIDTH / WORD_WIDTH))
           << dec;
    }
    cout << endl;
    in_stream.write(temp);
  }
}
#endif

template <unsigned NUM_SPLITS>
void split_stream(hls::stream<ap_uint<BITS_IN_LINE>> out_streams[NUM_SPLITS],
                  unsigned num_lines_per_split[NUM_SPLITS],
                  hls::stream<ap_uint<BITS_IN_LINE>> &in_stream,
                  unsigned num_lines) {
  unsigned temp_num_lines_per_split = num_lines / NUM_SPLITS;
  unsigned count = 0;

  for (unsigned s = 0; s < NUM_SPLITS - 1; s++) {
    unsigned count_local = 0;
    for (unsigned i = 0; i < temp_num_lines_per_split; i++) {
#pragma HLS PIPELINE
      out_streams[s].write(in_stream.read());
      count_local++;
    }
    num_lines_per_split[s] = count_local;
    count += count_local;
  }

  num_lines_per_split[NUM_SPLITS - 1] = num_lines - count;
  for (unsigned i = count; i < num_lines; i++) {
#pragma HLS PIPELINE
    out_streams[NUM_SPLITS - 1].write(in_stream.read());
  }
}

template <unsigned NUM_SPLITS>
void replicate_stream(
    hls::stream<ap_uint<BITS_IN_LINE>> out_streams[NUM_SPLITS],
    hls::stream<ap_uint<BITS_IN_LINE>> &in_stream, unsigned num_lines) {
  for (unsigned i = 0; i < num_lines; i++) {
#pragma HLS PIPELINE II = 1
    ap_uint<BITS_IN_LINE> temp = in_stream.read();
    for (unsigned p = 0; p < NUM_SPLITS; p++) {
#pragma HLS UNROLL
      out_streams[p].write(temp);
    }
  }
}

template <unsigned WIDTH, unsigned NUM_STREAMS>
void round_robin_select(
    hls::stream<ap_uint<1 + WIDTH>> &out_streams,
    hls::stream<ap_uint<1 + WIDTH>> in_streams[NUM_STREAMS]) {
  unsigned final_seen = 0;
  const ap_uint<1 + WIDTH> final = 0;
  final(WIDTH, WIDTH) = 1;

  while (final_seen < NUM_STREAMS) {
    for (unsigned i = 0; i < NUM_STREAMS; i++) {
#pragma HLS PIPELINE II = 1
      if (!in_streams[i].empty()) {
        ap_uint<1 + WIDTH> temp = in_streams[i].read();

        if (temp(WIDTH, WIDTH) == 1) {
          final_seen++;
        } else {
          out_streams.write(temp);
        }
      }
    }
  }

  out_streams.write(final);
}

template <unsigned WIDTH, unsigned MULTIPLY, unsigned FINAL_SEEN_THRESHOLD>
void assemble(hls::stream<ap_uint<1 + MULTIPLY * WIDTH>> &out_stream,
              hls::stream<ap_uint<1 + WIDTH>> &in_stream) {
  unsigned final_seen = 0;

  ap_uint<1 + MULTIPLY *WIDTH> result = -1;
  result(MULTIPLY * WIDTH, MULTIPLY * WIDTH) = 0;
  unsigned count = 0;

  while (final_seen < FINAL_SEEN_THRESHOLD) {
#pragma HLS PIPELINE II = 1

    if (!in_stream.empty()) {
      ap_uint<1 + WIDTH> temp = in_stream.read();
      if (temp(WIDTH, WIDTH) == 1) {
        final_seen++;
      } else {
        result((count + 1) * WIDTH - 1, count * WIDTH) = temp(WIDTH - 1, 0);

        if (count == MULTIPLY - 1) {
          out_stream.write(result);
          count = 0;
        } else {
          count++;
        }
      }
    }
  }

  for (unsigned p = 0; p < MULTIPLY; p++) {
#pragma HLS UNROLL
    if (p >= count) {
      result((p + 1) * WIDTH - 1, p * WIDTH) = -1;
    }
  }
  result(MULTIPLY * WIDTH, MULTIPLY * WIDTH) = 1;
  out_stream.write(result);
}

template <unsigned WIDTH, unsigned FACTOR, unsigned FINAL_SEEN_THRESHOLD>
void divide(hls::stream<ap_uint<1 + WIDTH>> &out_stream,
            hls::stream<ap_uint<1 + FACTOR * WIDTH>> &in_stream) {
  unsigned final_seen = 0;

  while (final_seen < FINAL_SEEN_THRESHOLD) {
#pragma HLS PIPELINE II = 1
    ap_uint<1 + FACTOR *WIDTH> temp = in_stream.read();

    for (unsigned i = 0; i < FACTOR; i++) {
      ap_uint<1 + WIDTH> temp_part = 0;
      temp_part = temp((i + 1) * WIDTH - 1, i * WIDTH);
      if (i == FACTOR - 1) {
        temp_part(WIDTH, WIDTH) = temp((i + 1) * WIDTH, (i + 1) * WIDTH);
      }
      out_stream.write(temp_part);
    }

    if (temp(FACTOR * WIDTH, FACTOR * WIDTH) == 1) {
      final_seen++;
    }
  }
}

template <unsigned WIDTH>
inline ap_uint<2 + WIDTH>
assemble_core(hls::stream<ap_uint<1 + WIDTH>> &in_stream) {
#pragma HLS PIPELINE II = 1

  ap_uint<2 + WIDTH> result = 0;
  result(WIDTH - 1, 0) = -1;
  if (!in_stream.empty()) {
    ap_uint<1 + WIDTH> temp = in_stream.read();
    result(WIDTH, 0) = temp;
    if (temp(WIDTH, WIDTH) == 0) {
      result(WIDTH + 1, WIDTH + 1) = 1;
    }
  }
  return result;
}

template <unsigned WIDTH, unsigned NUM_STREAMS>
void assemble_round_robin(
    hls::stream<ap_uint<1 + NUM_STREAMS * WIDTH>> &out_stream,
    hls::stream<ap_uint<1 + WIDTH>> in_stream[NUM_STREAMS]) {
  ap_uint<NUM_STREAMS> final_seen = 0;
  ap_uint<NUM_STREAMS> final_seen_1d = 0;
  ap_uint<NUM_STREAMS> stream_was_read = 0;

  ap_uint<1 + NUM_STREAMS *WIDTH> result = -1;
  result(NUM_STREAMS * WIDTH, NUM_STREAMS * WIDTH) = 0;

  do {
#pragma HLS PIPELINE II = 1
    final_seen_1d = final_seen;

    result = -1;
    result(NUM_STREAMS * WIDTH, NUM_STREAMS * WIDTH) = 0;

    for (unsigned p = 0; p < NUM_STREAMS; p++) {
#pragma HLS UNROLL
      ap_uint<2 + WIDTH> temp = assemble_core<WIDTH>(in_stream[p]);
      stream_was_read(p, p) = temp(WIDTH + 1, WIDTH + 1);
      final_seen(p, p) = final_seen(p, p) | temp(WIDTH, WIDTH);
      result((p + 1) * WIDTH - 1, p * WIDTH) = temp(WIDTH - 1, 0);
    }

    if (stream_was_read > 0) {
      out_stream.write(result);
    }
  } while (final_seen_1d != ((1 << NUM_STREAMS) - 1));

  result = -1;
  result(NUM_STREAMS * WIDTH, NUM_STREAMS * WIDTH) = 1;
  out_stream.write(result);
}

template <unsigned WIDTH, unsigned MULTIPLY>
void split(hls::stream<ap_uint<WIDTH>> out_streams[MULTIPLY],
           hls::stream<ap_uint<MULTIPLY * WIDTH>> &in_stream,
           unsigned num_lines) {
  for (unsigned i = 0; i < num_lines; i++) {
#pragma HLS PIPELINE II = 1
    ap_uint<MULTIPLY *WIDTH> temp = in_stream.read();
    for (unsigned p = 0; p < MULTIPLY; p++) {
#pragma HLS UNROLL
      out_streams[p].write(temp((p + 1) * WIDTH - 1, p * WIDTH));
    }
  }
}

template <unsigned WIDTH, unsigned MULTIPLY>
void split_assign_ids(hls::stream<ap_uint<2 * WIDTH>> out_streams[MULTIPLY],
                      hls::stream<ap_uint<MULTIPLY * WIDTH>> &in_stream,
                      unsigned num_lines) {
  for (unsigned i = 0; i < num_lines; i++) {
#pragma HLS PIPELINE II = 1
    ap_uint<MULTIPLY *WIDTH> temp = in_stream.read();
    unsigned id = i * MULTIPLY;

    for (unsigned p = 0; p < MULTIPLY; p++) {
#pragma HLS UNROLL
      ap_uint<2 * WIDTH> temp_tuple;
      temp_tuple(WIDTH - 1, 0) = id + p;
      temp_tuple(2 * WIDTH - 1, WIDTH) = temp((p + 1) * WIDTH - 1, p * WIDTH);
      out_streams[p].write(temp_tuple);
    }
  }
}

unsigned next_pow_2(unsigned v) {
  for (unsigned i = 1; i < LOG2_HASH_TABLE_SIZE; i++) {
    if (v >= (unsigned)(HASH_TABLE_SIZE >> i)) {
      return (unsigned)(HASH_TABLE_SIZE >> (i - 1));
    }
  }
  return HASH_TABLE_SIZE;
}
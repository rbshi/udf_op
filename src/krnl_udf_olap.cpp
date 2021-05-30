#include "krnl_udf_olap.h"
#include "krnl_udf_olap_misc.hpp"

void build_hash_table(
    ap_uint<LOG2_HASH_TABLE_SIZE> next[PROBE_PARALLELISM][HASH_TABLE_SIZE],
    ap_uint<LOG2_HASH_TABLE_SIZE> bucket[PROBE_PARALLELISM][HASH_TABLE_SIZE],
    ap_uint<BITS_IN_WORD> storage[PROBE_PARALLELISM][HASH_TABLE_SIZE],
    unsigned counts[PROBE_PARALLELISM],
    hls::stream<ap_uint<BITS_IN_LINE>> &in_stream, unsigned num_lines,
    unsigned hash_mask) {
  for (unsigned i = 0; i < num_lines; i++) {
#pragma HLS PIPELINE
    ap_uint<BITS_IN_LINE> line = in_stream.read();

    // While building we can read one tuple at a time
    // because we are replicating the hash table PROBE_PARALLELISM times
    for (unsigned s = 0; s < WORDS_IN_LINE; s++) {
      ap_uint<BITS_IN_WORD> value =
          line((s + 1) * BITS_IN_WORD - 1, s * BITS_IN_WORD);
      for (unsigned p = 0; p < PROBE_PARALLELISM; p++) {
#pragma HLS UNROLL
        if (value != 0xDEADBEEF) {
          storage[p][counts[p]] = value;
          unsigned hash = HASH_BIT_MODULO(value, hash_mask, 0);
          // FIXME:?
          next[p][counts[p]] = bucket[p][hash];
          bucket[p][hash] = counts[p] + 1;
          counts[p]++;
        }
      }
    }
  }
}

void join_build(
    hbm_t *p_hbm_in,
    ap_uint<LOG2_HASH_TABLE_SIZE> next[PROBE_PARALLELISM][HASH_TABLE_SIZE],
    ap_uint<LOG2_HASH_TABLE_SIZE> bucket[PROBE_PARALLELISM][HASH_TABLE_SIZE],
    ap_uint<BITS_IN_WORD> storage[PROBE_PARALLELISM][HASH_TABLE_SIZE],
    unsigned counts[PROBE_PARALLELISM], addr_t r_addr, unsigned num_lines,
    unsigned hash_mask) {
#pragma HLS DATAFLOW
  hls::stream<ap_uint<BITS_IN_LINE>> in_stream("in_stream");
  olap_read_hbm_to_stream(in_stream, p_hbm_in, r_addr, num_lines);
  build_hash_table(next, bucket, storage, counts, in_stream, num_lines,
                   hash_mask);
}

void probe_for_all(ap_uint<LOG2_HASH_TABLE_SIZE> next[HASH_TABLE_SIZE],
                   ap_uint<LOG2_HASH_TABLE_SIZE> bucket[HASH_TABLE_SIZE],
                   ap_uint<BITS_IN_WORD> storage[HASH_TABLE_SIZE],
                   unsigned &match_counts,
                   hls::stream<ap_uint<BITS_IN_TUPLE>> &in_streams,
                   hls::stream<ap_uint<1 + BITS_IN_WORD>> &out_l_streams,
                   hls::stream<ap_uint<1 + BITS_IN_WORD>> &out_r_streams,
                   unsigned r_index_offset, unsigned num_lines,
                   unsigned hash_mask) {
  unsigned local_match_counts = 0;
  for (unsigned i = 0; i < num_lines; i++) {
#pragma HLS PIPELINE II = 1
    ap_uint<BITS_IN_TUPLE> l_tuple = in_streams.read();

    int value = l_tuple(BITS_IN_TUPLE - 1, BITS_IN_WORD);
    unsigned hash = HASH_BIT_MODULO(value, hash_mask, 0);
    unsigned hit = 0;
    for (hit = bucket[hash]; hit > 0; hit = next[hit - 1]) {

      ap_uint<BITS_IN_WORD> r_value = storage[hit - 1];
      if (l_tuple(BITS_IN_TUPLE - 1, BITS_IN_WORD) == r_value) {
        local_match_counts++;

        ap_uint<1 + BITS_IN_WORD> result_l = 0;
        result_l(BITS_IN_WORD - 1, 0) = l_tuple(BITS_IN_WORD - 1, 0);
        out_l_streams.write(result_l);

        ap_uint<1 + BITS_IN_WORD> result_r = 0;
        result_r(BITS_IN_WORD - 1, 0) = hit - 1 + r_index_offset;
        out_r_streams.write(result_r);
      }
    }
  }
  match_counts = local_match_counts;
}

void probe_for_one(ap_uint<LOG2_HASH_TABLE_SIZE> next[HASH_TABLE_SIZE],
                   ap_uint<LOG2_HASH_TABLE_SIZE> bucket[HASH_TABLE_SIZE],
                   ap_uint<BITS_IN_WORD> storage[HASH_TABLE_SIZE],
                   unsigned &match_counts, unsigned &repeat_counts,
                   hls::stream<ap_uint<BITS_IN_TUPLE>> &in_streams,
                   hls::stream<ap_uint<1 + BITS_IN_WORD>> &out_l_streams,
                   hls::stream<ap_uint<1 + BITS_IN_WORD>> &out_r_streams,
                   hls::stream<ap_uint<1 + BITS_IN_TUPLE>> &out_repeat_stream,
                   unsigned r_index_offset, unsigned num_lines,
                   unsigned hash_mask) {
  unsigned local_match_counts = 0;
  unsigned local_repeat_counts = 0;
  for (unsigned i = 0; i < num_lines; i++) {
#pragma HLS PIPELINE II = 1
    ap_uint<BITS_IN_TUPLE> l_tuple = in_streams.read();

    int value = l_tuple(BITS_IN_TUPLE - 1, BITS_IN_WORD);
    unsigned hash = HASH_BIT_MODULO(value, hash_mask, 0);
    unsigned hit = bucket[hash];

    if (hit > 0) {
      unsigned hitnext = next[hit - 1];
      if (hitnext > 0) {
        out_repeat_stream.write(l_tuple);
        local_repeat_counts++;
      } else {
        ap_uint<BITS_IN_WORD> r_value = storage[hit - 1];
        if (l_tuple(BITS_IN_TUPLE - 1, BITS_IN_WORD) == r_value) {
          local_match_counts++;

          ap_uint<1 + BITS_IN_WORD> result_l = 0;
          result_l(BITS_IN_WORD - 1, 0) = l_tuple(BITS_IN_WORD - 1, 0);
          out_l_streams.write(result_l);

          ap_uint<1 + BITS_IN_WORD> result_r = 0;
          result_r(BITS_IN_WORD - 1, 0) = hit - 1 + r_index_offset;
          out_r_streams.write(result_r);
        }
      }
    }
  }
  match_counts = local_match_counts;
  repeat_counts = local_repeat_counts;
}

void probe_hash_table(
    ap_uint<LOG2_HASH_TABLE_SIZE> next[PROBE_PARALLELISM][HASH_TABLE_SIZE],
    ap_uint<LOG2_HASH_TABLE_SIZE> bucket[PROBE_PARALLELISM][HASH_TABLE_SIZE],
    ap_uint<BITS_IN_WORD> storage[PROBE_PARALLELISM][HASH_TABLE_SIZE],
    unsigned match_counts[PROBE_PARALLELISM],
    unsigned repeat_counts[PROBE_PARALLELISM],
    hls::stream<ap_uint<BITS_IN_TUPLE>> in_streams[PROBE_PARALLELISM],
    hls::stream<ap_uint<1 + BITS_IN_WORD>> out_l_streams[PROBE_PARALLELISM],
    hls::stream<ap_uint<1 + BITS_IN_WORD>> out_r_streams[PROBE_PARALLELISM],
    hls::stream<ap_uint<1 + BITS_IN_TUPLE>>
        out_repeat_stream[PROBE_PARALLELISM],
    unsigned r_index_offset, unsigned num_lines, unsigned hash_mask,
    bool handle_collisions) {

  if (handle_collisions) {
    for (unsigned p = 0; p < PROBE_PARALLELISM; p++) {
#pragma HLS UNROLL
      probe_for_all(next[p], bucket[p], storage[p], match_counts[p],
                    in_streams[p], out_l_streams[p], out_r_streams[p],
                    r_index_offset, num_lines, hash_mask);
    }
  } else {
    for (unsigned p = 0; p < PROBE_PARALLELISM; p++) {
#pragma HLS UNROLL
      probe_for_one(next[p], bucket[p], storage[p], match_counts[p],
                    repeat_counts[p], in_streams[p], out_l_streams[p],
                    out_r_streams[p], out_repeat_stream[p], r_index_offset,
                    num_lines, hash_mask);
    }
  }

  for (unsigned p = 0; p < PROBE_PARALLELISM; p++) {
#pragma HLS UNROLL
    ap_uint<1 + BITS_IN_WORD> result = -1;
    result(BITS_IN_WORD, BITS_IN_WORD) = 1;
    out_l_streams[p].write(result);
    out_r_streams[p].write(result);
    ap_uint<1 + BITS_IN_TUPLE> final_tuple = -1;
    final_tuple(BITS_IN_TUPLE, BITS_IN_TUPLE) = 1;
    out_repeat_stream[p].write(final_tuple);
  }
}

void writeback(unsigned &count_out_lines, unsigned &count_repeat_lines,
               hbm_t *p_hbm_out,
               hls::stream<ap_uint<1 + BITS_IN_LINE>> &in_l_stream,
               hls::stream<ap_uint<1 + BITS_IN_LINE>> &in_r_stream,
               hls::stream<ap_uint<1 + BITS_IN_LINE>> &in_repeat_stream,
               addr_t out_l_addr, addr_t out_r_addr, unsigned max_out_lines,
               addr_t out_repeat_addr) {
  unsigned final_seen = 0;
  unsigned count = 0;
  count_repeat_lines = 0;

  while (final_seen < 2) {
#pragma HLS PIPELINE II = 1
    if (!in_l_stream.empty()) {
      ap_uint<1 + BITS_IN_LINE> temp_l = in_l_stream.read();
      ap_uint<1 + BITS_IN_LINE> temp_r = in_r_stream.read();
      count_out_lines++;

      if (temp_l(BITS_IN_LINE, BITS_IN_LINE) == 1 &&
          temp_r(BITS_IN_LINE, BITS_IN_LINE) == 1) {
        final_seen++;
      }
      if (count < max_out_lines) {
        write_hbm_line(p_hbm_out, out_l_addr + count, temp_l);
        write_hbm_line(p_hbm_out, out_r_addr + count, temp_r);
        count++;
      }
    }

    if (!in_repeat_stream.empty()) {
      ap_uint<1 + BITS_IN_LINE> temp_repeat = in_repeat_stream.read();

      if (temp_repeat(BITS_IN_LINE, BITS_IN_LINE) == 1) {
        final_seen++;
      }
      write_hbm_line(p_hbm_out, out_repeat_addr + count_repeat_lines,
                     temp_repeat);
      count_repeat_lines++;
    }
  }
}

void join_probe(
    unsigned &count_out_lines, unsigned &count_repeat_lines, hbm_t *p_hbm_in,
    hbm_t *p_hbm_out,
    ap_uint<LOG2_HASH_TABLE_SIZE> next[PROBE_PARALLELISM][HASH_TABLE_SIZE],
    ap_uint<LOG2_HASH_TABLE_SIZE> bucket[PROBE_PARALLELISM][HASH_TABLE_SIZE],
    ap_uint<BITS_IN_WORD> storage[PROBE_PARALLELISM][HASH_TABLE_SIZE],
    unsigned match_counts[PROBE_PARALLELISM],
    unsigned repeat_counts[PROBE_PARALLELISM], size_t l_addr,
    unsigned r_index_offset, unsigned num_lines, size_t out_l_addr,
    size_t out_r_addr, unsigned max_out_lines, size_t out_repeat_addr,
    bool handle_collisions, unsigned hash_mask) {
#pragma HLS DATAFLOW

  const unsigned FACTOR = WORDS_IN_LINE / PROBE_PARALLELISM;
  const unsigned FACTOR2 = PROBE_PARALLELISM / TUPLES_IN_LINE;

  hls::stream<ap_uint<BITS_IN_LINE>> in_stream("in_stream");
  hls::stream<ap_uint<BITS_IN_TUPLE>> split_streams[PROBE_PARALLELISM];

  hls::stream<ap_uint<1 + BITS_IN_WORD>> match_l_streams[PROBE_PARALLELISM];
  hls::stream<ap_uint<1 + BITS_IN_WORD>> match_r_streams[PROBE_PARALLELISM];
  hls::stream<ap_uint<1 + BITS_IN_TUPLE>>
      match_repeat_streams[PROBE_PARALLELISM];

  hls::stream<ap_uint<1 + BITS_IN_WORD * PROBE_PARALLELISM>> out_l_stream(
      "out_l_stream");
  hls::stream<ap_uint<1 + BITS_IN_WORD * PROBE_PARALLELISM>> out_r_stream(
      "out_r_stream");
  hls::stream<ap_uint<1 + BITS_IN_TUPLE * PROBE_PARALLELISM>> out_repeat_stream(
      "out_repeat_stream");

  hls::stream<ap_uint<1 + BITS_IN_LINE>> out_l_mem_stream("out_l_mem_stream");
  hls::stream<ap_uint<1 + BITS_IN_LINE>> out_r_mem_stream("out_r_mem_stream");
  hls::stream<ap_uint<1 + BITS_IN_LINE>> out_repeat_mem_stream(
      "out_repeat_mem_stream");

  olap_read_hbm_to_stream(in_stream, p_hbm_in, l_addr, num_lines);

#ifdef DEBUG
  cout << "in_stream: " << endl;
  monitor<BITS_IN_TUPLE, BITS_IN_LINE>(in_stream);
#endif

#if PROBE_PARALLELISM == 16
  split_assign_ids<BITS_IN_WORD, PROBE_PARALLELISM>(split_streams, in_stream,
                                                    num_lines);
#else // PROBE_PARALLELISM == 8
  split<BITS_IN_TUPLE, PROBE_PARALLELISM>(split_streams, in_stream, num_lines);
#endif

#ifdef DEBUG
  for (unsigned p = 0; p < PROBE_PARALLELISM; p++) {
    cout << "split_streams[" << p << "]: " << endl;
    monitor<BITS_IN_TUPLE, BITS_IN_TUPLE>(split_streams[p]);
  }
#endif

  probe_hash_table(next, bucket, storage, match_counts, repeat_counts,
                   split_streams, match_l_streams, match_r_streams,
                   match_repeat_streams, r_index_offset, num_lines, hash_mask,
                   handle_collisions);

#ifdef DEBUG
  for (unsigned p = 0; p < PROBE_PARALLELISM; p++) {
    cout << "match_l_streams[" << p << "]: " << endl;
    monitor<BITS_IN_WORD, 1 + BITS_IN_WORD>(match_l_streams[p]);
  }
  for (unsigned p = 0; p < PROBE_PARALLELISM; p++) {
    cout << "match_repeat_streams[" << p << "]: " << endl;
    monitor<BITS_IN_TUPLE, 1 + BITS_IN_TUPLE>(match_repeat_streams[p]);
  }
#endif

  assemble_round_robin<BITS_IN_WORD, PROBE_PARALLELISM>(out_l_stream,
                                                        match_l_streams);
  assemble_round_robin<BITS_IN_WORD, PROBE_PARALLELISM>(out_r_stream,
                                                        match_r_streams);
  assemble_round_robin<BITS_IN_TUPLE, PROBE_PARALLELISM>(out_repeat_stream,
                                                         match_repeat_streams);

#ifdef DEBUG
  cout << "out_l_stream" << endl;
  monitor<BITS_IN_WORD, 1 + BITS_IN_WORD * PROBE_PARALLELISM>(out_l_stream);
  cout << "out_r_stream" << endl;
  monitor<BITS_IN_WORD, 1 + BITS_IN_WORD * PROBE_PARALLELISM>(out_r_stream);
#endif

  assemble<BITS_IN_WORD * PROBE_PARALLELISM, FACTOR, 1>(out_l_mem_stream,
                                                        out_l_stream);
  assemble<BITS_IN_WORD * PROBE_PARALLELISM, FACTOR, 1>(out_r_mem_stream,
                                                        out_r_stream);
  divide<BITS_IN_LINE, FACTOR2, 1>(out_repeat_mem_stream, out_repeat_stream);

#ifdef DEBUG
  cout << "out_l_mem_stream" << endl;
  monitor<BITS_IN_WORD, 1 + BITS_IN_LINE>(out_l_mem_stream);
  cout << "out_l_addr: " << out_l_addr << endl;
  cout << "out_r_addr: " << out_r_addr << endl;
#endif

  writeback(count_out_lines, count_repeat_lines, p_hbm_out, out_l_mem_stream,
            out_r_mem_stream, out_repeat_mem_stream, out_l_addr, out_r_addr,
            max_out_lines, out_repeat_addr);
}

void krnl_udf_olap(hbm_t *p_hbm_in, hbm_t *p_hbm_out, const addr_t status_addr,
                   const addr_t l_addr, unsigned l_num_lines,
                   const addr_t r_addr, unsigned r_num_lines,
                   const addr_t out_l_addr, const addr_t out_r_addr,
                   unsigned max_out_lines, const addr_t out_repeat_addr,
                   unsigned r_index_offset, bool build_r,
                   bool handle_collisions,
                   const unsigned num_times) {

#pragma HLS INTERFACE m_axi port = p_hbm_in offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = p_hbm_out offset = slave bundle = gmem1

#pragma HLS INTERFACE s_axilite port = p_hbm_in
#pragma HLS INTERFACE s_axilite port = p_hbm_out
#pragma HLS INTERFACE s_axilite port = status_addr
#pragma HLS INTERFACE s_axilite port = l_addr
#pragma HLS INTERFACE s_axilite port = l_num_lines
#pragma HLS INTERFACE s_axilite port = r_addr
#pragma HLS INTERFACE s_axilite port = r_num_lines
#pragma HLS INTERFACE s_axilite port = out_l_addr
#pragma HLS INTERFACE s_axilite port = out_r_addr
#pragma HLS INTERFACE s_axilite port = max_out_lines
#pragma HLS INTERFACE s_axilite port = out_repeat_addr
#pragma HLS INTERFACE s_axilite port = r_index_offset
#pragma HLS INTERFACE s_axilite port = build_r
#pragma HLS INTERFACE s_axilite port = handle_collisions
#pragma HLS INTERFACE s_axilite port = num_times

#pragma HLS INTERFACE s_axilite port = return

  // Build local hash table
  static ap_uint<LOG2_HASH_TABLE_SIZE> next[PROBE_PARALLELISM][HASH_TABLE_SIZE];
  static ap_uint<LOG2_HASH_TABLE_SIZE> bucket[PROBE_PARALLELISM]
                                             [HASH_TABLE_SIZE];
  static ap_uint<BITS_IN_WORD> storage[PROBE_PARALLELISM][HASH_TABLE_SIZE];
  static unsigned match_counts[PROBE_PARALLELISM];
  static unsigned repeat_counts[PROBE_PARALLELISM];

// FIXME: why use LUT?, let Vivado choose - some kernel use BRAM, others use LUT
// #pragma HLS RESOURCE variable = next core = RAM_1P_LUTRAM
#pragma HLS RESOURCE variable = bucket core = RAM_1P_BRAM
// #pragma HLS RESOURCE variable=storage core=RAM_1P_BRAM
// #pragma HLS RESOURCE variable=next core=XPM_MEMORY uram
// #pragma HLS RESOURCE variable=bucket core=XPM_MEMORY uram
#pragma HLS RESOURCE variable = storage core = XPM_MEMORY uram
#pragma HLS ARRAY_PARTITION variable = next dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = bucket dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = storage dim = 1 complete

#pragma HLS ARRAY_PARTITION variable = match_counts complete
#pragma HLS ARRAY_PARTITION variable = repeat_counts complete

#if PROBE_PARALLELISM == 8
  unsigned l_num_lines_after_id = l_num_lines << 1;
#else
  unsigned l_num_lines_after_id = l_num_lines;
#endif
  unsigned hash_table_size = next_pow_2(r_num_lines * WORDS_IN_LINE);
  unsigned hash_mask = hash_table_size - 1;
#ifdef DEBUG
  cout << "hash_table_size: " << hash_table_size << endl;
#endif

  for (unsigned p = 0; p < PROBE_PARALLELISM; p++) {
#pragma HLS UNROLL
    match_counts[p] = 0;
  }
  // If build_r is false -> Do not reset the previous hash table
  if (build_r) {
    for (unsigned i = 0; i < hash_table_size; i++) {
#pragma HLS PIPELINE
      for (unsigned p = 0; p < PROBE_PARALLELISM; p++) {
#pragma HLS UNROLL
        bucket[p][i] = 0;
      }
    }
  }

#ifdef DEBUG
  cout << "--- join_build ---" << endl;
#endif
  // If build_r is 0 -> Rely on the built hash table in the previous call
  if (build_r) {
    join_build(p_hbm_in, next, bucket, storage, match_counts, r_addr,
               r_num_lines, hash_mask);
  }


COUNT_LOOP:
for (int count = 0; count < num_times; count++) {

#ifdef DEBUG
  cout << "--- storage: " << endl;
  for (unsigned i = 0; i < match_counts[0]; i++) {
    cout << i << ": " << hex << storage[0][i] << dec << endl;
  }
#endif

  for (unsigned p = 0; p < PROBE_PARALLELISM; p++) {
#pragma HLS UNROLL
    match_counts[p] = 0;
    repeat_counts[p] = 0;
  }


#ifdef DEBUG
  cout << "--- join_probe ---" << endl;
#endif
  unsigned count_out_lines = 0;
  unsigned count_repeat_lines = 0;
  join_probe(count_out_lines, count_repeat_lines, p_hbm_in, p_hbm_out, next,
             bucket, storage, match_counts, repeat_counts, l_addr,
             r_index_offset, l_num_lines_after_id, out_l_addr, out_r_addr,
             max_out_lines, out_repeat_addr, handle_collisions, hash_mask);

  unsigned num_matches = 0;
  unsigned num_repeats = 0;
  for (unsigned p = 0; p < PROBE_PARALLELISM; p++) {
    // #pragma HLS UNROLL
    num_matches += match_counts[p];
    num_repeats += repeat_counts[p];
  }

  ap_uint<BITS_IN_LINE> status_line = 0;
  status_line(31, 0) = num_matches;
  status_line(63, 32) = count_out_lines;
  status_line(95, 64) = num_repeats;
  status_line(127, 96) = count_repeat_lines;

  write_hbm_line(p_hbm_out, status_addr, status_line);
}
}


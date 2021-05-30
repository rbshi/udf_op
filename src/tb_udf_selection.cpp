
#include <iostream>
#include <sys/time.h>

using namespace std;

// #define DEBUG

#define NUM_KERNEL 32

#include "krnl_udf_selection.h"
#include "hbm_column.hpp"

#define HBM_SIZE 32768



void datamover_write(hbm_t *hbm_memory, hbm_column<int> *in) {
  cout << "in->m_num_partitions: " << in->m_num_partitions << endl;

  uint32_t item_count = 0;
  for (uint32_t p = 0; p < in->m_num_partitions; p++) {
    cout << "in->m_num_lines[" << p << "]: " << in->m_num_lines[p] << endl;
    for (uint32_t i = 0; i < in->m_num_lines[p]; i++) {
      for (uint32_t j = 0; j < INTS_IN_HBM_LINE; j++) {
        hbm_memory[in->m_hbm_offset[p] + i](BITS_IN_INT * (j + 1) - 1,
                                            BITS_IN_INT * j) =
            in->get_item(item_count++);
      }
    }
  }
}

void datamover_read(hbm_t *hbm_memory, hbm_column<uint32_t> *out, unsigned num_in_lines) {
  // out->column_realloc(num_lines*INTS_IN_LINE);
  cout << "out->m_total_num_lines: " << out->m_total_num_lines << endl;
  for (uint32_t p = 0; p < out->m_num_partitions; p++) {
    // obtain the num_lines, num_possitivies is ignored
    uint32_t num_lines = (uint32_t)hbm_memory[out->m_hbm_offset[p]](63, 32);
    for (uint32_t i =0; i < num_lines; i++) {
    for (uint32_t j = 0; j < INTS_IN_HBM_LINE; j++) {
      // NOTE: +1 due to the status address
      uint32_t temp = (uint32_t)hbm_memory[out->m_hbm_offset[p] + 1 + i](
          BITS_IN_INT * (j + 1) - 1, BITS_IN_INT * j);
      if (temp != 0xFFFFFFFF)
        // FIXME: offset of index, use [0] because it contains the avg value
        out->append(temp + INTS_IN_HBM_LINE * num_in_lines * p);
    }
  }
  }
}

int main(int argc, char *argv[]) {

  unsigned num_values = 1024;
  int lower = 0;
  int upper = 0;
  if (argc != 4) {
    cout << "Usage: ./testbench <num_values> <lower> <upper>" << endl;
    return 1;
  }
  num_values = atoi(argv[1]);
  lower = atoi(argv[2]);
  upper = atoi(argv[3]);
  cout << "num_values: " << num_values << endl;
  cout << "lower: " << lower << endl;
  cout << "upper: " << upper << endl;

  srand(3);

  // Input
  hbm_column<int> in_column(num_values, 0);
  in_column.populate_int_column(num_values, 'm', '-', 0xDEADBEEF);

  // Output
  hbm_column<uint32_t> out_column(num_values, in_column.m_base_hbm_offset +
                                                  in_column.m_total_num_lines);

  // Transfer data to HBM
  hbm_t hbm_memory[HBM_SIZE];

  datamover_write(hbm_memory, &in_column);

  
  // addr_t in_addr = in_column.m_base_hbm_offset;

  // // the first HBM line is reserved for status
  // addr_t status_addr = out_column.m_base_hbm_offset;

  // addr_t out_addr = out_column.m_base_hbm_offset+1;


  // unsigned num_in_lines =
  //     ((num_values >> LOG2_INTS_IN_LINE) + (num_values % INTS_IN_LINE > 0))/NUM_KERNEL;
  for (unsigned i = 0; i < NUM_KERNEL; i++){
    addr_t in_addr = in_column.m_hbm_offset[i];
    addr_t status_addr = out_column.m_hbm_offset[i];
    // the first HBM line is reserved for status
    addr_t out_addr = out_column.m_hbm_offset[i]+1;
    // FIXME: exact line number
    unsigned num_in_lines = in_column.m_num_lines[i];
    krnl_udf_selection(hbm_memory, in_addr, out_addr, status_addr, num_in_lines,
                     lower, upper, 10);
  }

  // m_num_lines[0] have the average value
  datamover_read(hbm_memory, &out_column, in_column.m_num_lines[0]);

  cout << "out_column.get_num_items(): " << out_column.get_num_items() << endl;
  out_column.sort_items();

  hbm_column<uint32_t> sw_out_column(num_values, 0);
  for (unsigned i = 0; i < num_values; i++) {
    int value = in_column.get_item(i);
    if (value > lower && value < upper) {
      sw_out_column.append(i);
    }
  }
  sw_out_column.sort_items();

  bool success = true;
  if (out_column.get_num_items() != sw_out_column.get_num_items()) {
    cout << "HW (" << out_column.get_num_items() << ") and SW ("
         << sw_out_column.get_num_items() << ") num_matches are different!"
         << endl;
    success = false;
  } else {
    for (unsigned i = 0; i < out_column.get_num_items(); i++) {
      if (out_column.get_item(i) != sw_out_column.get_item(i)) {
        cout << "Mismatch at " << i << ": HW " << out_column.get_item(i)
             << ", SW " << sw_out_column.get_item(i) << endl;
        success = false;
      }
    }
  }

  if (success) {
    cout << "SUCCESS!" << endl;
  } else {
    cout << "FAIL!" << endl;
  }

  return 0;
}

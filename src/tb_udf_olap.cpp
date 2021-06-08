#include <sys/time.h>
#include <iostream>
#include <algorithm>

#define AP_INT_MAX_W 2048
#define HBM_PARTITION

// #define DEBUG
// #define VERBOSE

using namespace std;

#include "krnl_udf_olap.h"
#include "hbm_column.hpp"

#define HBM_SIZE 32768

void sw_join_main(hbm_column<uint32_t> *r1, hbm_column<uint32_t> *r2,
                  hbm_column<int> *in_l, hbm_column<int> *in_r) {
  uint32_t i, j;

  for (i = 0; i < in_l->get_num_items(); i++) {
    for (j = 0; j < in_r->get_num_items(); j++) {
      if (in_l->get_item(i) == in_r->get_item(j)) {
        r1->append(i);
        r2->append(j);
      }
    }
  }
}

uint8_t verify(hbm_column<uint32_t> *sw_r, hbm_column<uint32_t> *r) {
  if (sw_r->get_num_items() != r->get_num_items()) {
    printf("r->get_num_items() is not correct. sw_r: %d, r: %d\n",
           sw_r->get_num_items(), r->get_num_items());
    return 1;
  }

  sw_r->sort_items();
  r->sort_items();

  uint8_t result = 0;
  for (uint32_t i = 0; i < sw_r->get_num_items(); i++) {
    if (sw_r->get_item(i) != r->get_item(i)) {
#ifdef VERBOSE
      printf("Mismatch at %d: sw_r: %d, r:%d\n", i, sw_r->get_item(i),
             r->get_item(i));
#endif
      result = 1;
    }
#ifdef VERBOSE
    else {
      printf("Match at %d: sw_r: %d, r:%d\n", i, sw_r->get_item(i),
             r->get_item(i));
    }
#endif
  }
  return result;
}

void datamover_write(hbm_t *hbm_memory, hbm_column<int> *in) {
  cout << "in->m_num_partitions: " << in->m_num_partitions << endl;

  uint32_t item_count = 0;
  for (uint32_t p = 0; p < in->m_num_partitions; p++) {
    cout << "in->m_num_lines[" << p << "]: " << in->m_num_lines[p] << endl;
    for (uint32_t i = 0; i < in->m_num_lines[p]; i++) {
      for (uint32_t j = 0; j < WORDS_IN_HBM_LINE; j++) {
        hbm_memory[in->m_hbm_offset[p] + i](BITS_IN_WORD * (j + 1) - 1,
                                            BITS_IN_WORD * j) =
            in->get_item(item_count++);
      }
    }
  }
}

void datamover_read(hbm_t *hbm_memory, unsigned num_in_lines, hbm_column<uint32_t> *out) {
    // obtain the num_lines, num_possitivies is ignored
    for (uint32_t i = 0; i < num_in_lines; i++) {
      for (uint32_t j = 0; j < WORDS_IN_HBM_LINE; j++) {
        // NOTE: +1 due to the status address
        uint32_t temp = (uint32_t)hbm_memory[out->m_hbm_offset[0] + i](
            BITS_IN_WORD * (j + 1) - 1, BITS_IN_WORD * j);
        if (temp != 0xFFFFFFFF) {
          // FIXME: offset of index, use [0] because it contains the avg value
          out->append(temp);
          cout << "out->m_num_items=" << out.get_num_items() << 'with temp=' << temp << endl;
        }
      }
    }
  }

int main(int argc, char *argv[]) {

	unsigned num_kernel = 1;
	unsigned l_count = 1024;
	char* l_config;
	unsigned r_count = 2048;
	char* r_config;
	unsigned build_r = 1;
	unsigned handle_collisions = 1;
	unsigned do_verify = 1;
	if (argc != 8) {
		printf("Usage: ./testbench <l_count> <for l:'u'nique's'huffled> <r_count> <for r:'u'nique's'huffled> <build_r> <handle_collisions> <do_verify>\n");
		return 1;
	}
	l_count = atoi(argv[1]);
	l_config = argv[2];
	r_count = atoi(argv[3]);
	r_config = argv[4];
	build_r = atoi(argv[5]);
	handle_collisions = atoi(argv[6]);
	do_verify = atoi(argv[7]);
	printf("l_count: %d\n", l_count);
	printf("l_config: %s\n", l_config);
	printf("r_count: %d\n", r_count);
	printf("r_config: %s\n", r_config);
	printf("build_r: %d\n", build_r);
	printf("handle_collisions: %d\n", handle_collisions);
	printf("do_verify: %d\n", do_verify);

	srand(3);

	// Input
	hbm_column<int> in_r(num_kernel, r_count, 10);
	hbm_column<int> in_l(num_kernel, l_count, in_r.m_base_hbm_offset + in_r.m_total_num_lines);
	in_r.populate_int_column(r_count, r_config[0], r_config[1], 0xDEADBEEF);
	in_l.populate_int_column(l_count, l_config[0], l_config[1], 0xBEEFDEAD);

	// Output
	unsigned max_items = 2*l_count;
	hbm_column<uint32_t> r1(num_kernel, max_items, in_l.m_base_hbm_offset + in_l.m_total_num_lines);
	hbm_column<uint32_t> r2(num_kernel, max_items, r1.m_base_hbm_offset + r1.m_total_num_lines);
	hbm_column<tuple_t> repeat(num_kernel, max_items, r2.m_base_hbm_offset + r2.m_total_num_lines);

	// Transfer data to HBM
	hbm_t hbm_memory[HBM_SIZE];

	addr_t status_addr = 3;
	addr_t l_addr = in_l.m_base_hbm_offset;
	addr_t r_addr = in_r.m_base_hbm_offset;
	unsigned l_num_lines = in_l.m_total_num_lines;
	unsigned r_num_lines = in_r.m_total_num_lines;
	addr_t out_l_addr = r1.m_base_hbm_offset;
	addr_t out_r_addr = r2.m_base_hbm_offset;
	addr_t out_repeat_addr = repeat.m_base_hbm_offset;
	unsigned max_out_lines = r1.m_total_num_lines;

	datamover_write(hbm_memory, &in_r);
	datamover_write(hbm_memory, &in_l);

	cout << "l_addr: " << l_addr << endl;
	cout << "l_num_lines: " << l_num_lines << endl;
	cout << "r_addr: " << r_addr << endl;
	cout << "r_num_lines: " << r_num_lines << endl;

	const unsigned hash_table_size = HASH_TABLE_SIZE-WORDS_IN_LINE;
	unsigned num_hash_iterations = r_count/hash_table_size + (r_count%hash_table_size > 0);
	cout << "num_hash_iterations: " << num_hash_iterations << endl;

	unsigned r_num_lines_processed = 0;
	unsigned num_matches = 0;
	unsigned count_out_lines = 0;
	unsigned num_repeats = 0;
	unsigned count_repeat_lines = 0;
	for (unsigned i = 0; i < num_hash_iterations; i++) {
		unsigned r_num_lines_to_process;
		if (i == num_hash_iterations-1) {
			r_num_lines_to_process = r_num_lines - r_num_lines_processed;
		}
		else {
			r_num_lines_to_process = hash_table_size/WORDS_IN_LINE;
		}

		cout << "--------> r_num_lines_to_process: " << r_num_lines_to_process << endl;

		krnl_udf_olap(
			hbm_memory, hbm_memory,
			status_addr,
			l_addr, l_num_lines,
			r_addr + r_num_lines_processed, r_num_lines_to_process,
			out_l_addr + count_out_lines, out_r_addr + count_out_lines, max_out_lines,
			out_repeat_addr + count_repeat_lines,
			r_num_lines_processed*WORDS_IN_LINE,
			build_r, handle_collisions);

		r_num_lines_processed += r_num_lines_to_process;

		num_matches += hbm_memory[status_addr](31,0);
		count_out_lines += hbm_memory[status_addr](63,32);
		num_repeats += hbm_memory[status_addr](95,64);
		count_repeat_lines += hbm_memory[status_addr](127,96);
		cout << "--------> hash iteratios: " << i << endl;
		cout << "num_matches: " << num_matches << endl;
		cout << "count_out_lines: " << count_out_lines << endl;
		cout << "num_repeats: " << num_repeats << endl;
		cout << "count_repeat_lines: " << count_repeat_lines << endl;
	}

	// Verification
	if (do_verify == 1) {
		datamover_read(hbm_memory, count_out_lines, &r1);
		datamover_read(hbm_memory, count_out_lines, &r2);
		cout << "r1.get_num_items() : " << r1.get_num_items() << endl;
		cout << "r2.get_num_items() : " << r2.get_num_items() << endl;

		hbm_column<uint32_t> sw_r1(num_kernel, num_matches, 0);
		hbm_column<uint32_t> sw_r2(num_kernel, num_matches, 0);
		sw_join_main(&sw_r1, &sw_r2, &in_l, &in_r);
		
		printf("sw_join num_matches: %d\n", sw_r1.get_num_items());

		uint8_t result1 = verify(&sw_r1, &r1);
		uint8_t result2 = verify(&sw_r2, &r2);
		if (result1 + result2 == 0) {
			printf("Success!\n");
		}
		else {
			printf("Fail...\n");
		}
	}

	return 0;
}

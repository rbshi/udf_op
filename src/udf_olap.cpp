#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_ini.h"
#include "experimental/xrt_kernel.h"

#include "krnl_udf_olap.h"
#include "hbm_column.hpp"

#define HBM_CHANNEL_SIZE (1<<28)
#define AP_INT_MAX_W 2048
#define HBM_PARTITION



// #define DEBUG
#define VERBOSE

using namespace std;

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

void datamover_read(std::vector<int*> hbm_buffer_ptr, unsigned num_in_lines, hbm_column<uint32_t> *out, int idx_partition, int offset_partition) {
	int hbm_size = (1 << 28);
	uint64_t hbm_buffer_idx = out->m_hbm_offset[idx_partition] / uint64_t(hbm_size * 4 / BYTES_IN_HBM_LINE);
	uint64_t hbm_buffer_offset = out->m_hbm_offset[idx_partition] % uint64_t(hbm_size * 4 / BYTES_IN_HBM_LINE);
  for (uint32_t i = 0; i < num_in_lines; i++) {
    for (uint32_t j = 0; j < WORDS_IN_HBM_LINE; j++) {
      uint32_t temp = hbm_buffer_ptr[hbm_buffer_idx][(hbm_buffer_offset + i) * WORDS_IN_HBM_LINE + j];
      if (temp != 0xFFFFFFFF) {
        out->append(temp + idx_partition * offset_partition);
        // cout << "out->m_num_items=" << out.get_num_items() << 'with temp=' << temp << endl;
      }
    }
  }
}

int main(int argc, char *argv[]) {


	if (argc != 10) {
		printf("Usage: ./testbench <l_count> <for l:'u'nique's'huffled> <r_count> <for r:'u'nique's'huffled> <build_r> <handle_collisions> <do_verify> <num_kernel> <.xclbin>\n");
		return 1;
	}

	unsigned l_count = 1024;
	char* l_config;
	unsigned r_count = 2048;
	char* r_config;
	unsigned build_r = 1;
	unsigned handle_collisions = 1;
	unsigned do_verify = 1;

	l_count = atoi(argv[1]);
	l_config = argv[2];
	r_count = atoi(argv[3]);
	r_config = argv[4];
	build_r = atoi(argv[5]);
	handle_collisions = atoi(argv[6]);
	do_verify = atoi(argv[7]);
  int num_kernel = atoi(argv[8]);
  std::string xclbin_fnm = argv[9];

	printf("l_count: %d\n", l_count);
	printf("l_config: %s\n", l_config);
	printf("r_count: %d\n", r_count);
	printf("r_config: %s\n", r_config);
	printf("build_r: %d\n", build_r);
	printf("handle_collisions: %d\n", handle_collisions);
	printf("do_verify: %d\n", do_verify);

	srand(3);

  if (xclbin_fnm.empty())
    throw std::runtime_error("FAILED_TEST\nNo xclbin specified");

  std::string cu_name = "krnl_udf_olap";

  unsigned int device_index = 0;

  auto device = xrt::device(device_index);
  auto uuid = device.load_xclbin(xclbin_fnm);
  auto krnl_all = xrt::kernel(device, uuid, cu_name);


  int stride_partition_hbm_line = HBM_CHANNEL_SIZE / BYTES_IN_HBM_LINE * (32 / num_kernel);
  // int stride_partition_hbm_line = HBM_CHANNEL_SIZE / BYTES_IN_HBM_LINE * 2;
  cout << "stride_partition_hbm_line=" << stride_partition_hbm_line << endl;
  cout << "stride_partition_hbm_line/2=" << stride_partition_hbm_line/2 << endl;

	// Input, r is replicated in all kernels, DONOT partition
	hbm_column<int> in_r(num_kernel, r_count, 10);
	// FIXME: strided mode DOSE NOT work if len(in_r) % num_kernel != 0
	hbm_column<int> in_l(num_kernel, l_count, in_r.m_base_hbm_offset + in_r.m_total_num_lines, true, stride_partition_hbm_line);
	in_r.populate_int_column(r_count, r_config[0], r_config[1], 0xDEADBEEF);
	in_l.populate_int_column(l_count, l_config[0], l_config[1], 0xBEEFDEAD);

	// Output
	unsigned max_items = 2*l_count;

	// hbm_column<uint32_t> r1(num_kernel, max_items, in_l.m_base_hbm_offset + in_l.m_total_num_lines);
	// results base_offset will be 1/2 stride_partition (start from a new HBM channel)
	hbm_column<uint32_t> r1(num_kernel, max_items, stride_partition_hbm_line/2, true, stride_partition_hbm_line);
	hbm_column<uint32_t> r2(num_kernel, max_items, r1.m_base_hbm_offset + r1.m_num_lines[0], true, stride_partition_hbm_line);
	hbm_column<tuple_t> repeat(num_kernel, max_items, r2.m_base_hbm_offset + r2.m_num_lines[0], true, stride_partition_hbm_line);

  
  // Allocate input buffer on HBM, 8 large buffer (each with 1GB)
  std::vector<xrt::bo> hbm_buffer(8);
  std::vector<int*> hbm_buffer_ptr(8);
  int hbm_size = (1<<28);

	for (int i = 0; i < 8; i++) {
		hbm_buffer[i] = xrt::bo(device, hbm_size * 4, krnl_all.group_id(0));
		auto hbm_channel_ptr = hbm_buffer[i].map<int*>();
		hbm_buffer_ptr[i] = hbm_channel_ptr;
		cout << "allocated one large buffer." << endl;
	}


	int* ptr_start_in_l = in_l.get_base();
	for (int i = 0; i < num_kernel * 1; i++) {
		// targe hbm_buffer idx
		// in_r 
		uint64_t hbm_buffer_idx = (in_r.m_base_hbm_offset + stride_partition_hbm_line * i) / uint64_t(hbm_size * 4 / BYTES_IN_HBM_LINE);
		uint64_t hbm_buffer_offset = (in_r.m_base_hbm_offset + stride_partition_hbm_line * i) % uint64_t(hbm_size * 4 / BYTES_IN_HBM_LINE);

		std::cout << "hbm_buffer_idx=" << hbm_buffer_idx << endl;
		std::cout << "hbm_buffer_offset=" << hbm_buffer_offset << endl;		

		std::copy_n(in_r.get_base(), in_r.m_total_num_lines * WORDS_IN_HBM_LINE, (hbm_buffer_ptr[hbm_buffer_idx] + hbm_buffer_offset * WORDS_IN_HBM_LINE));
		hbm_buffer[hbm_buffer_idx].sync(XCL_BO_SYNC_BO_TO_DEVICE, in_r.m_total_num_lines * BYTES_IN_HBM_LINE, hbm_buffer_offset * BYTES_IN_HBM_LINE);
		
		std::cout << "Copy in_r finished\n";

		// in_l
		hbm_buffer_idx = in_l.m_hbm_offset[i] / uint64_t(hbm_size * 4 / BYTES_IN_HBM_LINE);
		hbm_buffer_offset = in_l.m_hbm_offset[i] % uint64_t(hbm_size * 4 / BYTES_IN_HBM_LINE);

		std::cout << "hbm_buffer_idx=" << hbm_buffer_idx << endl;
		std::cout << "hbm_buffer_offset=" << hbm_buffer_offset << endl;		

		std::copy_n(ptr_start_in_l, in_l.m_num_lines[i] * WORDS_IN_HBM_LINE, (hbm_buffer_ptr[hbm_buffer_idx] + hbm_buffer_offset * WORDS_IN_HBM_LINE));
		hbm_buffer[hbm_buffer_idx].sync(XCL_BO_SYNC_BO_TO_DEVICE, in_l.m_num_lines[i] * BYTES_IN_HBM_LINE, hbm_buffer_offset * BYTES_IN_HBM_LINE);

		ptr_start_in_l += in_l.m_num_lines[i] * WORDS_IN_HBM_LINE;

		std::cout << "Copy in_l finished\n";

	}
  std::cout << "Memory load finished\n";



	// cout << "l_addr: " << l_addr << endl;
	// cout << "l_num_lines: " << l_num_lines << endl;
	// cout << "r_addr: " << r_addr << endl;
	// cout << "r_num_lines: " << r_num_lines << endl;



	const unsigned hash_table_size = HASH_TABLE_SIZE-WORDS_IN_LINE;
	unsigned num_hash_iterations = r_count/hash_table_size + (r_count%hash_table_size > 0);
	cout << "num_hash_iterations: " << num_hash_iterations << endl;

	unsigned r_num_lines_processed = 0;

	unsigned num_matches[num_kernel] = {0};
	unsigned count_out_lines[num_kernel] = {0};
	unsigned num_repeats[num_kernel] = {0};
	unsigned count_repeat_lines[num_kernel] = {0};

	// kernel input parameters
	addr_t status_addr[num_kernel];
	addr_t r_addr[num_kernel];
	addr_t l_addr[num_kernel];
	unsigned r_num_lines = in_r.m_total_num_lines;
	unsigned l_num_lines[num_kernel];			
	addr_t out_l_addr[num_kernel];
	addr_t out_r_addr[num_kernel];
	addr_t out_repeat_addr[num_kernel];
	unsigned max_out_lines[num_kernel];

	for (int i = 0; i < num_kernel; i++) {
		status_addr[i] = 3 + i * stride_partition_hbm_line;
		r_addr[i] = in_r.m_base_hbm_offset + i * stride_partition_hbm_line;
		l_addr[i] = in_l.m_hbm_offset[i];
		l_num_lines[i] = in_l.m_num_lines[i];			
		out_l_addr[i] = r1.m_hbm_offset[i];
		out_r_addr[i] = r2.m_hbm_offset[i];
		out_repeat_addr[i] = repeat.m_hbm_offset[i];
		max_out_lines[i] = r1.m_num_lines[i];
	}

	cout << "r1.m_hbm_offset=" << r1.m_hbm_offset[0] << endl;

  // run the kernel
  std::vector<xrt::run> runs(num_kernel);

	for (unsigned j = 0; j < num_hash_iterations; j++) {

		unsigned r_num_lines_to_process;
		if (j == num_hash_iterations-1) {
			r_num_lines_to_process = r_num_lines - r_num_lines_processed;
		}
		else {
			r_num_lines_to_process = hash_table_size/WORDS_IN_LINE;
		}
		cout << "[INFO] r_num_lines_to_process: " << r_num_lines_to_process << endl;


		for (int i = 0; i < num_kernel; i++) {
	    std::string cu_id = std::to_string(i + 1);
	    std::string krnl_name_full = cu_name + ":{" + cu_name + "_" + cu_id + "}";
	    auto krnl_inst = xrt::kernel(device, uuid, krnl_name_full, 0);

	    cout << status_addr[i] << "\t" << l_addr[i] << "\t" << l_num_lines[i] << "\t" \
	     << r_addr[i] + r_num_lines_processed << "\t" << r_num_lines_to_process << "\t" << out_l_addr[i] + count_out_lines[i] << "\t" << out_r_addr[i] + count_out_lines[i] \
	     << "\t"  << max_out_lines[i] << "\t" << r_num_lines_processed * WORDS_IN_LINE << endl;

	    auto run = krnl_inst(
					hbm_buffer[0], hbm_buffer[0],
					status_addr[i],
					l_addr[i], l_num_lines[i],
					r_addr[i] + r_num_lines_processed, r_num_lines_to_process,
					out_l_addr[i] + count_out_lines[i], out_r_addr[i] + count_out_lines[i], max_out_lines[i],
					out_repeat_addr[i] + count_repeat_lines[i],
					r_num_lines_processed * WORDS_IN_LINE,
					build_r, handle_collisions);
	    runs[i] = run;
		}

		r_num_lines_processed += r_num_lines_to_process;


	  for (auto &run : runs) {
	    auto state = run.wait();
	  }

		// obtain the results of one iteration
		for (int i = 0; i < num_kernel; i++) {
			uint64_t hbm_buffer_idx = status_addr[i] / uint64_t(hbm_size * 4 / BYTES_IN_HBM_LINE);
			uint64_t hbm_buffer_offset = status_addr[i] % uint64_t(hbm_size * 4 / BYTES_IN_HBM_LINE);
			// std::copy_n(ptr_start_in_l, in_l.m_num_lines[i] * WORDS_IN_HBM_LINE, (hbm_buffer_ptr[hbm_buffer_idx] + hbm_buffer_offset * WORDS_IN_HBM_LINE));
			hbm_buffer[hbm_buffer_idx].sync(XCL_BO_SYNC_BO_FROM_DEVICE, BYTES_IN_HBM_LINE, hbm_buffer_offset * BYTES_IN_HBM_LINE);

			cout << "obtain status:" << hbm_buffer_idx << "\t" << hbm_buffer_offset << endl;

			// for (j = 0; j < 256; j++){
			// 	cout << hbm_buffer_ptr[hbm_buffer_idx][j] << "\t";
			// }
			// cout << endl;

			num_matches[i] +=	hbm_buffer_ptr[hbm_buffer_idx][hbm_buffer_offset * WORDS_IN_HBM_LINE + 0];
			count_out_lines[i] +=	hbm_buffer_ptr[hbm_buffer_idx][hbm_buffer_offset * WORDS_IN_HBM_LINE + 1];
			num_repeats[i] +=	hbm_buffer_ptr[hbm_buffer_idx][hbm_buffer_offset * WORDS_IN_HBM_LINE + 2];
			count_repeat_lines[i] += hbm_buffer_ptr[hbm_buffer_idx][hbm_buffer_offset * WORDS_IN_HBM_LINE + 3];

			cout << "[INFO] Hash iteration: " << j << "\t kernel [" << i << "]: ====================";
			cout << "num_matches: " << num_matches[i] << endl;
			cout << "count_out_lines: " << count_out_lines[i] << endl;
			cout << "num_repeats: " << num_repeats[i] << endl;
			cout << "count_repeat_lines: " << count_repeat_lines[i] << endl;
		}
	
	}


	// copyback
	for (int i = 0; i < 8; i++) {
		hbm_buffer[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE, hbm_size * 4, 0);
	}

	// Verification
	if (do_verify == 1) {

		unsigned num_matches_sum;

		for (int i = 0; i < num_kernel; i++) {
			datamover_read(hbm_buffer_ptr, count_out_lines[i], &r1, i, in_l.m_num_lines[0] * WORDS_IN_HBM_LINE);
			datamover_read(hbm_buffer_ptr, count_out_lines[i], &r2, i, 0);
			num_matches_sum += num_matches[i];
		}

		cout << "r1.get_num_items() : " << r1.get_num_items() << endl;
		cout << "r2.get_num_items() : " << r2.get_num_items() << endl;

		hbm_column<uint32_t> sw_r1(num_kernel, num_matches_sum, 0);
		hbm_column<uint32_t> sw_r2(num_kernel, num_matches_sum, 0);
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

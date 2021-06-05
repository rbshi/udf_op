#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_ini.h"
#include "experimental/xrt_kernel.h"


#include "krnl_udf_selection.h"
#include "hbm_column.hpp"

#define RES_BUF_FACTOR 1.5
#define HBM_PARTITION

using namespace std;


void datamover_read(std::vector<int*> buffer_d_hbm_out,
                    hbm_column<int> *out, unsigned num_in_lines) {
  for (uint32_t p = 0; p < out->m_num_partitions; p++) {
    uint32_t num_lines = buffer_d_hbm_out[p][0 + 1];
    for (uint32_t i = 0; i < num_lines; i++) {
      for (uint32_t j = 0; j < INTS_IN_HBM_LINE; j++) {
        // NOTE: +1 due to the status address
        uint32_t temp = buffer_d_hbm_out[p][(0 + 1 + i) * INTS_IN_HBM_LINE + j];
        if (temp != 0xFFFFFFFF){
          // FIXME: offset of index, use [0] because it contains the avg value
          out->append(temp + INTS_IN_HBM_LINE * num_in_lines * p);
          // cout << "index: " << temp + INTS_IN_HBM_LINE * num_in_lines * p << endl;
        }
      }
    }
  }
}

int main(int argc, char **argv) {


  if (argc != 7) {
    cout << "Usage: ./testbench <num_values> <lower> <upper> <num_kernel> <num_times> <.xclbin>" << endl;
    return 1;
  }

  int num_values = atoi(argv[1]);
  int lower = atoi(argv[2]);
  int upper = atoi(argv[3]);
  int num_kernel = atoi(argv[4]);
  int num_times = atoi(argv[5]);
  std::string xclbin_fnm = argv[6];
  srand(3);

  if (xclbin_fnm.empty())
    throw std::runtime_error("FAILED_TEST\nNo xclbin specified");

  std::string cu_name = "krnl_udf_selection";

  unsigned int device_index = 0;

  auto device = xrt::device(device_index);
  auto uuid = device.load_xclbin(xclbin_fnm);
  auto krnl_all = xrt::kernel(device, uuid, cu_name);

#ifdef HBM_PARTITION
  hbm_column<int> in_column(num_kernel, num_values, 0, true, 4 * 1024 * 1024);
#else
  hbm_column<int> in_column(num_kernel, num_values, 0);
  cout << "[INFO] HBM_PARTITION is not set." << endl;
#endif
  
  in_column.populate_int_column(num_values, 'v', 's', 0xDEADBEEF);
  
  // Output
  // NOTE: out_column (along with the stats will come from 0 of each channel)
  hbm_column<int> out_column(num_kernel, num_values, 0, true, 4 * 1024 * 1024);


  // Allocate input buffer on HBM
  std::vector<xrt::bo> hbm_buffer(num_kernel);
  std::vector<int*> hbm_buffer_ptr(num_kernel);
  int hbm_size = (1<<28);

  int* ptr_start = (&in_column)->get_base();
#ifdef HBM_PARTITION   
    // each kernel uses one channel
    for (int i = 0; i < num_kernel * 1; i++) {
      hbm_buffer[i] = xrt::bo(device, hbm_size, 0, i);
      auto hbm_channel_ptr = hbm_buffer[i].map<int*>();
      hbm_buffer_ptr[i] = hbm_channel_ptr;
      // move data to hbm, NEED COPY FIRST..
      std::copy_n(ptr_start, in_column.m_num_lines[i] * INTS_IN_HBM_LINE, hbm_buffer_ptr[i]);
      hbm_buffer[i].sync(XCL_BO_SYNC_BO_TO_DEVICE, in_column.m_num_lines[i] * BYTES_IN_HBM_LINE, 0);

      ptr_start += in_column.m_num_lines[i] * INTS_IN_HBM_LINE;
    }
#else
    hbm_buffer[0] = xrt::bo(device, in_column.m_total_num_lines * BYTES_IN_HBM_LINE, krnl_all.group_id(0));
    auto hbm_channel_ptr = hbm_buffer[0].map<int*>();
    hbm_buffer_ptr[0] = hbm_channel_ptr;
    std::copy_n(ptr_start, in_column.m_total_num_lines * INTS_IN_HBM_LINE, hbm_buffer_ptr[0]);
    hbm_buffer[0].sync(XCL_BO_SYNC_BO_TO_DEVICE, in_column.m_total_num_lines * BYTES_IN_HBM_LINE, 0);
#endif
  std::cout << "Memory load finished\n";



  // run the kernel
  std::vector<xrt::run> runs(num_kernel);

  for (int i = 0; i < num_kernel; i++){
    
    addr_t in_addr = in_column.m_hbm_offset[i];
    // the first HBM line is reserved for status
    addr_t status_addr = out_column.m_hbm_offset[i];    
    // FIXME: disable the out_column.m_base_hbm_offset due to the `non-int div`
    addr_t out_addr = out_column.m_hbm_offset[i] + 1; 
    unsigned num_in_lines = in_column.m_num_lines[i];

    // cout << in_addr << '\t' << status_addr << '\t' << out_addr << '\t' << num_in_lines << endl;

    // obtain the krnl
    std::string cu_id = std::to_string(i + 1);
    std::string krnl_name_full = cu_name + ":{" + cu_name + "_" + cu_id + "}";
    auto krnl_inst = xrt::kernel(device, uuid, krnl_name_full, 0);
    auto run = krnl_inst(hbm_buffer[0], in_addr, out_addr, status_addr, num_in_lines, lower, upper, num_times);
    runs[i] = run;
  }

  for (auto &run : runs) {
    auto state = run.wait();
  }

  std::chrono::duration<double> kernel_time(0);
  auto kernel_start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_kernel; i++){
    runs[i].start();
  }

  for (int i = 0; i < num_kernel; i++){
    runs[i].wait();
  }


  auto kernel_end = std::chrono::high_resolution_clock::now();
  kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);

  std::cout << "Task finished\n";  


#ifdef HBM_PARTITION
    for (int i = 0; i < num_kernel; i++){
      hbm_buffer[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE, hbm_size, 0);
    }
#else
    hbm_buffer.clear(); hbm_buffer.resize(num_kernel);
    for (int i = 0; i < num_kernel * 1; i++) {
      hbm_buffer[i] = xrt::bo(device, hbm_size, 0, i);
      auto hbm_channel_ptr = hbm_buffer[i].map<int*>();
      hbm_buffer_ptr[i] = hbm_channel_ptr;
      hbm_buffer[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE, hbm_size, 0);
    }
#endif

  cout << "Copy back finish" << endl;


  double kernel_time_in_sec = kernel_time.count();
  kernel_time_in_sec /= num_kernel;

  double result = (float)num_values * num_times * sizeof(uint32_t) / num_kernel;
  result /= 1024;               // to KB
  result /= 1024;               // to MB
  result /= 1024;               // to GB
  result /= kernel_time_in_sec; // to GBps

  std::cout << "THROUGHPUT = " << result << " GB/s" << std::endl;


  // m_num_lines[0] have the average value
  datamover_read(hbm_buffer_ptr, &out_column, in_column.m_num_lines[0]);

  out_column.sort_items();

  hbm_column<uint32_t> sw_out_column(1, num_values, 0);
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
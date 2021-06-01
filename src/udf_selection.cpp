
#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "xcl2.hpp"

using namespace std;

// #define DEBUG

#define HBM_SIZE 32768
#define RES_BUF_FACTOR 1.5

#include "krnl_udf_selection.h"
#include "hbm_column.hpp"

// HBM Pseudo-channel(PC) requirements
#define MAX_HBM_PC_COUNT 32
#define PC_NAME(n) n | XCL_MEM_TOPOLOGY
const int pc[MAX_HBM_PC_COUNT + 4] = {
    PC_NAME(0),  PC_NAME(1),  PC_NAME(2),  PC_NAME(3),  PC_NAME(4),
    PC_NAME(5),  PC_NAME(6),  PC_NAME(7),  PC_NAME(8),  PC_NAME(9),
    PC_NAME(10), PC_NAME(11), PC_NAME(12), PC_NAME(13), PC_NAME(14),
    PC_NAME(15), PC_NAME(16), PC_NAME(17), PC_NAME(18), PC_NAME(19),
    PC_NAME(20), PC_NAME(21), PC_NAME(22), PC_NAME(23), PC_NAME(24),
    PC_NAME(25), PC_NAME(26), PC_NAME(27), PC_NAME(28), PC_NAME(29),
    PC_NAME(30), PC_NAME(31), PC_NAME(34), PC_NAME(35), PC_NAME(36),
    PC_NAME(37)};

void datamover_write(std::vector<unsigned int, aligned_allocator<unsigned int>> *src_d_hbm, hbm_column<int> *in) {
  uint32_t item_count = 0;
  for (uint32_t p = 0; p < in->m_num_partitions; p++) {
    // cout << "in->m_num_lines[" << p << "]: " << in->m_num_lines[p] << endl;
    for (uint32_t i = 0; i < in->m_num_lines[p]; i++) {
      for (uint32_t j = 0; j < INTS_IN_HBM_LINE; j++) {
        src_d_hbm[p][i * INTS_IN_HBM_LINE +j] = in->get_item(item_count++);
      }
    }
  }
}

void datamover_read(std::vector<unsigned int, aligned_allocator<unsigned int>> *src_d_hbm, hbm_column<uint32_t> *out, unsigned num_in_lines) {
  for (uint32_t p = 0; p < out->m_num_partitions; p++) {
    // obtain the num_lines, num_possitivies is ignored
    // cout << "out addr of buffer" << p << "=" << out->m_hbm_offset[p] * INTS_IN_HBM_LINE << endl;
    uint32_t num_lines = src_d_hbm[p][out->m_hbm_offset[p] * INTS_IN_HBM_LINE + 1];
    // cout << "num_lines: " << num_lines << endl;
    for (uint32_t i =0; i < num_lines; i++) {
      for (uint32_t j = 0; j < INTS_IN_HBM_LINE; j++) {
        // NOTE: +1 due to the status address
        uint32_t temp = src_d_hbm[p][(out->m_hbm_offset[p] + 1 + i) * INTS_IN_HBM_LINE + j];
        // uint32_t temp = (uint32_t)hbm_memory[out->m_hbm_offset[p] + 1 + i](
        //     BITS_IN_INT * (j + 1) - 1, BITS_IN_INT * j);
        if (temp != 0xFFFFFFFF)
          // FIXME: offset of index, use [0] because it contains the avg value
          out->append(temp + INTS_IN_HBM_LINE * num_in_lines * p);
    }
  }
  }
}

int main(int argc, char *argv[]) {

  if (argc != 7) {
    cout << "Usage: ./testbench <num_values_kernel> <lower> <upper> <num_kernel> <num_times> <.xclbin>" << endl;
    return 1;
  }

  int num_values_kernel = atoi(argv[1]);
  int lower = atoi(argv[2]);
  int upper = atoi(argv[3]);
  int num_kernel = atoi(argv[4]);
  int num_times = atoi(argv[5]);
  std::string binaryFile = argv[6];

  cout << "num_values_kernel: " << num_values_kernel << endl;
  cout << "lower: " << lower << endl;
  cout << "upper: " << upper << endl;  

  srand(3);

  unsigned int access_offset_stride = 0; // FIXME

  int num_values = num_values_kernel * num_kernel;

  cl_int err;
  cl::CommandQueue q;
  std::string krnl_name = "krnl_udf_selection";
  std::vector<cl::Kernel> krnls(num_kernel);
  cl::Context context;

  // find and program the fpga (folding)
  {
    // OPENCL HOST CODE AREA START
    // The get_xil_devices will return vector of Xilinx Devices
    auto devices = xcl::get_xil_devices();

    // read_binary_file() command will find the OpenCL binary file created using
    // V++ compiler load into OpenCL Binary and return pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);

    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
      auto device = devices[i];
      // Creating Context and Command Queue for selected Device
      OCL_CHECK(err,
                context = cl::Context(device, nullptr, nullptr, nullptr, &err));
      OCL_CHECK(err,
                q = cl::CommandQueue(context, device,
                                     CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                                         CL_QUEUE_PROFILING_ENABLE,
                                     &err));

      std::cout << "Trying to program device[" << i
                << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
      cl::Program program(context, {device}, bins, nullptr, &err);
      if (err != CL_SUCCESS) {
        std::cout << "Failed to program device[" << i
                  << "] with xclbin file!\n";
      } else {
        std::cout << "Device[" << i << "]: program successful!\n";
        // Creating Kernel object using Compute unit names

        for (int i = 0; i < num_kernel; i++) {
          std::string cu_id = std::to_string(i + 1);
          std::string krnl_name_full =
              krnl_name + ":{" + krnl_name + "_" + cu_id + "}";

          // printf("Creating a kernel [%s] for CU(%d)\n",
          // krnl_name_full.c_str(),
          //        i + 1);

          // Here Kernel object is created by specifying kernel name along with
          // compute unit.
          // For such case, this kernel object can only access the specific
          // Compute unit

          OCL_CHECK(err, krnls[i] =
                             cl::Kernel(program, krnl_name_full.c_str(), &err));
        }
        valid_device = true;
        break; // we break because we found a valid device
      }
    }

    if (!valid_device) {
      std::cout << "Failed to program any device found, exit!\n";
      exit(EXIT_FAILURE);
    }
  }


  // Input
  hbm_column<int> in_column(num_kernel, num_values, 0);
  in_column.populate_int_column(num_values, 'v', 's', 0xDEADBEEF);

  // Output
  // NOTE: here `in_column.m_base_hbm_offset` can be ignored
  hbm_column<uint32_t> out_column(num_kernel, num_values, in_column.m_base_hbm_offset);

  // Transfer data to HBM

  std::vector<unsigned int, aligned_allocator<unsigned int>>
      src_d_hbm[num_kernel];
  for (int i = 0; i < num_kernel; i++) {
    // one extra line for status
    src_d_hbm[i].resize((&in_column)->m_num_lines[i] * INTS_IN_HBM_LINE * (1 + RES_BUF_FACTOR) + INTS_IN_HBM_LINE);
  }

  datamover_write(src_d_hbm, &in_column);

  // NOTE: update the m_hbm_offset of *out, since last channel may have different sizes of in_line, and out offset will be different
  for (int i = 0; i < num_kernel; i++){
    out_column.m_hbm_offset[i] = in_column.m_base_hbm_offset + in_column.m_num_lines[i];
  }

  // init ocl buffer
  std::vector<cl_mem_ext_ptr_t> d_hbm_ext(num_kernel);
  std::vector<cl::Buffer> buffer_d_hbm(num_kernel);

  // For Allocating Buffer to specific Global Memory PC, user has to use
  // cl_mem_ext_ptr_t and provide the PCs
  for (int i = 0; i < num_kernel; i++) {
    d_hbm_ext[i].obj = src_d_hbm[i].data();
    d_hbm_ext[i].param = 0;
    d_hbm_ext[i].flags = pc[i];
    OCL_CHECK(err, buffer_d_hbm[i] = cl::Buffer(
                       context,
                       CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX |
                           CL_MEM_USE_HOST_PTR,
                       sizeof(uint32_t) * (src_d_hbm[i].size()), &d_hbm_ext[i], &err));
}

  // Copy input data to Device Global Memory
  for (int i = 0; i < num_kernel; i++) {
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_d_hbm[i]},
                                                    0 /* 0 means from host*/));
  }
  q.finish();


  std::chrono::duration<double> kernel_time(0);
  auto kernel_start = std::chrono::high_resolution_clock::now();

  // run
  for (int i = 0; i < num_kernel; i++){
    // addr_t in_addr = in_column.m_hbm_offset[i];
    // addr_t status_addr = out_column.m_hbm_offset[i];
    // // the first HBM line is reserved for status
    // addr_t out_addr = out_column.m_hbm_offset[i]+1;
    
    addr_t in_addr = in_column.m_base_hbm_offset;
    // the first HBM line is reserved for status
    // FIXME: disable the out_column.m_base_hbm_offset due to the `non-int div`
    addr_t out_addr = out_column.m_hbm_offset[i] + 1; 
    addr_t status_addr = out_column.m_hbm_offset[i];
    // FIXME: exact line number
    unsigned num_in_lines = in_column.m_num_lines[i];

    OCL_CHECK(err, err = krnls[i].setArg(0, buffer_d_hbm[i]));
    OCL_CHECK(err, err = krnls[i].setArg(1, in_addr));
    OCL_CHECK(err, err = krnls[i].setArg(2, out_addr));
    OCL_CHECK(err, err = krnls[i].setArg(3, status_addr));
    OCL_CHECK(err, err = krnls[i].setArg(4, num_in_lines));
    OCL_CHECK(err, err = krnls[i].setArg(5, lower));
    OCL_CHECK(err, err = krnls[i].setArg(6, upper));
    OCL_CHECK(err, err = krnls[i].setArg(7, num_times));
    // Invoking the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnls[i]));
  }

  q.finish();

  auto kernel_end = std::chrono::high_resolution_clock::now();

  kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);

  double kernel_time_in_sec = kernel_time.count();
  kernel_time_in_sec /= num_kernel;

  double result = (float)num_values * num_times * sizeof(uint32_t) / num_kernel;
  result /= 1024;               // to KB
  result /= 1024;               // to MB
  result /= 1024;               // to GB
  result /= kernel_time_in_sec; // to GBps

  std::cout << "THROUGHPUT = " << result << " GB/s" << std::endl;

  cout << "Finish" << endl;

  // Copy Result from Device Global Memory to Host Local Memory
  for (int i = 0; i < num_kernel; i++) {
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
                       {buffer_d_hbm[i]}, CL_MIGRATE_MEM_OBJECT_HOST));
  }
  q.finish();

  cout << "Copy back finish" << endl;

  // m_num_lines[0] have the average value
  datamover_read(src_d_hbm, &out_column, in_column.m_num_lines[0]);

  // cout << "out_column.get_num_items(): " << out_column.get_num_items() << endl;
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

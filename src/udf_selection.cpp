
#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "xcl2.hpp"

using namespace std;

// #define DEBUG
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

void datamover_read(std::vector<int, aligned_allocator<int>> *buffer_d_hbm_out,
                    hbm_column<int> *out, unsigned num_in_lines) {
  for (uint32_t p = 0; p < out->m_num_partitions; p++) {
    // obtain the num_lines, num_possitivies is ignored
    // cout << "out addr of buffer" << p << "=" << out->m_hbm_offset[p] *
    // INTS_IN_HBM_LINE << endl;
    uint32_t num_lines = buffer_d_hbm_out[p][0 + 1];
    cout << "num_lines: " << num_lines << endl;
    cout << "possitive: " << buffer_d_hbm_out[p][0] << endl;
    for (uint32_t i = 0; i < num_lines; i++) {
      for (uint32_t j = 0; j < INTS_IN_HBM_LINE; j++) {
        // NOTE: +1 due to the status address
        uint32_t temp = buffer_d_hbm_out[p][(0 + 1 + i) * INTS_IN_HBM_LINE + j];
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


  std::string krnl_name = "krnl_udf_selection";

  if (argc != 8) {
    cout << "Usage: ./testbench <num_values> <lower> <upper> <num_kernel> <num_times> <w/wo hbm partition, 0: non-partition, 1: partition> <.xclbin>" << endl;
    return 1;
  }

  int num_values = atoi(argv[1]);
  int lower = atoi(argv[2]);
  int upper = atoi(argv[3]);
  int num_kernel = atoi(argv[4]);
  int num_times = atoi(argv[5]);
  int hbm_partition = atoi(argv[6]);
  std::string binaryFile = argv[7];

  srand(3);

  std::vector<cl::Kernel> krnls(num_kernel);
  cl::Context context;
  cl_int err;
  cl::CommandQueue q;

  // find and program the fpga (folding)
  {
    // OPENCL HOST CODE AREA START
    // The get_xil_devices will return vector of Xilinx Devices
    auto devices = xcl::get_xil_devices();

    // read_binary_file() command will find the OpenCL binary file created using
    // V++ compiler load into OpenCL Binary and return pointer to file buffer.

    // auto fileBuf = xcl::read_binary_file(binaryFile);
    // cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

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

      // program fpga with xbutil with hw run, avoid server hang
      // if (xcl::is_emulation()) {
      //   std::cout << "Trying to program device[" << i
      //             << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
      //   cl::Program program(context, {device}, bins, nullptr, &err);
      // }
      // for (int i = 0; i < num_kernel; i++) {
      //   std::string cu_id = std::to_string(i + 1);
      //   std::string krnl_name_full =
      //       krnl_name + ":{" + krnl_name + "_" + cu_id + "}";
      //   OCL_CHECK(err,
      //             krnls[i] = cl::Kernel(program, krnl_name_full.c_str(), &err));
      // }
      // valid_device = true;
      // break; // we break because we found a valid device
    }

    // if (!valid_device) {
    //   std::cout << "Failed to program any device found, exit!\n";
    //   exit(EXIT_FAILURE);
    // }
  }

  // Input
  hbm_column<int> in_column(num_kernel, num_values, 0, true, 4 * 1024 * 1024);
  in_column.populate_int_column(num_values, 'u', '-', 0xDEADBEEF);
  
  // Output
  // NOTE: out_column (along with the stats will come from 0 of each channel)
  hbm_column<int> out_column(num_kernel, num_values, 0, true, 4 * 1024 * 1024);

  // Allocate input buffer on HBM
  // std::vector<cl_mem_ext_ptr_t> d_hbm_in_ext(num_kernel);
  // std::vector<cl::Buffer> buffer_d_hbm_in(num_kernel);

  // if (hbm_partition){
  //   int* ptr_start = (&in_column)->get_base();
  //   for (int i = 0; i < num_kernel; i++) {
  //     d_hbm_in_ext[i].obj = ptr_start;
  //     d_hbm_in_ext[i].param = 0;
  //     d_hbm_in_ext[i].flags = pc[i];
  //     OCL_CHECK(err, buffer_d_hbm_in[i] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR, in_column.m_num_lines[i] * BYTES_IN_HBM_LINE, &d_hbm_in_ext[i], &err));
  //     OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_d_hbm_in[i]}, 0 /* 0 means from host*/));
  //     ptr_start += in_column.m_num_lines[i] * INTS_IN_HBM_LINE;
  //     cout << "in_column.m_num_lines=" << in_column.m_num_lines[i] << endl;
  //   }
  // }
  // q.finish();
  // std::cout << "Memory load finished\n";


  // allocate hbm buffers with individual banks
  std::vector<cl_mem_ext_ptr_t> d_hbm_ext(num_kernel);
  // std::vector<cl::Buffer> buffer_d_hbm(num_kernel);
  std::vector<int *> d_hbm_ptr(num_kernel);

  cl::Buffer* buffer_d_hbm[num_kernel];


  for (int i = 0; i < num_kernel; i++) {
    d_hbm_ext[i].obj = NULL; // DONOT map now
    d_hbm_ext[i].param = 0;
    d_hbm_ext[i].flags = pc[i];
    // OCL_CHECK(err, buffer_d_hbm[i] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, size_t(1 << 28), &d_hbm_ext[i], &err));


    OCL_CHECK(err, buffer_d_hbm[0] = new cl::Buffer(context, CL_MEM_READ_WRITE, size_t(1024000), nullptr, &err));
    // get the handle of device buffer to host ptr
    // unsigned char* map_input_buffer0 = (unsigned char*)q.enqueueMapBuffer(*(buffer_d_hbm[i]), CL_FALSE, CL_MAP_WRITE_INVALIDATE_REGION, 0, size_t(100));
    unsigned char* map_input_buffer0;
    // map_input_buffer0 = (unsigned char*) malloc (1024000);
    OCL_CHECK(err,
              map_input_buffer0 = (unsigned char*)q.enqueueMapBuffer(*(buffer_d_hbm[0]), CL_FALSE, CL_MAP_WRITE_INVALIDATE_REGION, 0, size_t(1024000), nullptr, nullptr, &err));    
    OCL_CHECK(err, err = q.finish());
    // void* ptr = q.enqueueMapBuffer(buffer_d_hbm[i] ,true, CL_MAP_WRITE | CL_MAP_READ, 0, size_t(1 << 10), nullptr, nullptr, nullptr);
    // d_hbm_ptr[i] = reinterpret_cast<int *>(ptr);

    // for (int j = 0; j < 100; j++){
    //   d_hbm_ptr[i][j] = 0xffffffff;
    // }
    // memset(d_hbm_ptr[i], 0xffffffff, 1024);
  }




  // std::chrono::duration<double> kernel_time(0);
  // auto kernel_start = std::chrono::high_resolution_clock::now();

  // // run
  // for (int i = 0; i < num_kernel; i++){
    
  //   addr_t in_addr = in_column.m_hbm_offset[i];
  //   // the first HBM line is reserved for status
  //   addr_t status_addr = out_column.m_hbm_offset[i];    
  //   // FIXME: disable the out_column.m_base_hbm_offset due to the `non-int div`
  //   addr_t out_addr = out_column.m_hbm_offset[i] + 1; 
  //   unsigned num_in_lines = in_column.m_num_lines[i];

  //   cout << "kernel-" << i << ": in address: " << in_addr << endl;
  //   cout << "kernel-" << i << ": status address: " << status_addr << endl;    

  //   OCL_CHECK(err, err = krnls[i].setArg(0, buffer_d_hbm_in[0]));
  //   OCL_CHECK(err, err = krnls[i].setArg(1, in_addr));
  //   OCL_CHECK(err, err = krnls[i].setArg(2, out_addr));
  //   OCL_CHECK(err, err = krnls[i].setArg(3, status_addr));
  //   OCL_CHECK(err, err = krnls[i].setArg(4, num_in_lines));
  //   OCL_CHECK(err, err = krnls[i].setArg(5, lower));
  //   OCL_CHECK(err, err = krnls[i].setArg(6, upper));
  //   OCL_CHECK(err, err = krnls[i].setArg(7, num_times));

  //   OCL_CHECK(err, err = q.enqueueTask(krnls[i]));
  // }

  // std::cout << "Task load finished\n";

  // q.finish();

  // auto kernel_end = std::chrono::high_resolution_clock::now();

  // kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);

  // std::cout << "Task finished\n";  

  // double kernel_time_in_sec = kernel_time.count();
  // kernel_time_in_sec /= num_kernel;

  // double result = (float)num_values * num_times * sizeof(uint32_t) / num_kernel;
  // result /= 1024;               // to KB
  // result /= 1024;               // to MB
  // result /= 1024;               // to GB
  // result /= kernel_time_in_sec; // to GBps

  // std::cout << "THROUGHPUT = " << result << " GB/s" << std::endl;




  // // Allocate result buffer
  // float selectivity = (upper - lower) / float(num_values);
  // cout << "selectivity=" << selectivity << endl;
  // std::vector<int, aligned_allocator<int>> buf_hbm_out[num_kernel];
  // for (int i = 0; i < num_kernel; i++) {
  //   int size_buf_hbm_out = int((&in_column)->m_num_lines[i] * INTS_IN_HBM_LINE * RES_BUF_FACTOR * selectivity);
  //   cout << "size_buf_hbm_out=" << size_buf_hbm_out << endl;
  //   // one extra line for status
  //   buf_hbm_out[i].resize(size_buf_hbm_out + INTS_IN_HBM_LINE);
  // }
  
  // for (int i = 0; i < num_kernel; i++) {
  //   q.enqueueUnmapMemObject(buffer_d_hbm_in[i], &d_hbm_in_ext[i] /*pointer returned by Map call*/);
  //   q.enqueueReadBuffer(buffer_d_hbm_in[i], CL_TRUE, 0, sizeof(int) * buf_hbm_out[i].size(), buf_hbm_out[i].data());
  // }
  // q.finish();

  // // // Allocate output buffer on HMB
  // // std::vector<cl_mem_ext_ptr_t> d_hbm_out_ext(num_kernel);
  // // std::vector<cl::Buffer> buffer_d_hbm_out(num_kernel);
  // // // For Allocating Buffer to specific Global Memory PC, user has to use
  // // // cl_mem_ext_ptr_t and provide the PCs
  // // for (int i = 0; i < num_kernel; i++) {
  // //   // CHECK
  // //   d_hbm_out_ext[i].obj = buf_hbm_out[i].data();
  // //   d_hbm_out_ext[i].param = 0;
  // //   d_hbm_out_ext[i].flags = pc[i];
  // //   OCL_CHECK(err, buffer_d_hbm_out[i] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR, sizeof(int) * buf_hbm_out[i].size(), &d_hbm_out_ext[i], &err));
  // // }

  // // // Copy input data to Device Global Memory
  // // for (int i = 0; i < num_kernel; i++) {
  // //   OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_d_hbm_out[i]}, CL_MIGRATE_MEM_OBJECT_HOST));
  // // }
  // // q.finish();



  // cout << "Copy back finish" << endl;

  // // m_num_lines[0] have the average value
  // datamover_read(buf_hbm_out, &out_column, in_column.m_num_lines[0]);


  // // cout << "out_column.get_num_items(): " << out_column.get_num_items() << endl;
  // out_column.sort_items();


  // hbm_column<uint32_t> sw_out_column(1, num_values, 0);
  // for (unsigned i = 0; i < num_values; i++) {
  //   int value = in_column.get_item(i);
  //   if (value > lower && value < upper) {
  //     sw_out_column.append(i);
  //     cout << "value:" << value << endl;
  //   }
  // }
  // sw_out_column.sort_items();

  // bool success = true;
  // if (out_column.get_num_items() != sw_out_column.get_num_items()) {
  //   cout << "HW (" << out_column.get_num_items() << ") and SW ("
  //        << sw_out_column.get_num_items() << ") num_matches are different!"
  //        << endl;
  //   success = false;
  // } else {
  //   for (unsigned i = 0; i < out_column.get_num_items(); i++) {
  //     if (out_column.get_item(i) != sw_out_column.get_item(i)) {
  //       cout << "Mismatch at " << i << ": HW " << out_column.get_item(i)
  //            << ", SW " << sw_out_column.get_item(i) << endl;
  //       success = false;
  //     }
  //   }
  // }

  // if (success) {
  //   cout << "SUCCESS!" << endl;
  // } else {
  //   cout << "FAIL!" << endl;
  // }

  // return 0;
}

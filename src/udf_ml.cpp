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



// #define DEBUG

#include "krnl_udf_ml.h"
#include "udf_ml_models.hpp"

using namespace std;

#define HBM_SIZE 32768*512

void datamover_write(int* hbm_memory, dataset* ds)
{
	// Transfer samples
	for (unsigned i = 0; i < ds->m_num_samples; i++) {
		for (unsigned j = 0; j < ds->m_num_features_in_lines; j++) {
			for (uint32_t k = 0; k < FLOATS_IN_LINE; k++) {
				uint32_t* temp = (uint32_t*)&(ds->m_row_samples[i*ds->m_num_features_in_lines*FLOATS_IN_LINE + j*FLOATS_IN_LINE + k]);
				hbm_memory[(ds->m_samples_hbm_offset + i*ds->m_num_features_in_lines + j) * FLOATS_IN_LINE + k] = *temp;
			}
		}
	}
	// Transfer labels
	for (unsigned i = 0; i < ds->m_num_samples_in_lines; i++) {
		for (uint32_t k = 0; k < FLOATS_IN_LINE; k++) {
			uint32_t* temp = (uint32_t*)&(ds->m_labels[i*FLOATS_IN_LINE + k]);
			hbm_memory[(ds->m_labels_hbm_offset + i) * FLOATS_IN_LINE + k] = *temp;
		}
	}
}

void datamover_read(int* hbm_memory, models* ms)
{
	for (unsigned e = 0; e < ms->m_x_num_epochs; e++) {
		for (unsigned j = 0; j < ms->m_x_in_lines; j++) {
			for (uint32_t k = 0; k < FLOATS_IN_LINE; k++) {
				uint32_t temp = hbm_memory[(ms->m_x_hbm_offset + e*ms->m_x_in_lines + j)*FLOATS_IN_LINE+k];
				ms->m_x_hbm[e*ms->m_x_in_lines*FLOATS_IN_LINE + j*FLOATS_IN_LINE + k] = *((float*)(&temp));
			}
		}
	}
}

int main(int argc, char *argv[]) {

	unsigned num_samples = 128;
	unsigned num_features = 32;
	unsigned minibatch_size = 1;
	unsigned do_logreg = 0;
	unsigned num_epochs = 10;


	if (argc != 8) {
		printf("Usage: ./testbench <num_samples> <num_features> <minibatch_size> <do_logreg> <num_epochs> <num_kernel> <.xclbin>\n");
		return 1;
	}
	
	num_samples = atoi(argv[1]);
	num_features = atoi(argv[2]);
	minibatch_size = atoi(argv[3]);
	do_logreg = atoi(argv[4]);
	num_epochs = atoi(argv[5]);
  int num_kernel = atoi(argv[6]);
  std::string xclbin_fnm = argv[7];	


  if (xclbin_fnm.empty())
    throw std::runtime_error("FAILED_TEST\nNo xclbin specified");

  std::string cu_name = "krnl_udf_ml";

  unsigned int device_index = 0;

  auto device = xrt::device(device_index);
  auto uuid = device.load_xclbin(xclbin_fnm);
  auto krnl_all = xrt::kernel(device, uuid, cu_name);


	float step_size = 0.0001;
	float lambda = 0; //0.1;

	dataset dataset_inst(10);
	dataset_inst.generate_synthetic_data(num_samples, num_features, do_logreg == 1, norm_t::zero_to_one);

	uint64_t m_x_hbm_offset = dataset_inst.m_hbm_offset + dataset_inst.m_dataset_in_lines + dataset_inst.m_num_samples_in_lines;
	models models_inst(m_x_hbm_offset, &dataset_inst);

	// models_inst.sgd(NULL, num_epochs, do_logreg == 1, minibatch_size, step_size, lambda);

	models_inst.alloc_x_hbm(num_epochs);

  // Allocate input buffer on HBM
  std::vector<xrt::bo> hbm_buffer(num_kernel);
  std::vector<int*> hbm_buffer_ptr(num_kernel);
  int hbm_size = (1<<28);



  // each kernel uses one channel
  for (int i = 0; i < num_kernel * 1; i++) {
    hbm_buffer[i] = xrt::bo(device, hbm_size, 0, i);
    auto hbm_channel_ptr = hbm_buffer[i].map<int*>();
    hbm_buffer_ptr[i] = hbm_channel_ptr;
    // move data to hbm, NEED COPY FIRST..
  	datamover_write(hbm_buffer_ptr[i], &dataset_inst);
  	// m_x_hbm_offset is the size of base_offset + dataset_feature + dataset_label
    hbm_buffer[i].sync(XCL_BO_SYNC_BO_TO_DEVICE, m_x_hbm_offset * BYTES_IN_HBM_LINE, 0);
  }

  std::cout << "Memory load finished\n";


	uint64_t samples_addr = dataset_inst.m_samples_hbm_offset;
	uint64_t labels_addr = dataset_inst.m_labels_hbm_offset;
	uint64_t model_addr = models_inst.m_x_hbm_offset;


  // run the kernel
  std::vector<xrt::run> runs(num_kernel);
  for (int i = 0; i < num_kernel; i++){
    // obtain the krnl
    std::string cu_id = std::to_string(i + 1);
    std::string krnl_name_full = cu_name + ":{" + cu_name + "_" + cu_id + "}";
    auto krnl_inst = xrt::kernel(device, uuid, krnl_name_full, 0);
    auto run = krnl_inst(
			hbm_buffer[i],
			samples_addr,
			labels_addr,
			model_addr,
			step_size/minibatch_size,
			(step_size*lambda)/minibatch_size,
			do_logreg == 1,
			minibatch_size,
			num_epochs,
			dataset_inst.m_num_features_in_lines,
			dataset_inst.m_num_samples);
    runs[i] = run;
  }


  for (auto &run : runs) {
    auto state = run.wait();
  }

	// copy back
  for (int i = 0; i < num_kernel * 1; i++) {
  	cout << "Now presenting mdoel from kernel["  << i << "]." << endl;
  	// m_x_hbm_offset is the size of base_offset + dataset_feature + dataset_label
  	int size_copyback = models_inst.m_x_num_epochs * models_inst.m_x_in_lines * BYTES_IN_LINE;
    hbm_buffer[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE, size_copyback, m_x_hbm_offset * BYTES_IN_HBM_LINE);
  	datamover_read(hbm_buffer_ptr[i],  &models_inst);
		models_inst.print_x_hbm_loss(lambda, do_logreg == 1);
  }

	return 0;
}


#include <sys/time.h>
#include <iostream>

using namespace std;

// #define DEBUG

#include "krnl_udf_ml.h"
#include "udf_ml_models.hpp"

#define HBM_SIZE 32768*512

void datamover_write(hbm_t *hbm_memory, dataset* ds)
{
	// Transfer samples
	for (unsigned i = 0; i < ds->m_num_samples; i++) {
		for (unsigned j = 0; j < ds->m_num_features_in_lines; j++) {
			for (uint32_t k = 0; k < FLOATS_IN_LINE; k++) {
				uint32_t* temp = (uint32_t*)&(ds->m_row_samples[i*ds->m_num_features_in_lines*FLOATS_IN_LINE + j*FLOATS_IN_LINE + k]);
				hbm_memory[ds->m_samples_hbm_offset + i*ds->m_num_features_in_lines + j](BITS_IN_FLOAT*(k+1)-1, BITS_IN_FLOAT*k) = *temp;
			}
		}
	}
	// Transfer labels
	for (unsigned i = 0; i < ds->m_num_samples_in_lines; i++) {
		for (uint32_t k = 0; k < FLOATS_IN_LINE; k++) {
			uint32_t* temp = (uint32_t*)&(ds->m_labels[i*FLOATS_IN_LINE + k]);
			hbm_memory[ds->m_labels_hbm_offset + i](BITS_IN_FLOAT*(k+1)-1, BITS_IN_FLOAT*k) = *temp;
		}
	}
}

void datamover_read(hbm_t *hbm_memory, models* ms)
{
	for (unsigned e = 0; e < ms->m_x_num_epochs; e++) {
		for (unsigned j = 0; j < ms->m_x_in_lines; j++) {
			for (uint32_t k = 0; k < FLOATS_IN_LINE; k++) {
				uint32_t temp = hbm_memory[ms->m_x_hbm_offset + e*ms->m_x_in_lines + j](BITS_IN_FLOAT*(k+1)-1, BITS_IN_FLOAT*k);
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


	// if (argc != 6) {
	// 	printf("Usage: ./testbench <num_samples> <num_features> <minibatch_size> <do_logreg> <num_epochs>\n");
	// 	return 1;
	// }
	
	// num_samples = atoi(argv[1]);
	// num_features = atoi(argv[2]);
	// minibatch_size = atoi(argv[3]);
	// do_logreg = atoi(argv[4]);
	// num_epochs = atoi(argv[5]);

	float step_size = 0.0001;
	float lambda = 0; //0.1;

	dataset dataset_inst(10);
	dataset_inst.generate_synthetic_data(num_samples, num_features, do_logreg == 1, norm_t::zero_to_one);

	uint64_t m_x_hbm_offset = dataset_inst.m_hbm_offset + dataset_inst.m_dataset_in_lines + dataset_inst.m_num_samples_in_lines;
	models models_inst(m_x_hbm_offset, &dataset_inst);

	models_inst.sgd(NULL, num_epochs, do_logreg == 1, minibatch_size, step_size, lambda);

	// Transfer data to HBM
	models_inst.alloc_x_hbm(num_epochs);


	hbm_t* hbm_memory = (hbm_t*)malloc(HBM_SIZE*BYTES_IN_HBM_LINE);

	datamover_write(hbm_memory, &dataset_inst);

	uint64_t samples_addr = dataset_inst.m_samples_hbm_offset;
	uint64_t labels_addr = dataset_inst.m_labels_hbm_offset;
	uint64_t model_addr = models_inst.m_x_hbm_offset;

	krnl_udf_ml(
		hbm_memory,
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

	datamover_read(hbm_memory, &models_inst);

	models_inst.print_x_hbm_loss(lambda, do_logreg == 1);

	free(hbm_memory);

	return 0;
}

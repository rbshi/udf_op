
#include <stdint.h>
#include <string>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#include <sys/time.h>


#define ALIGNMENT BYTES_IN_LINE*2


enum norm_t {zero_to_one, minus_one_to_one};
enum norm_direction_t {row, column};

using namespace std;

class dataset {
public:
	float** m_samples;
	float* m_labels;
	float* m_row_samples;

	uint32_t m_num_samples;
	uint32_t m_num_features;

	norm_t m_samples_norm;
	norm_t m_labels_norm;

	float* m_samples_range;
	float* m_samples_min;
	float m_labels_range;
	float m_labels_min;

	bool m_samples_biased;

	// HBM side
	uint64_t m_hbm_offset;
	uint64_t m_samples_hbm_offset;
	uint64_t m_labels_hbm_offset;
	uint32_t m_num_samples_in_lines;
	uint32_t m_num_features_in_lines;
	uint32_t m_dataset_in_lines;

	dataset (uint64_t hbm_offset) {
		m_samples = nullptr;
		m_labels = nullptr;
		m_row_samples = nullptr;

		m_samples_range = nullptr;
		m_samples_min = nullptr;

		m_hbm_offset = hbm_offset;
	}

	~dataset () {
		dealloc_data();

		if (m_samples_range != nullptr) {
			free(m_samples_range);
		}
		if (m_samples_min != nullptr) {
			free(m_samples_min);
		}
	}

	void print_samples(uint32_t num) {
		for (uint32_t i = 0; i < num; i++) {
			cout << "sample " << i << ": " << endl;
			for (uint32_t j = 0; j < m_num_features; j++) {
				cout << m_samples[j][i] << " ";
			}
			cout << endl;
			cout << "label " << i << ": " << m_labels[i] << endl;
		}
	}

	void load_raw_data(char* path_to_file, uint32_t num_samples, uint32_t num_features, bool label_present) {
		cout << "load_raw_data is reading " << path_to_file << endl;

		m_samples_biased = true;
		m_num_samples = num_samples;
		m_num_features = num_features+1; // For the bias term
		
		realloc_data();

		FILE* f = fopen(path_to_file, "r");
		if (f == NULL) {
			cout << "Can't find files at path_to_file" << endl;
			exit(1);
		}

		double* temp;
		
		uint32_t num_features_without_bias = m_num_features-1;

		if (label_present) {
			temp = (double*)malloc(m_num_samples*(num_features_without_bias+1)*sizeof(double));
			fread(temp, sizeof(double), m_num_samples*(num_features_without_bias+1), f);
			for (uint32_t i = 0; i < m_num_samples; i++) {
				m_labels[i] = (float)temp[i*(num_features_without_bias+1)];
				for (uint32_t j = 0; j < num_features_without_bias; j++) {
					m_samples[j+1][i] = (float)temp[i*(num_features_without_bias+1) + j + 1];
				}
			}
		}
		else {
			temp = (double*)malloc(m_num_samples*num_features_without_bias*sizeof(double));
			fread(temp, sizeof(double), m_num_samples*num_features_without_bias, f);
			for (uint32_t i = 0; i < m_num_samples; i++) {
				m_labels[i] = 0;
				for (uint32_t j = 0; j < num_features_without_bias; j++) {
					m_samples[j+1][i] = (float)temp[i*num_features_without_bias + j];
				}
			}
		}

		for (uint32_t i = 0; i < m_num_samples; i++) { // Bias term
			m_samples[0][i] = 1.0;
		}

		free(temp);
		fclose(f);

		populate_row_samples();

		cout << "m_num_samples: " << m_num_samples << endl;
		cout << "m_num_features: " << m_num_features << endl;
	}

	void generate_synthetic_data(uint32_t num_samples, uint32_t num_features, bool label_binary, norm_t labels_norm) {
		m_num_samples = num_samples;
		m_num_features = num_features;
		m_samples_biased = false;
		m_labels_norm = labels_norm;

		realloc_data();
		
		srand(7);
		float* x = (float*)malloc(m_num_features*sizeof(float));
		for (uint32_t j = 0; j < m_num_features; j++) {
			x[j] = ((float)rand())/RAND_MAX;
		}

		for (uint32_t i = 0; i < m_num_samples; i++) {
			if (label_binary) {
				float temp = ((float)rand())/RAND_MAX;
				if (temp > 0.5) {
					m_labels[i] = 1.0;
				}
				else {
					if (labels_norm == minus_one_to_one) {
						m_labels[i] = -1.0;
					}
					else {
						m_labels[i] = 0.0;
					}
				}
			}
			else {
				m_labels[i] = (float)rand()/RAND_MAX;
			}
			for (uint32_t j = 0; j < m_num_features; j++) {
				m_samples[j][i] = m_labels[i]*x[j] + 0.001*(float)rand()/(RAND_MAX);
			}
		}

		free(x);
		
		populate_row_samples();

		cout << "m_num_samples: " << m_num_samples << endl;
		cout << "m_num_features: " << m_num_features << endl;
	}

	void copy_from_dataset(dataset* ds) {
		m_num_samples = ds->m_num_samples;
		m_num_features = ds->m_num_features;
		m_samples_biased = ds->m_samples_biased;
		m_labels_norm = ds->m_labels_norm;

		realloc_data();

		for (uint32_t i = 0; i < m_num_samples; i++) {
			m_labels[i] = ds->m_labels[i];
			for (uint32_t j = 0; j < m_num_features; j++) {
				m_samples[j][i] = ds->m_samples[j][i];
			}
		}

		populate_row_samples();
	}

	void normalize_samples(norm_t norm, norm_direction_t direction) {
		m_samples_norm = norm;

		if (direction == row) {
			m_samples_range = (float*)realloc(m_samples_range, m_num_samples*sizeof(float));
			m_samples_min = (float*)realloc(m_samples_min, m_num_samples*sizeof(float));

			for (uint32_t i = 0; i < m_num_samples; i++) {
				float samples_min = numeric_limits<float>::max();
				float samples_max = -numeric_limits<float>::max();
				for (uint32_t j = 0; j < m_num_features; j++) {
					if (m_samples[j][i] > samples_max) {
						samples_max = m_samples[j][i];
					}
					if (m_samples[j][i] < samples_min) {
						samples_min = m_samples[j][i];
					}
				}
				float samples_range = samples_max - samples_min;
				if (samples_range > 0) {
					if (m_samples_norm == minus_one_to_one) {
						for (uint32_t j = 0; j < m_num_features; j++) {
							m_samples[j][i] = ((m_samples[j][i] - samples_min)/samples_range)*2.0-1.0;
						}
					}
					else {
						for (uint32_t j = 0; j < m_num_features; j++) {
							m_samples[j][i] = ((m_samples[j][i] - samples_min)/samples_range);
						}
					}
				}
				m_samples_range[i] = samples_range;
				m_samples_min[i] = samples_min;
			}
		}
		else {
			m_samples_range = (float*)realloc(m_samples_range, m_num_features*sizeof(float));
			m_samples_min = (float*)realloc(m_samples_min, m_num_features*sizeof(float));

			uint32_t start_coordinate = m_samples_biased ? 1 : 0;
			m_samples_range[0] = 0.0;
			m_samples_min[0] = 0.0;
			for (uint32_t j = start_coordinate; j < m_num_features; j++) {
				float samples_min = numeric_limits<float>::max();
				float samples_max = -numeric_limits<float>::max();
				for (uint32_t i = 0; i < m_num_samples; i++) {
					if (m_samples[j][i] > samples_max) {
						samples_max = m_samples[j][i];;
					}
					if (m_samples[j][i] < samples_min) {
						samples_min = m_samples[j][i];;
					}
				}
				float samples_range = samples_max - samples_min;
				if (samples_range > 0) {
					if (m_samples_norm == minus_one_to_one) {
						for (uint32_t i = 0; i < m_num_samples; i++) {
							m_samples[j][i] = ((m_samples[j][i] - samples_min)/samples_range)*2.0-1.0;
						}
					}
					else {
						for (uint32_t i = 0; i < m_num_samples; i++) {
							m_samples[j][i] = ((m_samples[j][i] - samples_min)/samples_range);
						}
					}
				}
				m_samples_range[j] = samples_range;
				m_samples_min[j] = samples_min;
			}
		}
		populate_row_samples();
	}

	void normalize_labels(norm_t norm, bool binarize_labels, float labels_to_binarize_to) {
		m_labels_norm = norm;

		if (!binarize_labels) {
			float labels_min = numeric_limits<float>::max();
			float labels_max = -numeric_limits<float>::max();
			for (uint32_t i = 0; i < m_num_samples; i++) {
				if (m_labels[i] > labels_max) {
					labels_max = m_labels[i];
				}
				if (m_labels[i] < labels_min) {
					labels_min = m_labels[i];
				}
			}
			
			float labels_range = labels_max - labels_min;
			if (labels_range > 0) {
				if (m_labels_norm == minus_one_to_one) {
					for (uint32_t i = 0; i < m_num_samples; i++) {
						m_labels[i] = ((m_labels[i] - labels_min)/labels_range)*2.0 - 1.0;
					}
				}
				else {
					for (uint32_t i = 0; i < m_num_samples; i++) {
						m_labels[i] = (m_labels[i] - labels_min)/labels_range;
					}
				}
			}
			m_labels_min = labels_min;
			m_labels_range = labels_range;
		}
		else {
			for (uint32_t i = 0; i < m_num_samples; i++) {
				if(m_labels[i] == labels_to_binarize_to) {
					m_labels[i] = 1.0;
				}
				else {
					if (m_labels_norm == minus_one_to_one) {
						m_labels[i] = -1.0;
						m_labels_min = -1.0;
						m_labels_range = 2.0;
					}
					else {
						m_labels[i] = 0.0;
						m_labels_min = 0.0;
						m_labels_range = 1.0;
					}
				}
			}
		}
	}


private:
	void populate_row_samples() {
		if (m_row_samples == nullptr) {
			return;
		}
		for (uint32_t i = 0; i < m_num_samples; i++) {
			for (uint32_t j = 0; j < m_num_features; j++) {
				m_row_samples[i*m_num_features_in_lines*FLOATS_IN_LINE + j] = m_samples[j][i];
			}
		}
	}

	void realloc_data() {
		dealloc_data();

		m_num_samples_in_lines = m_num_samples/FLOATS_IN_LINE + (m_num_samples%FLOATS_IN_LINE > 0);
		m_num_samples_in_lines += (m_num_samples_in_lines%2 == 0) ? 0 : 1;
		m_num_features_in_lines = m_num_features/FLOATS_IN_LINE + (m_num_features%FLOATS_IN_LINE > 0);
		m_dataset_in_lines = m_num_samples*m_num_features_in_lines;
		m_dataset_in_lines += (m_dataset_in_lines%2 == 0) ? 0 : 1;

		m_samples_hbm_offset = m_hbm_offset;
		m_labels_hbm_offset = m_samples_hbm_offset + m_dataset_in_lines;

		// Allocate memory
		m_samples = (float**)malloc(m_num_features*sizeof(float*));
		for (uint32_t j = 0; j < m_num_features; j++) {
			posix_memalign((void**)&m_samples[j], ALIGNMENT, m_num_samples*sizeof(float));
		}
		posix_memalign((void**)&m_labels, ALIGNMENT, m_num_samples_in_lines*FLOATS_IN_LINE*sizeof(float));
		posix_memalign((void**)&m_row_samples, ALIGNMENT, m_dataset_in_lines*FLOATS_IN_LINE*sizeof(float));
	}

	void dealloc_data() {
		// dealloc if not nullptr
		if (m_samples != nullptr) {
			cout << "Freeing m_samples..." << endl;
			for (uint32_t j = 0; j < m_num_features; j++) {
				free(m_samples[j]);
			}
			free(m_samples);
		}
		if (m_labels != nullptr) {
			cout << "Freeing m_labels..." << endl;
			free(m_labels);
		}
		if (m_row_samples != nullptr) {
			free(m_row_samples);
		}
	}
};
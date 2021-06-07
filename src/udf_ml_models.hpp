
#include "udf_ml_dataset.hpp"
#include "math.h"

using namespace std;

class models {
public:
	dataset* m_dataset;
	float* m_x_hbm;
	uint64_t m_x_hbm_offset;
	uint32_t m_x_num_epochs;
	uint32_t m_x_in_lines;

	models(uint64_t x_hbm_offset, dataset* dataset) {
		m_x_hbm_offset = x_hbm_offset;
		m_dataset = dataset;
		m_x_hbm = NULL;
	}

	void alloc_x_hbm(uint32_t num_epochs) {
		m_x_num_epochs = num_epochs;
		m_x_in_lines = m_dataset->m_num_features_in_lines;

		if (m_x_hbm != NULL) {
			free(m_x_hbm);
		}
		posix_memalign((void**)&m_x_hbm, ALIGNMENT, m_x_num_epochs*m_x_in_lines*FLOATS_IN_LINE*sizeof(float));
		memset(m_x_hbm, 0, m_x_num_epochs*m_x_in_lines*FLOATS_IN_LINE*sizeof(float));
	}

	void print_x_hbm_loss(float lambda, bool do_logreg) {
		for (uint32_t e = 0; e < m_x_num_epochs; e++) {
			if (do_logreg) {
				cout << "epoch " << e << ": " << logreg_loss(m_x_hbm + e*m_x_in_lines*FLOATS_IN_LINE, lambda) << endl;
			}
			else {
				cout << "epoch " << e << ": " << linreg_loss(m_x_hbm + e*m_x_in_lines*FLOATS_IN_LINE, lambda) << endl;
			}
		}
	}

	float linreg_loss(float* x, float lambda) {
		float loss = 0;
		for(uint32_t i = 0; i < m_dataset->m_num_samples; i++) {
			float label = m_dataset->m_labels[i];
			float dot = get_dot(x, i);
			loss += (dot - label)*(dot - label);
		}
		loss /= (float)(2*m_dataset->m_num_samples);
		loss += l2_regularization(x, lambda);

		return loss;
	}

	float logreg_loss(float* x, float lambda) {
		float loss = 0;
		for(uint32_t i = 0; i < m_dataset->m_num_samples; i++) {
			float label = m_dataset->m_labels[i];
			float dot = get_dot(x, i);
			float prediction = 1.0/(1.0 + exp(-dot));
			float positiveLoss = log(prediction);
			float negativeLoss = log(1 - prediction);
			if (isinf(positiveLoss)) {
				positiveLoss = -std::numeric_limits<float>::max();
			}
			if (isinf(negativeLoss)) {
				negativeLoss = -std::numeric_limits<float>::max();
			}

			loss += label*positiveLoss + (1-label)*negativeLoss;
		}
		loss /= (float)(m_dataset->m_num_samples);
		loss = -loss;
		loss += l2_regularization(x, lambda);

		return loss;
	}

	void sgd(
		float* x_history,
		uint32_t num_epochs,
		bool do_logreg,
		uint32_t minibatch_size,
		float step_size,
		float lambda)
	{
		float* x = (float*)aligned_alloc(64, m_dataset->m_num_features*sizeof(float));
		memset(x, 0, m_dataset->m_num_features*sizeof(float));
		float* gradient = (float*)aligned_alloc(64, m_dataset->m_num_features*sizeof(float));
		memset(gradient, 0, m_dataset->m_num_features*sizeof(float));

		cout << "SGD ---------------------------------------" << endl;
		uint32_t num_minibatches = m_dataset->m_num_samples/minibatch_size;
		cout << "num_minibatches: " << num_minibatches << endl;
		uint32_t rest = m_dataset->m_num_samples - num_minibatches*minibatch_size;
		cout << "rest: " << rest << endl;

		if (x_history == nullptr) {
			if (do_logreg) {
				cout << "Initial loss: " << logreg_loss(x, lambda) << endl;
			}
			else {
				cout << "Initial loss: " << linreg_loss(x, lambda) << endl;
			}
		}

		float scaled_step_size = step_size/minibatch_size;
		float scaled_lambda = step_size*lambda;
		for(uint32_t epoch = 0; epoch < num_epochs; epoch++) {
			for (uint32_t k = 0; k < num_minibatches; k++) {
				uint32_t offset = k*minibatch_size;
				for (uint32_t i = 0; i < minibatch_size; i++) {
					float dot = get_dot(x, offset + i);
					if (do_logreg) {
						dot = 1.0/(1.0 + exp(-dot));
					}
					// cout << "--------------------------------------------" << endl;
					// cout << offset + i << ", dot: " << dot << endl;
					// cout << "m_dataset->m_labels[" << offset+i << "]: " << m_dataset->m_labels[offset+i] << endl;

					float step = (dot - m_dataset->m_labels[offset+i]);
					update_gradient(gradient, step, offset + i);
				}
				update_model(x, gradient, scaled_step_size, scaled_lambda);
			}

			if (x_history != nullptr) {
				copy_model(x_history, x, epoch);
			}
			else {
				if (do_logreg) {
					cout << logreg_loss(x, lambda) << endl;
				}
				else {
					cout << linreg_loss(x, lambda) << endl;
				}
			}
		}

		free(x);
		free(gradient);
	}

private:

	float l2_regularization(float* x, float lambda) {
		float regularizer = 0;
		for (uint32_t j = 0; j < m_dataset->m_num_features; j++) {
			regularizer += x[j]*x[j];
		}
		regularizer *= (lambda*0.5)/m_dataset->m_num_samples;
		return regularizer;
	}

	inline float get_dot(float* x, uint32_t sample_index) {
		float dot = 0.0;
		uint32_t offset = sample_index*m_dataset->m_num_features_in_lines*FLOATS_IN_LINE;
		for (uint32_t j = 0; j < m_dataset->m_num_features; j++) {
			dot += x[j]*m_dataset->m_row_samples[offset + j];
		}
		return dot;
	}

	inline void update_gradient(float* gradient, float step, uint32_t sample_index) {
		uint32_t offset = sample_index*m_dataset->m_num_features_in_lines*FLOATS_IN_LINE;
		for (uint32_t j = 0; j < m_dataset->m_num_features; j++) {
			gradient[j] += step*m_dataset->m_row_samples[offset + j];
		}
	}

	inline void update_model(float* x , float* gradient, float scaled_step_size, float scaled_lambda) {
		for (uint32_t j = 0; j < m_dataset->m_num_features; j++) {
			x[j] -= (scaled_step_size*gradient[j] + scaled_lambda*x[j]);
			gradient[j] = 0;
		}
	}

	inline void copy_model(float* x_history, float* x, uint32_t epoch) {
		for (uint32_t j = 0; j < m_dataset->m_num_features; j++) {
			x_history[epoch*m_dataset->m_num_features + j] = x[j];
		}
	}
};
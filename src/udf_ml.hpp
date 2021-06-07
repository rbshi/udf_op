
#include "udf_ml_models.hpp"

typedef union {
	struct {
		uint64_t m_samples_addr;
		uint64_t m_labels_addr;
		uint64_t m_model_addr;
		float m_step_size;
		float m_lambda;
		uint32_t m_minibatch_size;
		uint32_t m_num_epochs;
		uint32_t m_num_features_in_lines;
		uint32_t m_num_samples;
		uint32_t do_logreg;
	} reg;
	uint32_t val[32];
} ml_config_t;

class ml {
private:
	ifpga* m_ifpga;
	datamover* m_datamover;
	uint64_t m_hbm_config_offset;
	uint32_t m_which_datamover = 0;
	ml_config_t* m_config;
	models* m_models;

	struct timespec pause;
public:
	uint64_t m_id;

	ml(ifpga* ifpga_inst,
		datamover* datamover_inst,
		uint64_t id,
		models* models_inst)
	{
		m_ifpga = ifpga_inst;
		m_datamover = datamover_inst;
		m_id = id;
		m_hbm_config_offset = (2*m_id << 8) | m_id;
		m_models = models_inst;
		posix_memalign((void**)&m_config, ALIGNMENT, sizeof(ml_config_t));

#ifdef SIM
		pause.tv_sec = 1;
#else
		pause.tv_sec = 0;
#endif
		pause.tv_nsec = 1000;
	}

	void sgd_fpga(
		float* x_history,
		uint32_t num_epochs,
		bool do_logreg,
		uint32_t minibatch_size,
		float step_size,
		float lambda)
	{
		m_models->alloc_x_hbm(num_epochs);

		m_config->reg.m_samples_addr = m_models->m_dataset->m_samples_hbm_offset;
		m_config->reg.m_labels_addr = m_models->m_dataset->m_labels_hbm_offset;
		m_config->reg.m_model_addr =  m_models->m_x_hbm_offset;
		m_config->reg.m_step_size = step_size/minibatch_size;
		m_config->reg.m_lambda = (step_size*lambda)/minibatch_size;
		m_config->reg.m_minibatch_size = minibatch_size;
		m_config->reg.m_num_epochs = num_epochs;
		m_config->reg.m_num_features_in_lines =  m_models->m_dataset->m_num_features_in_lines;
		m_config->reg.m_num_samples =  m_models->m_dataset->m_num_samples;
		m_config->reg.do_logreg = do_logreg ? 1 : 0;

		m_ifpga->set_config(3, m_hbm_config_offset);

		move_data_args_t args;

		// Set model
		args.m_which_datamover = m_which_datamover;
		args.m_dataflow_config = 1;
		args.m_input = (char*)m_models->m_x_hbm;
		args.m_in_num_lines = num_epochs*m_models->m_x_in_lines;
		args.m_output = 0;
		args.m_out_num_lines = num_epochs*m_models->m_x_in_lines;
		args.m_hbm_offset_lines = m_models->m_x_hbm_offset;

		m_datamover->datamover_thread(&args);

		// Transfer config
		args.m_which_datamover = m_which_datamover;
		args.m_dataflow_config = 1;
		args.m_input = (char*)m_config;
		args.m_in_num_lines = 2;
		args.m_output = NULL;
		args.m_out_num_lines = 2;
		args.m_hbm_offset_lines = 2*m_id;
		
		nanosleep(&pause, NULL);
		m_datamover->datamover_thread(&args);

		double start = get_time();

		transaction();
		double end = get_time();
		cout << "ml " << m_id << ", transaction time: " << end-start << endl;

		args.m_which_datamover = m_which_datamover;
		args.m_dataflow_config = 2;
		args.m_input = NULL;
		args.m_in_num_lines = num_epochs*m_models->m_x_in_lines;
		args.m_output = (char*)m_models->m_x_hbm;
		args.m_out_num_lines = num_epochs*m_models->m_x_in_lines;
		args.m_hbm_offset_lines = m_models->m_x_hbm_offset;

		if (x_history == nullptr) {
			// Get trained model
			m_datamover->datamover_thread(&args);
			m_models->print_x_hbm_loss(lambda, do_logreg);
		}
	}

	void transaction() {
		m_ifpga->set_bit_config(0, m_id + m_datamover->get_num_datamovers());

		unsigned timeout = 200000;
		while ( (m_ifpga->get_bit_config(12, m_id + m_datamover->get_num_datamovers()) == false) && timeout > 0 ) {
			nanosleep(&pause, NULL);
			timeout--;
		}

		if (timeout == 0) {
			printf("timeout has expired!!!\n");
		}
#ifdef OLAP_VERBOSE
		else {
			printf("timeout remaining: %d\n", timeout);
		}
#endif
		m_ifpga->reset_bit_config(0, m_id + m_datamover->get_num_datamovers());
	}
};
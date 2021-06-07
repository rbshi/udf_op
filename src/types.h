#ifndef TYPES_HH
#define TYPES_HH

// #define AP_INT_MAX_W 2048

#include <ap_int.h>

typedef ap_uint<253> resp_combo_t; //tlrflit (252:215) + tlxscflit_t (214:0)
typedef ap_uint<32> count_t;
typedef ap_uint<513> ocxflitbdi_t; 
typedef ap_uint<512> ocxflit_t; 
typedef ap_uint<256> ocxflitH_t;
typedef ap_uint<1> valid_t;
typedef ap_uint<199> tlxcflit_t;
typedef ap_uint<38> tlrflit_t;
typedef ap_uint<215> tlxscflit_t;
typedef ap_uint<3> dlsize_t;
typedef ap_uint<36> cutimer_t;
typedef ap_uint<20> pasid_t;
typedef ap_uint<4> stream_t;
typedef ap_uint<12> actag_t;
typedef ap_uint<16> bdf_t;
typedef ap_uint<3> pl_t;
typedef ap_uint<32> addr_t;
typedef ap_uint<4> code_t;
typedef ap_uint<2> dp_t;
typedef ap_uint<1> pad_t;
typedef ap_uint<6> nid_t;
typedef ap_uint<8> opcode_t;
typedef ap_uint<2> dl_t;
typedef ap_uint<16> tag_t;
typedef ap_uint<18> tag_dl_t;
typedef ap_uint<1> signal_t;
typedef ap_uint<3> sel_t;
typedef ap_uint<10> combo_t;
typedef ap_uint<10> fifocnt_t;
typedef ap_uint<8> token_t;
typedef ap_uint<4> xlatecnt_t;

// doppiodb additions
typedef ap_uint<64> config_t;
typedef ap_uint<512> hbm_t;
typedef ap_uint<512> dram_t;
typedef ap_uint<96> addr_length_t;
typedef ap_uint<10> order_t;
typedef ap_uint<6> channel_t;
typedef ap_uint<518> channel_ocxflit_t;
typedef ap_uint<528> tag_ocxflit_t;

#define LOG2_MAX_IN_FLIGHT 7
#define MAX_IN_FLIGHT (1 << LOG2_MAX_IN_FLIGHT)
typedef ap_uint<LOG2_MAX_IN_FLIGHT+2> buffer_addr_t;

typedef struct {
	hbm_t* p1;
	hbm_t* p2;
	addr_t strided_hbm_offset;
} hbm_gmem_t;

#endif
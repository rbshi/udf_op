#include "krnl_udf_ml.h"

inline ocxflit_t ml_read_hbm_line(
	hbm_t* p_hbm,
	addr_t hbm_addr)
{
	ocxflit_t temp = 0;
	temp(BITS_IN_LINE-1,0) = p_hbm[hbm_addr];
	return temp;
}

inline void ml_write_hbm_line(
	hbm_t* p_hbm,
	addr_t hbm_addr,
	ocxflit_t line)
{
	p_hbm[hbm_addr] = line(BITS_IN_LINE-1,0);
}

void ml_read_hbm_to_stream (
	hls::stream<ocxflit_t>& out_stream,
	hbm_t* p_hbm,
	addr_t hbm_addr,
	unsigned size_in_lines)
{
	ocxflit_t temp = 0;
	unsigned k = 0;
	while (k < size_in_lines) {
#pragma HLS PIPELINE II=1
		temp(BITS_IN_LINE-1,0) = p_hbm[hbm_addr + k];
		out_stream.write(temp);
		k++;
	}
}

void ml_write_stream_to_hbm (
	hls::stream<ocxflit_t>& in_stream,
	hbm_t* p_hbm,
	addr_t hbm_addr,
	unsigned size_in_lines)
{
	ocxflit_t temp = 0;
	unsigned k = 0;
	while (k < size_in_lines) {
#pragma HLS PIPELINE II=1
		temp = in_stream.read();
		p_hbm[hbm_addr + k] = temp(BITS_IN_LINE-1, 0);
		k++;
	}
}

void ml_duplicate_stream(
	hls::stream<ap_uint<BITS_IN_LINE> >& out_stream1,
	hls::stream<ap_uint<BITS_IN_LINE> >& out_stream2,
	hls::stream<ap_uint<BITS_IN_LINE> >& in_stream,
	unsigned num_lines)
{
	for (unsigned i = 0; i < num_lines; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<BITS_IN_LINE> temp = in_stream.read();
		out_stream1.write(temp);
		out_stream2.write(temp);
	}
}
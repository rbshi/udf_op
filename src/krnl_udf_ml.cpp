#include "krnl_udf_ml.h"
#include "krnl_udf_ml_misc.hpp"

#include "hls_math.h"

#define DATAFLOW_BASED

inline float linepart2float(ap_uint<BITS_IN_LINE> line, uint32_t k) {
    uint32_t temp = line( (k+1)*BITS_IN_FLOAT-1, k*BITS_IN_FLOAT );
    float* temp_float = (float*)&temp;
    return (*temp_float);
}

inline void float2linepart(ap_uint<BITS_IN_LINE>& line, uint32_t k, float value) {
    uint32_t* temp = (uint32_t*)&value;
    line( (k+1)*BITS_IN_FLOAT-1, k*BITS_IN_FLOAT ) = (*temp);
}

void read_model(
    hbm_t* p_hbm,
    addr_t model_addr,
    ap_int<MODEL_BITS> model[PARALLELISM][MAX_DIMENSIONALITY],
    unsigned num_features_in_lines)
{
#pragma HLS DATAFLOW
    hls::stream<ap_uint<BITS_IN_LINE> > in_stream("in_stream");
    ml_read_hbm_to_stream(in_stream, p_hbm, model_addr, num_features_in_lines);

    for (unsigned i = 0; i < num_features_in_lines; i++) {
#pragma HLS PIPELINE II=16
        ap_uint<BITS_IN_LINE> temp = in_stream.read();
        for (unsigned p = 0; p < PARALLELISM; p++) {
// FIXME: unroll consumes tooo many resource on linepart2float
// #pragma HLS UNROLL
            float temp_model = linepart2float(temp, p);

            model[p][i] = (ap_int<MODEL_BITS>)(temp_model*FIXED_SCALE);
#pragma HLS RESOURCE variable=model[p][i] core=FMul_maxdsp            
        }
    }
}

void write_model(
    hbm_t* p_hbm,
    addr_t model_addr,
    ap_int<MODEL_BITS> model[PARALLELISM][MAX_DIMENSIONALITY],
    unsigned num_features_in_lines)
{
#pragma HLS DATAFLOW

    hls::stream<ap_uint<BITS_IN_LINE> > out_stream("out_stream");
    ap_uint<BITS_IN_LINE> temp = 0;

    for (unsigned i = 0; i < num_features_in_lines; i++) {
#pragma HLS PIPELINE
        for (unsigned p = 0; p < PARALLELISM; p++) {
#pragma HLS UNROLL
            float temp_float = (float)model[p][i]/FIXED_SCALE;
            float2linepart(temp, p, temp_float);
        }
        out_stream.write(temp);
    }

    ml_write_stream_to_hbm(out_stream, p_hbm, model_addr, num_features_in_lines);
}

void read_labels(
    hbm_t* p_hbm,
    addr_t label_addr,
    float labels[MAX_PARTITION_SIZE],
    unsigned partition_size_in_lines)
{
#pragma HLS DATAFLOW
    hls::stream<ap_uint<BITS_IN_LINE> > in_stream("in_stream");
    ml_read_hbm_to_stream(in_stream, p_hbm, label_addr, partition_size_in_lines);

    for (unsigned i = 0; i < partition_size_in_lines; i++) {
#pragma HLS PIPELINE
        ap_uint<BITS_IN_LINE> temp = in_stream.read();

        unsigned offset = i*FLOATS_IN_LINE;
        for (unsigned p = 0; p < FLOATS_IN_LINE; p++) {
            labels[offset + p] = linepart2float(temp, p);
#ifdef DEBUG
            cout << "labels[" << offset+p << "]: " << labels[offset+p] << endl;
#endif
        }
    }
}

void dot(
    hls::stream<ap_uint<BITS_IN_LINE> >& in_stream,
    hls::stream<ap_uint<BITS_IN_LINE> >& model_stream,
    hls::stream<ap_int<MODEL_BITS> >& dot_stream,
    unsigned num_features_in_lines,
    unsigned minibatch_size)
{
    ap_int<MODEL_BITS> dot_partial_result_fixed[PARALLELISM];
    unsigned dimension = 0;
    ap_int<MODEL_BITS> result = 0;
    ap_int<MODEL_BITS> dot_result = 0;

    for (unsigned i = 0; i < minibatch_size*num_features_in_lines; i++) {
#pragma HLS PIPELINE

        ap_uint<BITS_IN_LINE> temp_in = in_stream.read();
        ap_uint<BITS_IN_LINE> temp_model = model_stream.read();

        for (unsigned p = 0; p < PARALLELISM; p++) {
#pragma HLS UNROLL
            float in = linepart2float(temp_in, p);
            float model_float = linepart2float(temp_model, p);
#ifdef DEBUG
            cout << "in: " << in << endl;
            cout << "model_float: " << model_float << endl;
#endif

            float temp = in*model_float;
#pragma HLS RESOURCE variable=temp core=FMul_maxdsp            
            ap_int<MODEL_BITS> temp_fixed = (ap_int<MODEL_BITS>)temp;
            dot_partial_result_fixed[p] = temp_fixed;
        }

        result = 0;
        for (unsigned p = 0; p < PARALLELISM; p++) {
#pragma HLS UNROLL
            result += dot_partial_result_fixed[p];
        }
        dot_result = (dimension == 0) ? result : (ap_int<MODEL_BITS>)(dot_result + result);

        if (dimension == num_features_in_lines-1) {
            dot_stream.write(dot_result);
            dimension = 0;
        }
        else {
            dimension++;
        }
    }
}

void scalar_engine(
    hls::stream<ap_int<MODEL_BITS> >& dot_stream,
    hls::stream<float >& gradient_stream,
    float labels[MAX_PARTITION_SIZE],
    float step_size,
    bool do_logreg,
    unsigned minibatch_size)
{
    float gradient = 0;
    for (unsigned i = 0; i < minibatch_size; i++) {
#pragma HLS PIPELINE
        ap_int<MODEL_BITS> dot = dot_stream.read();
        float dot_float = (float)dot;
        if (do_logreg) {
            dot_float = dot_float/FIXED_SCALE;
            dot_float = 1.0/(1.0 + exp(-dot_float));
#pragma HLS RESOURCE variable=dot_float core=FMul_maxdsp
            dot_float = dot_float*FIXED_SCALE;
        }

        float label_scaled = labels[i]*FIXED_SCALE;
#pragma HLS RESOURCE variable=label_scaled core=FMul_maxdsp        
        gradient = dot_float - label_scaled;
#pragma HLS RESOURCE variable=gradient core=FMul_maxdsp
        gradient = step_size*gradient;
#ifdef DEBUG
        cout << "--------------------------------------------" << endl;
        cout << "dot: " << dot << endl;
        cout << "dot_float: " << dot_float/FIXED_SCALE << endl;
        cout << "label: " << labels[i] << endl;
        cout << "label_scaled: " << label_scaled << endl;
        cout << "gradient: " << gradient << endl;
#endif
        gradient_stream.write(gradient);
    }
}

void update(
    hls::stream<ap_uint<BITS_IN_LINE> >& in_stream,
    hls::stream<float >& gradient_stream,
    ap_int<MODEL_BITS> model[PARALLELISM][MAX_DIMENSIONALITY],
    hls::stream<ap_uint<BITS_IN_LINE> >& model_stream,
    unsigned num_features_in_lines,
    unsigned minibatch_size,
    float lambda,
    bool first_minibatch, bool last_minibatch)
{
#pragma HLS dependence variable=model false
    uint32_t dimension = 0;
    float gradient = 0;

    for (unsigned i = 0; i < minibatch_size*num_features_in_lines; i++) {
#pragma HLS PIPELINE

        if (!first_minibatch && dimension == 0) {
            gradient = gradient_stream.read();
        }

        ap_uint<BITS_IN_LINE> temp = 0;
        if (!first_minibatch) {
            temp = in_stream.read();
        }

        ap_uint<BITS_IN_LINE> model_output = 0;
        for (unsigned p = 0; p < PARALLELISM; p++) {
#pragma HLS UNROLL
            float in = linepart2float(temp, p);

            float step = gradient*in;
#pragma HLS RESOURCE variable=step core=FMul_maxdsp            
            ap_int<MODEL_BITS> step_fixed = (ap_int<MODEL_BITS>)step;
            ap_int<MODEL_BITS> model_fixed = model[p][dimension];
#ifdef DEBUG
            cout << "in " << p << ": " << in << endl;
            cout << "step " << p << ": " << step << endl;
            cout << "model_fixed " << p << ": " << model_fixed << endl;
#endif
            // float regularization = lambda*model_float;
            // ap_int<MODEL_BITS> regularization_fixed = (ap_int<MODEL_BITS>)regularization;
            // ap_int<MODEL_BITS> temp = model_fixed - step_fixed - regularization_fixed;
            ap_int<MODEL_BITS> temp = model_fixed - step_fixed;
            model[p][dimension] = temp;
            float model_float = (float)temp;
            float2linepart(model_output, p, model_float);
        }
        if (!last_minibatch) {
#ifdef DEBUG
            cout << "model_stream.write " << i << hex << ": " << model_output << dec << endl;
#endif
            model_stream.write(model_output);
        }
        
        if (dimension == num_features_in_lines-1) {
            dimension = 0;
        }
        else {
            dimension++;
        }
    }
}

void sgd_pipeline(
    hls::stream<ap_uint<BITS_IN_LINE> >& in_stream,
    hls::stream<ap_uint<BITS_IN_LINE> >& in_stream_for_dot,
    hls::stream<ap_uint<BITS_IN_LINE> >& in_stream_for_update,
    hls::stream<float >& gradient_stream,
    hls::stream<ap_uint<BITS_IN_LINE> >& model_stream,
    hls::stream<ap_int<MODEL_BITS> >& dot_stream,
    ap_int<MODEL_BITS> model[PARALLELISM][MAX_DIMENSIONALITY],
    float labels[MAX_PARTITION_SIZE],
    float step_size,
    float lambda,
    unsigned num_features_in_lines,
    unsigned minibatch_size,
    bool do_logreg,
    bool first_minibatch, bool last_minibatch)
{
#pragma HLS DATAFLOW

    update(in_stream_for_update, gradient_stream, model, model_stream, num_features_in_lines, minibatch_size, lambda, first_minibatch, last_minibatch);

    dot(in_stream_for_dot, model_stream, dot_stream, num_features_in_lines, last_minibatch ? 0 : minibatch_size);

    scalar_engine(dot_stream, gradient_stream, labels, step_size, do_logreg, last_minibatch ? 0 : minibatch_size);
}

void sgd_pipeline_top(
    hls::stream<ap_uint<BITS_IN_LINE> >& in_stream,
    hls::stream<ap_uint<BITS_IN_LINE> >& in_stream_for_dot,
    hls::stream<ap_uint<BITS_IN_LINE> >& in_stream_for_update,
    hls::stream<float >& gradient_stream,
    hls::stream<ap_uint<BITS_IN_LINE> >& model_stream,
    hls::stream<ap_int<MODEL_BITS> >& dot_stream,
    hbm_t* p_hbm,
    addr_t samples_addr,
    ap_int<MODEL_BITS> model[PARALLELISM][MAX_DIMENSIONALITY],
    float labels[MAX_PARTITION_SIZE],
    float step_size,
    float lambda,
    unsigned num_features_in_lines,
    unsigned num_minibatches_in_partition,
    unsigned minibatch_size,
    bool do_logreg)
{
#pragma HLS DATAFLOW

    ml_read_hbm_to_stream(in_stream, p_hbm, samples_addr, num_minibatches_in_partition*minibatch_size*num_features_in_lines);

    ml_duplicate_stream(in_stream_for_dot, in_stream_for_update, in_stream, num_minibatches_in_partition*minibatch_size*num_features_in_lines);

    unsigned processed_samples_in_partition = 0;
    for (unsigned k = 0; k < num_minibatches_in_partition + 1; k++) {
        sgd_pipeline(
            in_stream, in_stream_for_dot, in_stream_for_update,
            gradient_stream, model_stream, dot_stream,
            model,
            labels + processed_samples_in_partition,
            step_size,
            lambda,
            num_features_in_lines,
            minibatch_size,
            do_logreg,
            k == 0, k == num_minibatches_in_partition);

        processed_samples_in_partition += minibatch_size;
    }
}

void sgd_update(
    hls::stream<ap_uint<BITS_IN_LINE> >& in_stream_for_dot,
    hls::stream<ap_uint<BITS_IN_LINE> >& in_stream_for_update,
    ap_int<MODEL_BITS> model[PARALLELISM][MAX_DIMENSIONALITY],
    float labels[MAX_PARTITION_SIZE],
    float step_size,
    float lambda,
    unsigned num_features_in_lines,
    unsigned partition_size,
    unsigned minibatch_size,
    bool do_logreg)
{
#pragma HLS dependence variable=model false

    ap_uint<MAX_DIMENSIONALITY> model_ready = -1;
    ap_int<MODEL_BITS> model_stream[PARALLELISM][MAX_DIMENSIONALITY];
#pragma HLS RESOURCE variable=model_stream core=XPM_MEMORY uram
#pragma HLS ARRAY_PARTITION variable=model_stream dim=0 complete
#pragma HLS dependence variable=model_stream false

    uint32_t dot_dimension = 0;
    uint32_t dot_sample_index = 0;

    uint32_t update_dimension = 0;
    uint32_t update_sample_index = 0;

    ap_int<MODEL_BITS> dot_partial_result_fixed[PARALLELISM];
    ap_int<MODEL_BITS> result = 0;
    ap_int<MODEL_BITS> dot_result = 0;

    ap_int<MODEL_BITS> dot = 0;
    float gradient = 0;

    while (true) {
#pragma HLS PIPELINE

        if (dot_sample_index < partition_size && model_ready(dot_dimension, dot_dimension) == 1) {
// DOT
            ap_uint<BITS_IN_LINE> in_for_dot = in_stream_for_dot.read();

            for (unsigned p = 0; p < PARALLELISM; p++) {
#pragma HLS UNROLL
                float in = linepart2float(in_for_dot, p);
                float model_float = (float)model_stream[p][dot_dimension];

                float temp = in*model_float;
#pragma HLS RESOURCE variable=temp core=FMul_maxdsp                
                ap_int<MODEL_BITS> temp_fixed = (ap_int<MODEL_BITS>)temp;
                dot_partial_result_fixed[p] = temp_fixed;
            }

            result = 0;
            for (unsigned p = 0; p < PARALLELISM; p++) {
#pragma HLS UNROLL
                result += dot_partial_result_fixed[p];
            }

            model_ready(dot_dimension, dot_dimension) = 0;

// SCALAR ENGINE
            if (dot_dimension == num_features_in_lines-1) {
                float dot_float = (float)(dot_result + result);
                dot_result = 0;
                if (do_logreg) {
                    dot_float = dot_float/FIXED_SCALE;
                    dot_float = 1.0/(1.0 + exp(-dot_float));
#pragma HLS RESOURCE variable=dot_float core=FMul_maxdsp
                    dot_float = dot_float*FIXED_SCALE;
                }
        
                float label_scaled = labels[dot_sample_index]*FIXED_SCALE;
#pragma HLS RESOURCE variable=label_scaled core=FMul_maxdsp                        
                gradient = dot_float - label_scaled;
#pragma HLS RESOURCE variable=gradient core=FMul_maxdsp
                gradient = step_size*gradient;
                dot_dimension = 0;
                dot_sample_index++;
            }
            else {
                dot_dimension++;
                dot_result += result;
            }
        }

// UPDATE
        if (dot_sample_index > 0 && update_sample_index < partition_size && model_ready(update_dimension,update_dimension) == 0) {
            ap_uint<BITS_IN_LINE> in_for_update = in_stream_for_update.read();

            for (unsigned p = 0; p < PARALLELISM; p++) {
#pragma HLS UNROLL
                float in = linepart2float(in_for_update, p);

                float step = gradient*in;
#pragma HLS RESOURCE variable=step core=FMul_maxdsp                
                ap_int<MODEL_BITS> step_fixed = (ap_int<MODEL_BITS>)step;
                ap_int<MODEL_BITS> model_fixed = model[p][update_dimension];
                ap_int<MODEL_BITS> temp = model_fixed - step_fixed;
                model[p][update_dimension] = temp;
                model_stream[p][update_dimension] = temp;
            }

            model_ready(update_dimension,update_dimension) = 1;

            if (update_dimension == num_features_in_lines-1) {
                update_dimension = 0;
                if (update_sample_index == partition_size-1) {
                    break;
                }
                else {
                    update_sample_index++;
                }
            }
            else {
                update_dimension++;
            }
        }
    }
}

void sgd_pipeline2(
    hls::stream<ap_uint<BITS_IN_LINE> >& in_stream,
    hls::stream<ap_uint<BITS_IN_LINE> >& in_stream_for_dot,
    hls::stream<ap_uint<BITS_IN_LINE> >& in_stream_for_update,
    hbm_t* p_hbm,
    addr_t samples_addr,
    ap_int<MODEL_BITS> model[PARALLELISM][MAX_DIMENSIONALITY],
    float labels[MAX_PARTITION_SIZE],
    float step_size,
    float lambda,
    unsigned num_features_in_lines,
    unsigned partition_size,
    unsigned minibatch_size,
    bool do_logreg)
{
#pragma HLS DATAFLOW

    ml_read_hbm_to_stream(in_stream, p_hbm, samples_addr, num_features_in_lines*partition_size);

    ml_duplicate_stream(in_stream_for_dot, in_stream_for_update, in_stream, num_features_in_lines*partition_size);

    sgd_update(
        in_stream_for_dot, in_stream_for_update,
        model,
        labels,
        step_size,
        lambda,
        num_features_in_lines,
        partition_size,
        minibatch_size,
        do_logreg);
}

inline unsigned CEILING(unsigned value, unsigned log2_divider) {
    return (value >> log2_divider) + (((value & ((1 << log2_divider)-1)) > 0) ? 1 : 0 );
}

inline unsigned FLOOR(unsigned value, unsigned log2_divider) {
    return (value >> log2_divider);
}

void krnl_udf_ml(
    hbm_t* p_hbm,
    addr_t samples_addr,
    addr_t labels_addr,
    addr_t model_addr,
    float step_size,
    float lambda,
    bool do_logreg,
    unsigned minibatch_size,
    unsigned num_epochs,
    unsigned num_features_in_lines,
    unsigned num_samples)
{

#pragma HLS INTERFACE m_axi port = p_hbm offset = slave bundle = gmem0

#pragma HLS INTERFACE s_axilite port = p_hbm
#pragma HLS INTERFACE s_axilite port = samples_addr
#pragma HLS INTERFACE s_axilite port = labels_addr
#pragma HLS INTERFACE s_axilite port = model_addr
#pragma HLS INTERFACE s_axilite port = step_size
#pragma HLS INTERFACE s_axilite port = lambda
#pragma HLS INTERFACE s_axilite port = do_logreg
#pragma HLS INTERFACE s_axilite port = minibatch_size
#pragma HLS INTERFACE s_axilite port = num_epochs
#pragma HLS INTERFACE s_axilite port = num_features_in_lines
#pragma HLS INTERFACE s_axilite port = num_samples

#pragma HLS INTERFACE s_axilite port = return

    static ap_int<MODEL_BITS> model[PARALLELISM][MAX_DIMENSIONALITY];
    static float labels[MAX_PARTITION_SIZE];
#pragma HLS RESOURCE variable=model core=XPM_MEMORY uram
#pragma HLS RESOURCE variable=labels core=XPM_MEMORY uram
#pragma HLS ARRAY_PARTITION variable=model dim=1 complete

    static hls::stream<ap_uint<BITS_IN_LINE> > in_stream("in_stream");
    static hls::stream<ap_uint<BITS_IN_LINE> > in_stream_for_dot("in_stream_for_dot");
    static hls::stream<ap_uint<BITS_IN_LINE> > in_stream_for_update("in_stream_for_update");
// MAX_MINIBATCH_SIZE=16
#pragma HLS STREAM variable=in_stream depth=256
#pragma HLS STREAM variable=in_stream_for_dot depth=16
// MAX_MINIBATCH_SIZE(=16)*MAX_DIMENSIONALITY(=2048)/FLOATS_IN_LINE(=16)
#pragma HLS STREAM variable=in_stream_for_update depth=16*128
#ifdef DATAFLOW_BASED
    static hls::stream<float > gradient_stream("gradient_stream");
    static hls::stream<ap_uint<BITS_IN_LINE> > model_stream("model_stream");
    static hls::stream<ap_int<MODEL_BITS> > dot_stream("dot_stream");
#pragma HLS STREAM variable=gradient_stream depth=16
#pragma HLS STREAM variable=model_stream depth=16
#pragma HLS STREAM variable=dot_stream depth=16
#endif

    read_model(
        p_hbm,
        model_addr,
        model,
        num_features_in_lines);

    unsigned num_partitions = CEILING(num_samples, LOG2_MAX_PARTITION_SIZE);

    for (unsigned e = 0; e < num_epochs; e++) {

        unsigned processed_samples = 0;
        unsigned labels_read_offset = 0;

        for (unsigned i = 0; i < num_partitions; i++) {
            unsigned partition_size = MAX_PARTITION_SIZE;
            if (i == num_partitions-1) {
                partition_size = num_samples - processed_samples;
            }

            unsigned partition_size_in_lines = CEILING(partition_size, LOG2_FLOATS_IN_LINE);

            read_labels(
                p_hbm,
                labels_addr + labels_read_offset,
                labels,
                partition_size_in_lines);

#ifdef DATAFLOW_BASED
            unsigned num_minibatches_in_partition = partition_size/minibatch_size;
            sgd_pipeline_top (
                in_stream, in_stream_for_dot, in_stream_for_update, gradient_stream, model_stream, dot_stream,
                p_hbm,
                samples_addr + processed_samples*num_features_in_lines,
                model,
                labels,
                step_size,
                lambda,
                num_features_in_lines,
                num_minibatches_in_partition,
                minibatch_size,
                do_logreg);
#else
            sgd_pipeline2(
                in_stream, in_stream_for_dot, in_stream_for_update,
                p_hbm,
                samples_addr + processed_samples*num_features_in_lines,
                model,
                labels,
                step_size,
                lambda,
                num_features_in_lines,
                partition_size,
                minibatch_size,
                do_logreg);
#endif
            processed_samples += partition_size;
            labels_read_offset += partition_size_in_lines;
        }

        write_model(
            p_hbm,
            model_addr + e*num_features_in_lines,
            model,
            num_features_in_lines);
    }
}
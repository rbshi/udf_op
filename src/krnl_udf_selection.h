// #pragma once

#include <hls_stream.h>
#include <ap_int.h>
#include "types.h"

const unsigned LOG2_BUFFER_SIZE = 10;
const unsigned BUFFER_SIZE = (1 << LOG2_BUFFER_SIZE);
const unsigned PARALLELISM = 16;

const unsigned LOG2_BYTES_IN_LINE =              6;
const unsigned LOG2_BYTES_IN_HBM_LINE =          6;
const unsigned LOG2_BYTES_IN_INT =               2;

const unsigned LOG2_BITS_IN_LINE =   (LOG2_BYTES_IN_LINE + 3);
const unsigned LOG2_BITS_IN_INT =    (LOG2_BYTES_IN_INT + 3);
const unsigned BITS_IN_LINE =        (1 << LOG2_BYTES_IN_LINE)*8;
const unsigned BITS_IN_HBM_LINE =    (1 << LOG2_BYTES_IN_HBM_LINE)*8;
const unsigned BITS_IN_INT =         (1 << LOG2_BYTES_IN_INT)*8;
const unsigned BYTES_IN_LINE =       (1 << LOG2_BYTES_IN_LINE);
const unsigned BYTES_IN_HBM_LINE =   (1 << LOG2_BYTES_IN_HBM_LINE);
const unsigned BYTES_IN_INT =        (1 << LOG2_BYTES_IN_INT);

const unsigned LOG2_INTS_IN_HBM_LINE =   (LOG2_BYTES_IN_HBM_LINE - LOG2_BYTES_IN_INT);
const unsigned INTS_IN_HBM_LINE =        (1 << LOG2_INTS_IN_HBM_LINE);
const unsigned LOG2_INTS_IN_LINE =       (LOG2_BYTES_IN_LINE - LOG2_BYTES_IN_INT);
const unsigned INTS_IN_LINE =            (1 << LOG2_INTS_IN_LINE);
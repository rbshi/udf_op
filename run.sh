#! /bin/bash

if [ $# != 2 ] ; then
        echo "Usage: ./run.sh <kernel_name> <.cfg file index>";
        exit
fi

CURRENT_DIR=$(pwd)
BUILD_DIR="build_$1_$2"
CONFIG_FILE="$CURRENT_DIR/udf_op/cfg/krnl_$1_$2.cfg"

if [ ! -f $CONFIG_FILE ]; then
    echo "config file DOES NOT exist!";
    exit
fi

mkdir $BUILD_DIR && cd $BUILD_DIR && cmake ../udf_op -DHOST_NAME=$1 -DKERNEL_CFG_FILE=krnl_$1_$2.cfg && make hw && cd ..

echo "[Done] $1_$2"
#!/bin/bash

mkdir /tmp/sorthoData1/ || echo "already exists"

RA_RAW_DATA='/tmp/sorthoData1/flight.ra'
RA_OPT_DATA='/tmp/sorthoData1/flight.opt.ra'
RAW_TIFF='/tmp/sorthoData1/raw.tif'
OPT_TIFF='/tmp/sorthoData1/opt.tif'

frameStride=8
maxFrames=10
maxFrames=3

if [ ! -f ${RA_RAW_DATA} ]; then
    python3 -m sortho.loading.converters.terrapixel path='/data/inertialLabs/flightFeb15/irnOutput/1707947224/eval.bin' output=${RA_RAW_DATA} frameStride=${frameStride} maxFrames=${maxFrames}
fi

python3 -m sortho.solve.solve input.path=${RA_RAW_DATA} outputPath=${RA_OPT_DATA}


if [ ! -f ${RAW_TIFF} ]; then
    python3 -m sortho.blending.blend_main1 input.frameStride=1 input.maxFrames=999 input.path=${RA_RAW_DATA} outputPath='/tmp/sorthoData1/raw.tif'
fi
python3 -m sortho.blending.blend_main1 input.frameStride=1 input.maxFrames=999 input.path=${RA_OPT_DATA} outputPath='/tmp/sorthoData1/opt.tif'

python3 -m sortho.utils.tiff_viewer ${RAW_TIFF} &
python3 -m sortho.utils.tiff_viewer ${OPT_TIFF}

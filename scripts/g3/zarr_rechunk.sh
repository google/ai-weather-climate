#!/bin/bash

source gbash.sh || exit 1

DEFINE_string cells "qo:qr" "colon-separated list of cells to run the pipeline"
DEFINE_string input_zarr \
  "/namespace/aiforweather/primary/jyh/heatnet/raw.zarr" \
  "Input Zarr volume"
DEFINE_string output_zarr \
  "/namespace/aiforweather/primary/jyh/heatnet/r=3.2/raw.zarr" \
  "Output Zarr volume"

gbash::init_google "$@"

FLAGS=(
  --input_zarr="${FLAGS_input_zarr}"
  --output_zarr="${FLAGS_output_zarr}"
  --flume_borg_cells="${FLAGS_cells}"
  --flume_borg_accounting_charged_user_name=aiforweather
  --flume_use_batch_scheduler
  --runner=google3.pipeline.flume.py.runner.FlumeRunner
  )

set -x

blaze run -c opt //third_party/py/ai_weather_climate/beam:zarr_rechunk.par -- "${FLAGS[@]}" --alsologtostderr

#!/bin/bash

source gbash.sh || exit 1

DEFINE_string cells "qo:qr:qn:qm:yo:yq:yr" "colon-separated list of cells to run the pipeline"
DEFINE_string input_zarr \
  "/namespace/aiforweather/primary/jyh/heatnet/r=3.2/era5_cs.zarr" \
  "Input Zarr volume"
DEFINE_string stats_zarr \
  "/namespace/aiforweather/primary/jyh/heatnet/r=3.2/stats_cs.zarr" \
  "Zarr volume containing climatology"
DEFINE_string static_zarr \
  "/namespace/aiforweather/primary/jyh/heatnet/r=3.2/static_cs.zarr" \
  "Zarr volume containing static inputs"
DEFINE_string output_rio \
  "/namespace/aiforweather/primary/jyh/heatnet/r=3.2/examples" \
  "Output recordio file"
DEFINE_string train_date_range \
  "1970-01-01:2000-01-01" \
  "Date range for training dataset (range is closed, open)"
DEFINE_string validate_date_range \
  "2000-01-01:2010-01-01" \
  "Date range for validation dataset (range is closed, open)"
DEFINE_string test_date_range \
  "2010-01-01:" \
  "Date range for test dataset (range is closed, open)"
DEFINE_string runner \
  "google3.pipeline.flume.py.runner.FlumeRunner" \
  "Beam runner"

gbash::init_google "$@"

FLAGS=(
  --input_zarr="${FLAGS_input_zarr}"
  --stats_zarr="${FLAGS_stats_zarr}"
  --static_zarr="${FLAGS_static_zarr}"
  --output_rio="${FLAGS_output_rio}"
  --train_date_range="${FLAGS_train_date_range}"
  --test_date_range="${FLAGS_test_date_range}"
  --validate_date_range="${FLAGS_validate_date_range}"
  --runner="${FLAGS_runner}"
  --flume_borg_cells="${FLAGS_cells}"
  --flume_borg_accounting_charged_user_name=aiforweather
  --flume_worker_remote_hdd_scratch=1G
  --flume_use_batch_scheduler
  --flume_exec_mode=BORG
  )

set -x

blaze run -c opt //third_party/py/heatnet:create_examples.par -- "${FLAGS[@]}" --alsologtostderr "${GBASH_ARGV[@]}"

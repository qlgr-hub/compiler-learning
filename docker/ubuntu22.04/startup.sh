#!/usr/bin/env bash

set -e

cd /workspace || exit
/home/george/miniconda3/bin/conda run -n tvm --no-capture-output jupyter lab --allow-root --no-browser --ip=0.0.0.0 &

sudo /usr/sbin/sshd -D

#!/bin/bash

set -e

pred=$1
gold=$2

cp $pred tools/amr-evaluation-tool-enhanced/test.pred.txt
cp $gold tools/amr-evaluation-tool-enhanced/test.txt
cd tools/amr-evaluation-tool-enhanced && ./evaluation.sh test.pred.txt test.txt

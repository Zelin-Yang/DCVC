#!/bin/bash
# 用法: ./record_experiment.sh "实验描述"
TAG="$1"
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=../manifests

echo "=== $TS $TAG ===" >> $MANIFEST/experiments.log
echo "commit: $(git rev-parse HEAD)" >> $MANIFEST/experiments.log
echo "branch: $(git branch --show-current)" >> $MANIFEST/experiments.log
git diff --stat >> $MANIFEST/experiments.log
echo "" >> $MANIFEST/experiments.log

# 同时单独存一份该实验的完整 diff，方便回溯
git diff > $MANIFEST/diff_${TS}.patch
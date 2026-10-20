#!/bin/zsh
# prep.sh <ds...>: AIST features then MiniLM-L6 teacher for each built dataset (one MPS process at a time)
export HF_HOME=/Volumes/Build/cortext/headroom/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/Volumes/Build/cortext/reranker/venv/bin/python
D=/Volumes/Build/cortext/reranker/data
cd ${0:A:h}
for ds in "$@"; do
  nice -n 5 $PY -I featurize.py --data $D/$ds
  nice -n 5 $PY -I teacher.py --data $D/$ds --model cross-encoder/ms-marco-MiniLM-L6-v2 --key minilm
done
echo "prep done $(date +%T)"

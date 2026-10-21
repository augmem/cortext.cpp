#!/bin/sh
# Retrain and re-evaluate with OpenAssistant oasst1 real-chat rows (README, "oasst1 round").
# Precondition: oasst1 message files or parquet shards are under $O/data/oasst1 (data only; nothing there is executed).
# usage: sh tools/directive_head/run_oasst.sh
set -eu
H="$(cd "$(dirname "$0")" && pwd)"
O=/Volumes/Build/cortext/directive-head
P=/Volumes/Build/cortext/reranker/venv/bin/python
S=$O/streams
test -d "$O/data/oasst1" || { echo "missing $O/data/oasst1"; exit 1; }
cd "$H"
# 1. label + split by conversation, merge with the synthetic rows
$P -I oasst_build.py --src "$O/data/oasst1" --out "$O/data_oasst" --merge-with "$O/data/rows.jsonl" --merged "$O/data_mix"
# 2. features for the merged rows (engine tokenizer, token states)
$P -I featurize.py --in "$O/data_mix/rows.jsonl" --out "$O/data_mix/feat_engine" --tokens --device mps
# 3. train: mixed (synthetic + oasst1) attention head and pooled heads
$P -I train.py --data "$O/data_mix" --feat feat_engine --out "$O/runs5/attn_h8" --arms attn --tokens --attn-heads 8
$P -I train.py --data "$O/data_mix" --feat feat_engine --out "$O/runs5/pooled" --arms mlp256,mlp1280
# 4. stream scores for the mixed head
$P -I featurize.py --in "$S/texts.jsonl" --out "$S/feat_attn5" --max-bytes 480 --device mps --head "$O/runs5/attn_h8/attn.pt"
# 5. evaluate both ways on the merged test split (synthetic templates/topics + oasst1 conversations)
mkdir -p "$O/runs5/streams" "$O/runs5/streams_syn"
$P -I evaluate.py --data "$O/data_mix" --runs "$O/runs5/pooled" \
  --arms "mlp256,mlp1280,attn=$O/runs5/attn_h8/attn.pt,attn_syn=$O/runs4/attn_h8_pw1.0/attn.pt" \
  --arm attn --streams "$S" --stream-feat feat_attn5 --modality "$O/modality" --modality-arm mlp256 \
  --out "$O/runs5/streams/eval.json"
$P -I evaluate.py --data "$O/data_mix" --runs "$O/runs5/pooled" --arms "attn_syn=$O/runs4/attn_h8_pw1.0/attn.pt" \
  --arm attn_syn --streams "$S" --stream-feat feat_attn3 --out "$O/runs5/streams_syn/eval.json"
# 6. judge disagreements and the high band for the mixed head
$P -I judge.py --dis "$O/runs5/streams/disagreements.jsonl" --out "$O/runs5/streams/judged.jsonl" --n 50
$P -I judge.py --bands "$O/runs5/streams" --out "$O/runs5/streams/bands.jsonl" --n 25 --band-lo 0 --band-hi 6
# 7. export
$P -I export_gguf.py --ckpt "$O/runs5/attn_h8/attn.pt" --out "$O/AIST-87M-DH.gguf" --fixture "$O/dh_parity.json" --data "$O/data_mix"

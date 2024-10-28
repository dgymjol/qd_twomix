ckpt_path=$1
eval_path=data/tacos/val.jsonl
PYTHONPATH=$PYTHONPATH:. python la_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name val \
--eval_path ${eval_path} \
${@:3}
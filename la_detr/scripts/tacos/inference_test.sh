ckpt_path=$1
eval_split_name=$2
eval_path=data/tacos/${eval_split_name}.jsonl
PYTHONPATH=$PYTHONPATH:. python la_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:3}
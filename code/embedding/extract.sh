model=$1
config=$2
win_length=$3
dev_out=$4
train_out=$5
type=$6
embed_out=$7

dev_label=/mnt/lustre/sjtu/home/pyz99/depression/henri_model/labels/dev_split_Depression_AVEC2017.csv
train_label=/mnt/lustre/sjtu/home/pyz99/depression/henri_model/labels/train_split_Depression_AVEC2017.csv
path=/mnt/lustre/sjtu/users/hedi7/depression/woz/data

python3 extract_embedding.py \
-m $model -p $path -w $win_length -o $embed_out -t $type --config $config

python3 split_traindev.py \
-d $dev_label -t $train_label -i $embed_out --dev_out $dev_out --train_out $train_out

rm -rf $embed_out






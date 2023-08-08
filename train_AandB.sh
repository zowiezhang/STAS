
EXP_NAME="$1_AandB"

cp ./configs/$EXP_NAME.py arguments.py

conda activate stas

nohup python -u train_AandB.py --exp_name $EXP_NAME \
                        --method_name $1 \
                        --num_episodes 2000 \
                        --max_step_per_round 40 \
                        --cuda \
                        --seed 123 \
>> stdlogs/$EXP_NAME.out &
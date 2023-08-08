
EXP_NAME="$1_MPE"

cp ./configs/$EXP_NAME.py arguments.py
cd envs/multiagent-particle-envs/
pip install -e ./envs/multiagent-particle-envs/
cd ../..

conda activate stas

nohup python -u train_MPE.py --exp_name $EXP_NAME \
                        --scenario simple_tag_n3 \
                        --method_name $1 \
                        --num_episodes 1000 \
                        --max_step_per_round 25 \
                        --cuda
>> stdlogs/$EXP_NAME.out &

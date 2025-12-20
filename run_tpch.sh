#!/bin/bash
is_dws=("dw") #@ "nodw")
attn_types=("CAI") # "WOCAI") 
tpch_sizes=("50") # "100" "150") 
dag_numbers=("16")
n_stepss=("4")
GNN_models=("GAT") # "Graphormer" "RF")
#GNN_models=("RF") 

gpu=0

count=0
max_jobs=6
for f in "${!is_dws[@]}"; do 
    for g in "${!attn_types[@]}"; do
        for h in "${!tpch_sizes[@]}"; do
            for i in "${!dag_numbers[@]}"; do
                for j in "${!n_stepss[@]}"; do
                    for k in "${!GNN_models[@]}"; do
                        is_dw="${is_dws[f]}"
                        attn_type="${attn_types[g]}"
                        tpch_size="${tpch_sizes[h]}"
                        dag_number="${dag_numbers[i]}"
                        n_steps="${n_stepss[j]}"
                        GNN_model="${GNN_models[k]}"
                        echo "$is_dw, $attn_type, $tpch_size, $dag_number, $n_steps, $GNN_model" 
                        python ./launchers/launch_static_tpch.py \
                                            --is_dw $is_dw \
                                            --attn_type $attn_type \
                                            --tpch_size $tpch_size \
                                            --dag_number $dag_number\
                                            --n_steps $n_steps \
                                            --GNN_model $GNN_model \
                                            --gpu $gpu &
                        ((count++))
                        if (( count % max_jobs == 0 )); then
                            wait  # 等待当前 batch 的任务
                        fi                                            
                    done
                done
            done
        done
    done
done
wait

echo "All combinations have been processed."

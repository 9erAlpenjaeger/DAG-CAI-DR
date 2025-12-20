#!/bin/bash
is_dws=("nodw") # "dw")
attn_types=("CAI" "WOCAI")
dataset_names=("SIPHT") # "GENOME" "LIGO") # "CYBERSHAKE" "MONTAGE") # "GENOME")
dataset_sizes=("50") # "200" "300" "400") 
n_stepss=("4") # "32")
GNN_models=("GAT") # "Graphormer" "GCN" "RF")
#GNN_models=("RF") 

gpu=1
count=0
max_jobs=12 

for g in "${!attn_types[@]}"; do
    for i in "${!dataset_names[@]}"; do
        for f in "${!is_dws[@]}"; do
            for h in "${!dataset_sizes[@]}"; do
                for j in "${!n_stepss[@]}"; do
                    for k in "${!GNN_models[@]}"; do
                        is_dw="${is_dws[f]}"
                        attn_type="${attn_types[g]}"
                        dataset_size="${dataset_sizes[h]}"
                        dataset_name="${dataset_names[i]}"
                        n_steps="${n_stepss[j]}"
                        GNN_model="${GNN_models[k]}"
                        echo " $is_dw $attn_type, $dataset_name, $dataset_size, $n_steps, $GNN_model" &
                        python ./launchers/launch_static.py \
                                            --is_dw $is_dw \
                                            --attn_type $attn_type \
                                            --dataset_size $dataset_size \
                                            --dataset_name $dataset_name \
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

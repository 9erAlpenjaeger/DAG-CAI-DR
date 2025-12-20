#!/bin/bash
attn_types=("CAI" "WOCAI") 
j_sizes=("10"  "20") 
m_sizes=("10"  "20")
dag_numbers=("16")
seeds=("200")
n_stepss=("4")
GNN_models=("GAT") # "GCN" "RF")

gpu=1
count=0
max_jobs=12

for f in "${!attn_types[@]}"; do
    for g in "${!j_sizes[@]}"; do
        for h in "${!m_sizes[@]}"; do
            for i in "${!dag_numbers[@]}"; do
                for s in "${!seeds[@]}"; do
                    for j in "${!n_stepss[@]}"; do
                        for k in "${!GNN_models[@]}"; do
                            attn_type="${attn_types[f]}"
                            j_size="${j_sizes[g]}"
                            m_size="${m_sizes[h]}"
                            dag_number="${dag_numbers[i]}"
                            seed="${seeds[s]}"
                            n_steps="${n_stepss[j]}"
                            GNN_model="${GNN_models[k]}"
                            echo " $j_size, $m_size, $dag_number, $seed, $n_steps, $GNN_model" 
                            python ./launchers/launch_static_jssp.py \
                                                --attn_type $attn_type \
                                                --j_size $j_size \
                                                --m_size $m_size \
                                                --dag_number $dag_number\
                                                --seed $seed \
                                                --n_steps $n_steps \
                                                --GNN_model $GNN_model \
                                                --gpu $gpu &
                            ((count++))
                            if (( count % max_jobs == 0 )); then
                                wait  
                            fi
                        done
                    done
                done
            done
        done
    done
done
wait

echo "All combinations have been processed."

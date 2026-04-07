# conda activate teal

# OUTPUT_NAME="temp_not_model"
OUTPUT_NAME="temp_sve"

# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --hist_path ../models/Llama-2-7B/histograms --sparsity 0.1 > ${OUTPUT_NAME}_10.txt 2>&1
# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --hist_path ../models/Llama-2-7B/histograms --sparsity 0.2 > ${OUTPUT_NAME}_20.txt 2>&1
# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --hist_path ../models/Llama-2-7B/histograms --sparsity 0.3 > ${OUTPUT_NAME}_30.txt 2>&1
# # --interactive
# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --hist_path ../models/Llama-2-7B/histograms --sparsity 0.4 > ${OUTPUT_NAME}_40.txt 2>&1
# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --hist_path ../models/Llama-2-7B/histograms --sparsity 0.5 > ${OUTPUT_NAME}_50.txt 2>&1
# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --hist_path ../models/Llama-2-7B/histograms --sparsity 0.6 > ${OUTPUT_NAME}_60.txt 2>&1
# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --hist_path ../models/Llama-2-7B/histograms --sparsity 0.7 > ${OUTPUT_NAME}_70.txt 2>&1
# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --hist_path ../models/Llama-2-7B/histograms --sparsity 0.8 > ${OUTPUT_NAME}_80.txt 2>&1
# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --hist_path ../models/Llama-2-7B/histograms --sparsity 0.9 > ${OUTPUT_NAME}_90.txt 2>&1
# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --hist_path ../models/Llama-2-7B/histograms --sparsity 0.95 > ${OUTPUT_NAME}_95.txt 2>&1
# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --hist_path ../models/Llama-2-7B/histograms --sparsity 0.99 > ${OUTPUT_NAME}_99.txt 2>&1


# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --greedy_sparsity_path ../models/Llama-2-7B/lookup --greedy_sparsity_level 0.05 > ${OUTPUT_NAME}_05.txt 2>&1
# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --greedy_sparsity_path ../models/Llama-2-7B/lookup --greedy_sparsity_level 0.1 > ${OUTPUT_NAME}_10.txt 2>&1
# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --greedy_sparsity_path ../models/Llama-2-7B/lookup --greedy_sparsity_level 0.2 > ${OUTPUT_NAME}_20.txt 2>&1
python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --greedy_sparsity_path ../models/Llama-2-7B/lookup --greedy_sparsity_level 0.3 > ${OUTPUT_NAME}_30.txt 2>&1
# # --interactive
# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --greedy_sparsity_path ../models/Llama-2-7B/lookup --greedy_sparsity_level 0.4 > ${OUTPUT_NAME}_40.txt 2>&1
# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --greedy_sparsity_path ../models/Llama-2-7B/lookup --greedy_sparsity_level 0.5 > ${OUTPUT_NAME}_50.txt 2>&1
# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --greedy_sparsity_path ../models/Llama-2-7B/lookup --greedy_sparsity_level 0.6 > ${OUTPUT_NAME}_60.txt 2>&1
# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --greedy_sparsity_path ../models/Llama-2-7B/lookup --greedy_sparsity_level 0.7 > ${OUTPUT_NAME}_70.txt 2>&1
# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --greedy_sparsity_path ../models/Llama-2-7B/lookup --greedy_sparsity_level 0.8 > ${OUTPUT_NAME}_80.txt 2>&1
# python generate.py --checkpoint_path /dev/xvdb/meta-llama/Llama-2-7b/consolidated.00.pth --greedy_sparsity_path ../models/Llama-2-7B/lookup --greedy_sparsity_level 0.9 > ${OUTPUT_NAME}_90.txt 2>&1

#!/usr/bin/env bash
# 
# Runs experiments/summary_critic.py in a loop, improving the prompt
# each iteration based on the previous output.
# 
# Usage:
#   ./experiments/prompt_improvement_loop.sh [llm_summarizer] [llm_critic] [paper_list]
# 
# env variables:
#   UPDATE_ROUNDS: number of rounds to run (default: 4)
#   SUMMARIES_PER_ROUND: number of summaries to generate per round (default: 8)
#   INIT_PROMPT_FN: initial prompt file (default: system_summary_prompt.txt)
# 

llm_summarizer="${1}"
llm_critic="${2}"
paper_list="${3}"

n_rounds=${UPDATE_ROUNDS:-4}
n_summaries_per_round=${SUMMARIES_PER_ROUND:-8}
init_prompt_fn=${INIT_PROMPT_FN:-"system_summary_prompt.txt"}

export prompt_fn="${init_prompt_fn}"

export rand_suffix=`cat /dev/urandom | tr -dc 'a-zA-Z0-9' | head -c 6`

for round in $(seq 1 $n_rounds); do
    echo "Round ${round} / ${n_rounds}"
    echo "=============================="
    echo ""

    python experiments/summary_critic.py \
        --system-prompt "${init_prompt_fn}" \
        --summarizer-config "${llm_summarizer}" \
        --critic-config "${llm_critic}" \
        --papers "${paper_list}" \
        --count "${n_summaries_per_round}" \
        --summarize
    
    # Get the new prompt filename
    export prompt_fn_next=`ls -1rt experiments/summary_criticism/prompt_*_*.txt | tail -n 1`

    # Print diff of new vs. old prompt
    echo "New vs. old prompt diff:"
    echo "------------------------------"
    diff --color=always "${prompt_fn}" "${prompt_fn_next}"
    echo ""

    # Update prompt_fn for the next round
    export prompt_fn="${prompt_fn_next}"

    # Move the old summaries to a backup directory
    export date_suffix=`date +"%Y_%m_%d"`
    export backup_dir="experiments/summary_criticism/backup_${date_suffix}_${rand_suffix}/round_${round}"
    mkdir -p "${backup_dir}"
    mv experiments/summary_criticism/critic_*.txt "${backup_dir}/"
done

echo "=============================="
echo ""
echo "Final prompt:"
echo "------------------------------"
cat "${prompt_fn}"
echo "------------------------------"
echo ""
echo "New prompt filename: ${prompt_fn}"
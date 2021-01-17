import utils
import pandas as pd
import torch
import transformers

def calculate_jaccard_score(original_context, target_string, question_val, idx_start, idx_end):
    
    if idx_end < idx_start:
        idx_end = idx_start
    
    filtered_output  = original_context[idx_start:idx_end+1]

    jac = utils.jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output
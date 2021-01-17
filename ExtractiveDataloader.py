import transformers
import torch
import pandas as pd
import torch.nn as nn

class ExtractionDataset:
    def __init__(self, context, answer, question, config):
        self.context = context
        self.answer = answer
        self.question = question
        self.tokenizer = transformers.RobertaTokenizerFast.from_pretrained(config.TOKENIZER)
        self.max_len = config.MAX_LEN
    
    def __len__(self):
        return len(self.context)
    
    def __getitem__(self, item):
        context = str(self.context[item])
        context = " ".join(context.split())
        answer = str(self.answer[item]['text'])
        answer = " ".join(answer.split())
        question = self.question[item]


        len_sel_txt = len(answer)
        start_indx = -1
        end_indx  = -1

        for indx in (i for i, e in enumerate(context) if e == answer[0]):
          if context[indx:indx + len_sel_txt] == answer:
            start_indx = indx
            end_indx = indx + len_sel_txt - 1
            break

        
        tokenized_contexts = self.tokenizer.encode_plus(context, return_offsets_mapping=True, add_special_tokens=False)
        

        context_ids = tokenized_contexts.input_ids
        context_offset_mapping = tokenized_contexts.offset_mapping
        context_token_type_ids = tokenized_contexts.token_type_ids

        tokenized_inputs = self.tokenizer.encode_plus(question, return_offsets_mapping=True)

        input_ids = tokenized_inputs.input_ids + context_ids + [102]
        
        token_type_ids = tokenized_inputs.token_type_ids + [1] * (len(context_token_type_ids)+1)
        mask = [1]*len(input_ids)
        offset_mapping = tokenized_inputs.offset_mapping + context_offset_mapping + [(0,0)]

        padding_length = self.max_len - len(token_type_ids)

        if padding_length > 0:
          input_ids = input_ids + ([0] * padding_length)
          mask = mask + ([0] * padding_length)
          token_type_ids = token_type_ids + ([0] * padding_length)
          offset_mapping = offset_mapping + ([(0, 0)] * padding_length)
        input_ids = input_ids[:self.max_len]
        mask = mask[:self.max_len]
        token_type_ids = token_type_ids[:self.max_len]
        offset_mapping = offset_mapping[:self.max_len]
        if start_indx > self.max_len:
          start_indx = self.max_len
        if end_indx > self.max_len:
          end_indx = self.max_len
        
        self.tokenizer.save_pretrained(config.DIRECTORY)

        return {'ids': torch.tensor(input_ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets_start': torch.tensor(start_indx, dtype=torch.long),
                'targets_end': torch.tensor(end_indx, dtype=torch.long),
                'orig_context': context,
                'orig_answer': answer,
                'question': question,
                'offset_mapping': torch.tensor(offset_mapping, dtype=torch.long)}
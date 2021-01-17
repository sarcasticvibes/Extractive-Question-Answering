import utils
import torch
import transformers
import pandas as pd
import tqdm
from metrics import calculate_jaccard_score

def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for _, d in enumerate(tk0):

        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        question = d["question"]
        orig_answer = d["orig_answer"]
        orig_context = d["orig_context"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        model.zero_grad()
        outputs = model(ids, 
                        attention_mask=mask, 
                        #token_type_ids=token_type_ids,
                        start_positions=targets_start, 
                        end_positions=targets_end)
        
        loss = outputs[0]
        outputs_start = outputs[1]
        outputs_end = outputs[2]

        loss.backward()
        optimizer.step()
        scheduler.step()

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        jaccard_scores = []
        for px, context in enumerate(orig_context):
            answer_context = orig_answer[px]
            context_question = question[px]
            jaccard_score, _ = calculate_jaccard_score(
                original_context=context,
                target_string=answer_context,
                question_val=context_question,
                idx_start=np.argmax(outputs_start[px, :]),
                idx_end=np.argmax(outputs_end[px, :]),
            )
            jaccard_scores.append(jaccard_score)

        jaccards.update(np.mean(jaccard_scores), ids.size(0))
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)



def eval_fn(data_loader, model, device):
    model.eval()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for _, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            question = d["question"]
            orig_answer = d["orig_answer"]
            orig_context = d["orig_context"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            outputs = model(ids, 
                        attention_mask=mask, 
                        #token_type_ids=token_type_ids,
                        start_positions=targets_start, 
                        end_positions=targets_end)
        
            loss = outputs[0]
            outputs_start = outputs[1]
            outputs_end = outputs[2]
           
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            jaccard_scores = []
            for px, context in enumerate(orig_context):
                answer_context = orig_answer[px]
                context_question = question[px]
                jaccard_score, _ = calculate_jaccard_score(
                    original_context=context,
                    target_string=answer_context,
                    question_val=context_question,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                )
                jaccard_scores.append(jaccard_score)

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
    
    print(f"Jaccard = {jaccards.avg}")
    return jaccards.avg
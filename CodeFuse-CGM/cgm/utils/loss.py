import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

def loss_CGM(output_logits, labels, loss_mask):

    lm_logits = output_logits.contiguous()
    labels = labels.to(device=lm_logits.device).contiguous()
    loss_mask = loss_mask.to(device=lm_logits.device)
    # logits: (bs, l, v); labels, loss_mask: (bs, l)

    # lm loss
    bsz = labels.shape[0]
    loss_func = CrossEntropyLoss(reduction='none')
    losses = loss_func(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))  # logits: (bs * l, v); labels: (bs * l,)
    # losses -> (bs, l)
    losses = losses.contiguous().view(bsz, -1)

    loss_mask = loss_mask.view(-1)
    losses = losses.view(-1)
    if loss_mask.sum() < 1:
        loss_lm = torch.sum(losses * loss_mask)
    else:
        loss_lm = torch.sum(losses * loss_mask) / loss_mask.sum()

    return loss_lm

def acc_lp(logits,labels):
    predictions = torch.sigmoid(logits)
    acc = ((predictions > 0.5) == labels.bool()).float().mean()
    return acc.item()

def loss_lp(outputs, edge_label_dict):
    loss_func = BCEWithLogitsLoss(reduction='mean')
    losses = []
    edge_loss = {}
    edge_acc = {}
    total_acc = 0
    total_edges = 0
    for edge_type in edge_label_dict.keys():
        lm_logits = outputs[edge_type].view(-1)
        labels = edge_label_dict[edge_type].to(device=lm_logits.device).view(-1)
        loss = loss_func(lm_logits,labels)
        losses.append(loss)
        acc = acc_lp(lm_logits, labels)
        edge_loss[edge_type] = loss.item()
        edge_acc[edge_type] = acc
        total_acc += len(labels) * acc
        total_edges += len(labels)
        # del lm_logits, labels, loss
    loss1 = torch.sum(torch.stack(losses))
    total_acc = total_acc / total_edges
    # del losses, loss_func
    return loss1, edge_loss, edge_acc, total_acc

def loss_ng(outputs, y_dict, mask_dict):
    loss_func = MSELoss(reduction='sum')
    loss2 = loss_func(outputs,y_dict['Method'])
    return loss2

def loss_lpng(lp_outputs, ng_outputs, edge_label_dict, y_dict, mask_dict):
    loss1, edge_loss, edge_acc, total_acc = loss_lp(lp_outputs, edge_label_dict)
    loss2 = loss_ng(ng_outputs, y_dict, mask_dict)
    loss = loss1 + loss2
    return loss, loss1.item(), loss2.item(), edge_loss, edge_acc, total_acc

    


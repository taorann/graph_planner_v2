import os
import sys
import math
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from tqdm.auto import tqdm
import time
import datetime
from collections import OrderedDict

sys.path.append("..")
from utils.common_utils import touch_print, print_rank_0
from utils.loss import loss_CGM
from torch.utils.tensorboard import SummaryWriter
from accelerate.logging import get_logger
from torch.cuda.amp import autocast

logger = get_logger(__name__)

task_ids = {
    (0, 'graph_query'),
    (1, 'api'),
    (2, 'issue_fix'),
    (3, 'unit_test'),
    (4, 'readme_summary'),
}

task_to_id = {task: idx for idx, task in task_ids}
id_to_task = {idx: task for idx, task in task_ids}

def check_weight_dtype(model):
    for name, param in model.named_parameters():
        print_rank_0(f"Layer {name}: {param.dtype}")

def write_tensorboard(summary_writer: SummaryWriter, log_dict: dict, completed_steps):
    for key, value in log_dict.items():
        summary_writer.add_scalar(f'{key}', value, completed_steps)

def accelerate_saving_checkpoint_CGM(accelerator, model, tokenizer, output_dir: str, completed_steps: int, args):
    accelerator.wait_for_everyone()

    accelerator.print(f"[CHECKPOINT] Saving checkpoint")
    unwrapped_model = accelerator.unwrap_model(model)

    save_encoder = False
    save_adapter = False
    save_lm = False
    if 'e' in args.mode: 
        save_encoder = True
    if 'a' in args.mode: 
        save_adapter = True
    if 'l' in args.mode: 
        save_lm = True

    if accelerator.is_main_process:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        tokenizer.save_pretrained(output_dir)

        if save_adapter: 
            torch.save(accelerator.get_state_dict(model.adapter), f"{output_dir}/adapter.pth")
        
    
    if save_encoder:
        unwrapped_model.encoder.save_pretrained(
            f"{output_dir}/encoder",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model.encoder)
        )

    if save_lm:
        unwrapped_model.lm.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model.lm)
        )

    accelerator.print(
        f"[CHECKPOINT][complete_steps={completed_steps}], checkpoint {output_dir} saved"
    )

    accelerator.wait_for_everyone()

def accelerate_evaluate_CGM(accelerator, model, tokenizer, valid_dataloader, args, completed_steps, step, min_eval_loss, stall_num,
                        best_step, summary_writer):

    losses = []
    end_eval = False
    eval_step = 0
    for batch in valid_dataloader:
        with torch.no_grad():
            if args.task == 'mft':
                task = batch['task']

            x = batch['x']
            y = batch['y']
            loss_mask = batch['loss_mask']
            qa_mask = batch['qa_mask']
            embeddings = batch['embeddings']

            len_y = y.shape[1]

            outputs = model(
                graph_embeddings = embeddings,
                qa_embeddings = x,
                qa_mask = qa_mask 
            )
            output_logits = outputs['logits'][:,-len_y:,:]

            if args.task == 'mft':
                loss_dict = {task_id: torch.tensor(0.0, device=output_logits.device) for task_id, _ in task_ids}

                task_id = task.item()
                loss_dict[task_id] += loss_CGM(
                    output_logits = output_logits,
                    labels = y,
                    loss_mask = loss_mask,
                )

                loss = sum(loss_dict.values())
            else:
                loss = loss_CGM(
                    output_logits = output_logits,
                    labels = y,
                    loss_mask = loss_mask,
                )
                
            eval_step += 1

            losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))
    
    accelerator.wait_for_everyone()
    valid_batch_num = len(losses)
    gathered_size = losses[0].shape
    losses = torch.cat(losses)

    try:
        eval_loss = torch.mean(losses)
        if eval_loss <= min_eval_loss:
            min_eval_loss = eval_loss
            stall_num = 0
            best_step = completed_steps
        else:
            stall_num += 1
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    logger.info(f"[EVAL][global_steps={step + 1}][completed_steps={completed_steps}]"
                f"[valid_batch_num={valid_batch_num}], [gather_size={gathered_size}]"
                f"[perplexity={perplexity:.4f}][eval_loss={eval_loss:.6f}]")
    eval_log_dict = {
        "valid/valid_loss": eval_loss.float(),
        "valid/perplexity": perplexity
    }

    if accelerator.is_main_process:
        write_tensorboard(summary_writer, eval_log_dict, completed_steps)

    return eval_loss, min_eval_loss, stall_num, best_step

def accelerate_evaluate_CGM_mft(accelerator, model, tokenizer, valid_dataloader, args, completed_steps, step, min_eval_loss, stall_num,
                        best_step, summary_writer):

    losses = []
    task_eval_counts = {}
    task_losses = {}
    eval_step = 0

    for batch in valid_dataloader:
        with torch.no_grad():
            if args.task == 'mft':
                task = batch['task']
                task_id = task.item()
                if task_id not in task_eval_counts:
                    task_eval_counts[task_id] = 0
                    task_losses[task_id] = []
                if task_eval_counts[task_id] >= 50:
                    continue

            x = batch['x']
            y = batch['y']
            loss_mask = batch['loss_mask']
            qa_mask = batch['qa_mask']
            embeddings = batch['embeddings']

            len_y = y.shape[1]

            outputs = model(
                graph_embeddings = embeddings,
                qa_embeddings = x,
                qa_mask = qa_mask 
            )
            output_logits = outputs['logits'][:,-len_y:,:]

            if args.task == 'mft':
                loss = loss_CGM(
                    output_logits = output_logits,
                    labels = y,
                    loss_mask = loss_mask,
                )
                task_losses[task_id].append(loss.item())
            else:
                loss = loss_CGM(
                    output_logits = output_logits,
                    labels = y,
                    loss_mask = loss_mask,
                )
                
            eval_step += 1
            losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))
            task_eval_counts[task_id] += 1

    accelerator.wait_for_everyone()
    valid_batch_num = len(losses)
    gathered_size = losses[0].shape
    losses = torch.cat(losses)

    try:
        eval_loss = torch.mean(losses)
        if eval_loss <= min_eval_loss:
            min_eval_loss = eval_loss
            stall_num = 0
            best_step = completed_steps
        else:
            stall_num += 1
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    logger.info(f"[EVAL][global_steps={step + 1}][completed_steps={completed_steps}]"
                f"[valid_batch_num={valid_batch_num}], [gather_size={gathered_size}]"
                f"[perplexity={perplexity:.4f}][eval_loss={eval_loss:.6f}]")
    
    for task_id, task_loss_list in task_losses.items():
        task_eval_loss = sum(task_loss_list) / len(task_loss_list) if task_loss_list else 0.0
        logger.info(f"[EVAL][task_id={task_id}][task_loss={task_eval_loss:.6f}]")
        eval_log_dict = {
            "valid/valid_loss": eval_loss.float(),
            "valid/perplexity": perplexity,
            f"valid/{id_to_task[task_id]}": task_eval_loss
        }

    if accelerator.is_main_process:
        write_tensorboard(summary_writer, eval_log_dict, completed_steps)

    return eval_loss, min_eval_loss, stall_num, best_step

def accelerate_monitor_CGM_mft(accelerator, reduce_loss_dict, args, completed_steps,
                       lr_scheduler, optimizer, summary_writer):

    """
    gather reduce_loss from all N devices.
    train logging and tensorboarding.
    """
    # gathered_loss_dict = {task_id: accelerator.gather(reduce_loss) for task_id, reduce_loss in reduce_loss_dict.items()}
    gathered_loss_dict = {task_id: reduce_loss for task_id, reduce_loss in reduce_loss_dict.items()}
    print_rank_0(f"*******************gathered_loss_dict*******************")
    print_rank_0(gathered_loss_dict)

    train_log_dict = {
        f"train/{id_to_task[task_id]}": torch.mean(gathered_loss) / max(reduce_loss_count_dict[task_id], 1)
        for task_id, gathered_loss in gathered_loss_dict.items()
    }
    train_log_dict["train/lr"] = optimizer.param_groups[0]['lr']

    logger.info(
        f"[TRAIN][completed_steps={completed_steps}]"
        f"[lr={optimizer.param_groups[0]['lr']:.4e}]",
    )
    for task_id, train_loss in train_log_dict.items():
        if task_id != "train/lr":
            logger.info(f"{task_id}={train_loss:.6f}")

    if accelerator.is_main_process:
        write_tensorboard(summary_writer, train_log_dict, completed_steps)


def accelerate_monitor_CGM(accelerator, reduce_loss, args, completed_steps,
                       lr_scheduler, optimizer, summary_writer):

    """
    gather reduce_loss from all N devices.
    train logging and tensorboarding.
    """
    reduce_losses = accelerator.gather(reduce_loss)
    # reduce_losses = reduce_loss

    train_loss = torch.mean(reduce_losses) / (args.log_interval * args.gradient_accumulation_steps)


    logger.info(
        f"[TRAIN][complete_steps={completed_steps}][train_loss={train_loss:.6f}]"
        f"[gather shape={reduce_losses.shape}][lr={optimizer.param_groups[0]['lr']:.4e}]",
    )

    train_log_dict = {
        "train/train_loss": train_loss,
        "train/lr": optimizer.param_groups[0]['lr']
    }

    if accelerator.is_main_process:
        write_tensorboard(summary_writer, train_log_dict, completed_steps)


def accelerate_train_CGM(accelerator, model, train_dataloader, valid_dataloader, optimizer, lr_scheduler, tokenizer,
                     total_train_dataset_size, args):

    summary_writer = SummaryWriter(log_dir=args.tb_dir, filename_suffix=args.tb_dir.split('/')[-1]) if accelerator.is_main_process else None
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("**************************************** Running training ****************************************")
    logger.info(f"  Num examples = {total_train_dataset_size}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total global train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization(update/completed) steps = {args.max_train_steps}")
    logger.info(f"  Complete/Optimization steps per Epoch = {args.max_train_steps // args.num_train_epochs}")
    logger.info("***************************************************************************************************")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    # check_weight_dtype(model.lm)
    # exit()
    # set starting_epoch, completed_steps and resume_step of train_dataloader
    completed_steps = 0
    starting_epoch = 0

    # monitor minimum eval_loss, stalling num, and best_step
    min_eval_loss = float('inf')
    eval_loss = 100.0
    checkpoint_eval_loss = float('inf')
    checkpoint_stall_num = 0
    stall_num = 0
    best_step = None
    
    reduce_loss = 0
    reduce_loss_dict = OrderedDict((task_id, 0) for task_id, _ in task_ids)
    reduce_loss_count_dict = OrderedDict((task_id, 0) for task_id, _ in task_ids)
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(model):

                graph = batch['graph']
                x = batch['x']
                y = batch['y'].unsqueeze(0)
                loss_mask = batch['loss_mask'].unsqueeze(0)
                qa_mask = batch['qa_mask'].unsqueeze(0)

                if args.task == 'mft':
                    task = batch['task']

                outputs = model(
                    graph = graph,
                    qa_ids = x,
                    qa_mask = qa_mask,
                )

                len_y = y.shape[1]
                output_logits = outputs['logits'][:,-len_y:,:]

                if args.task == 'mft':
                    loss_dict = {task_id: torch.tensor(0.0, device=output_logits.device) for task_id, _ in task_ids}
                    
                    task_id = task.item()
                    loss_dict[task_id] += loss_CGM(
                        output_logits = output_logits,
                        labels = y,
                        loss_mask = loss_mask,
                    )

                    loss = sum(loss_dict.values())
                else:
                    loss = loss_CGM(
                        output_logits = output_logits,
                        labels = y,
                        loss_mask = loss_mask,
                    )

                accelerator.backward(loss)

                optimizer.step()
                if not args.lr_scheduler_type == "reduce_lr_on_plateau":
                    lr_scheduler.step()
                optimizer.zero_grad()

                if optimizer.param_groups[0]['lr'] <= args.min_lr:
                    optimizer.param_groups[0]['lr'] = args.min_lr

                if args.task == 'mft':
                    for task_id in reduce_loss_dict.keys():
                        if not torch.isnan(loss_dict[task_id]):
                            reduce_loss_dict[task_id] += loss_dict[task_id].detach().float()
                            reduce_loss_count_dict[task_id] += 1
                else:

                    if not torch.isnan(loss):
                        reduce_loss += loss.detach().float()
                    else:
                        logger.info("loss nan")

                if accelerator.sync_gradients:
                    completed_steps += 1
                    if args.task == 'mft':
                        reduce_loss = sum(reduce_loss_dict.values())
                    logger.info(f"accelerator step (accumulate) {completed_steps}, loss: {reduce_loss}")
                        
                    if completed_steps % args.log_interval == 0:
                        if args.task == 'mft':
                            progress_bar.update(args.log_interval)
                            accelerate_monitor_CGM_mft(
                                accelerator, reduce_loss_dict, reduce_loss_count_dict, args, completed_steps, 
                                lr_scheduler, optimizer, summary_writer
                            )
                            reduce_loss_dict = OrderedDict((task_id, 0) for task_id, _ in task_ids)
                            reduce_loss_count_dict = OrderedDict((task_id, 0) for task_id, _ in task_ids)
                        else:
                            if isinstance(reduce_loss, torch.Tensor):
                                progress_bar.update(args.log_interval)
                                accelerate_monitor_CGM(
                                    accelerator, reduce_loss, args, completed_steps, 
                                    lr_scheduler, optimizer, summary_writer
                                )
                            reduce_loss = 0

                    # steps checkpointing
                    if args.step_checkpointing and completed_steps % args.checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerate_saving_checkpoint_CGM(accelerator, model, tokenizer, output_dir, completed_steps, args)

                    if args.step_evaluation and completed_steps % args.evaluation_steps == 0:
                        logger.info(f"start evaluation...")
                        model.eval()
                        model.lm.gradient_checkpointing_disable()
                        model.lm.config.use_cache = True
                        if args.task == 'mft':
                            eval_loss, min_eval_loss, stall_num, best_step = accelerate_evaluate_CGM_mft(
                                accelerator, model, tokenizer, valid_dataloader, args, completed_steps, step,
                                min_eval_loss, stall_num, best_step, summary_writer
                            )
                        else:
                            eval_loss, min_eval_loss, stall_num, best_step = accelerate_evaluate_CGM(
                                accelerator, model, tokenizer, valid_dataloader, args, completed_steps, step,
                                min_eval_loss, stall_num, best_step, summary_writer
                            )
                        model.train()
                        model.lm.gradient_checkpointing_enable()
                        model.lm.config.use_cache = False

                        if args.lr_scheduler_type == "reduce_lr_on_plateau":
                            lr_scheduler.step(eval_loss)

                        if eval_loss < checkpoint_eval_loss:
                            checkpoint_eval_loss = eval_loss
                            output_dir = f"step_{completed_steps}_stall_{checkpoint_stall_num}"
                            if args.output_dir is not None:
                                output_dir = os.path.join(args.output_dir, output_dir)
                            accelerate_saving_checkpoint_CGM(accelerator, model, tokenizer, output_dir, completed_steps, args)
                            checkpoint_stall_num = 0
                        else:
                            if checkpoint_stall_num < 2:
                                output_dir = f"step_{completed_steps}_stall_{checkpoint_stall_num}"
                                if args.output_dir is not None:
                                    output_dir = os.path.join(args.output_dir, output_dir)
                                accelerate_saving_checkpoint_CGM(accelerator, model, tokenizer, output_dir, completed_steps, args)
                                checkpoint_stall_num += 1
                            
                            if args.lr_scheduler_type == "reduce_lr_on_plateau":
                                pass
                            elif args.lr_scheduler_type == 'cosine':
                                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.33
                                lr_scheduler.base_lrs = optimizer.param_groups[0]['lr']
                                lr_scheduler.step()
                            else:
                                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.33
                                lr_scheduler.step()

                    # adapter warmup
                    if args.adapter_warmup and completed_steps >= args.adapter_warmup_steps:
                        if 'l' in args.mode:
                            for param in model.lm.parameters():
                                param.requires_grad = True
                        args.adapter_warmup = False

                    accelerator.wait_for_everyone()

            load_t = time.time()

        if args.epoch_evaluation:
            model.eval()
            model.lm.gradient_checkpointing_disable()
            model.lm.config.use_cache = True
            if args.task == 'mft':
                eval_loss, min_eval_loss, stall_num, best_step = accelerate_evaluate_CGM_mft(
                accelerator, model, tokenizer, valid_dataloader, args, completed_steps, step,
                min_eval_loss, stall_num, best_step, summary_writer
                )
            else:
                eval_loss, min_eval_loss, stall_num, best_step = accelerate_evaluate_CGM(
                    accelerator, model, tokenizer, valid_dataloader, args, completed_steps, step,
                    min_eval_loss, stall_num, best_step, summary_writer
                )
            model.train()
            model.lm.gradient_checkpointing_enable()
            model.lm.config.use_cache = False 

            if args.lr_scheduler_type == "reduce_lr_on_plateau":
                lr_scheduler.step(eval_loss)

            if eval_loss < checkpoint_eval_loss:
                checkpoint_eval_loss = eval_loss
                output_dir = f"epoch_{epoch}"
                ckpt_tag = output_dir
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerate_saving_checkpoint_CGM(accelerator, model, tokenizer, output_dir, completed_steps, args)
            else:
                if args.lr_scheduler_type == "reduce_lr_on_plateau":
                    pass
                    # lr_scheduler.step(eval_loss)
                elif args.lr_scheduler_type == 'cosine':
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.33
                    lr_scheduler.base_lrs = optimizer.param_groups[0]['lr']
                    lr_scheduler.step()
                else:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.33
                    lr_scheduler.step()

        # epoch checkpointing
        if args.epoch_checkpointing:
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerate_saving_checkpoint_CGM(accelerator, model, tokenizer, output_dir, completed_steps, args)

    if summary_writer:
        summary_writer.close()

    output_dir = f"final_step_{completed_steps}"
    if args.output_dir is not None:
        output_dir = os.path.join(args.output_dir, output_dir)
    accelerate_saving_checkpoint_CGM(accelerator, model, tokenizer, output_dir, completed_steps, args)

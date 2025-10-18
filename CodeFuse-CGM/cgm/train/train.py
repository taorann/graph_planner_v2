import os
import sys

sys.path.append(os.getcwd())

import time, os, json, math, logging
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, random_split, Subset
import random
# from deepspeed.ops.adam import FusedAdam as AdamW
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import (
    set_seed,
    get_scheduler,
)
from utils.arguments import prepare_args

from modeling.cgm import CGM

from data.encode import CGMEncoder

from utils.common_utils import print_args, print_with_rank, print_rank_0
from utils.train_utils import accelerate_train_CGM

from datasets import load_dataset
import datetime
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

from torch.optim.lr_scheduler import ReduceLROnPlateau


# from load_hetero_dataset import load_dataset, perpare_dataloader

def str_to_tuple(s):
    st = s.strip('()')
    return tuple(item.strip().strip("'") for item in st.split(','))

def getRawGraph(filename, suffix="json"):
    if os.path.exists(filename):
        if suffix == 'json':
            with open(filename) as f:
                example_graph = json.load(f)
            f.close()
        elif suffix == 'pt':
            with open(filename, 'rb') as f:
                example_graph = torch.load(f)
                # example_graph = torch.load(filename)
            f.close()
        return example_graph
    return None

task_ids = {
    (0, 'graph_query'),
    (1, 'api'),
    (2, 'issue_fix'),
    (3, 'unit_test'),
    (4, 'readme_summary'),
}

task_to_id = {task: idx for idx, task in task_ids}

def collate_cgm(graph_dir, encoder, qa_type='mft', seq_l=8192, use_chat=True):
    def collate(batches):
        result_batches = []
        for batch in batches:
            result_batch = {}
            graph = getRawGraph(batch['repo'], suffix='json')

            if graph is not None:
                graph['reponame'] = batch['repo'].split('/')[-1].split('.')[0]
                graph['language'] = batch['language']
                result_batch['graph'] = graph
                if use_chat:
                    features = encoder.dataToInput(batch)
                    input_ids = features['input_ids']
                    loss_mask = features['loss_mask']
                    qa_mask = features['qa_mask']
                else:
                    query_ids = encoder.tokenizer.encode(batch['prompt'], add_special_tokens=False)
                    answer_ids = encoder.tokenizer.encode(batch['answer'], add_special_tokens=False) + [
                        encoder.tokenizer.eos_token_id]
                    qa_mask = [1] * len(query_ids) + [0] * len(answer_ids)
                    loss_mask = [0] * len(query_ids) + [1] * len(answer_ids)
                    input_ids = query_ids + answer_ids

                min_seq = min(seq_l, len(input_ids))
                result_batch['x'] = torch.tensor(input_ids, dtype=torch.int64)[:min_seq - 1].contiguous()
                result_batch['qa_mask'] = torch.tensor(qa_mask, dtype=torch.bool)[:min_seq - 1].contiguous()
                result_batch['y'] = torch.tensor(input_ids, dtype=torch.int64)[1:min_seq].contiguous()
                result_batch['loss_mask'] = torch.tensor(loss_mask, dtype=torch.bool)[1:min_seq].contiguous()

                if qa_type == 'mft':
                    result_batch['task'] = task_to_id[batch['task']]
            else:
                raise ValueError(f"graph none for {batch['repo']}")

            result_batches.append(result_batch)

        final_result_batch = {}
        for key in result_batches[0].keys():
            if key == 'task':
                final_result_batch[key] = torch.tensor([rb[key] for rb in result_batches])
            elif key == 'graph':
                final_result_batch[key] = [rb[key] for rb in result_batches]
            else:
                final_result_batch[key] = torch.stack([rb[key] for rb in result_batches])
        return final_result_batch

    return collate


def train(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    print_args(args, accelerator)

    # prepare logger
    logger = get_logger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)

    train_files = args.train_files
    valid_files = args.valid_files

    graph_dir = args.graph_dir

    dataset = load_dataset('json', data_files={'train': train_files, 'valid': valid_files})

    train_dataset = dataset['train']
    valid_dataset = dataset['valid']

    epoch_train = len(train_dataset)

    if args.peft:
        save_suffix = args.framework_type + '_' + str(
            args.pretrained_model_path.split('/')[-1]) + '_' + args.task + '_M' + args.mode + '_LR' + str(
            args.learning_rate) + '_GA' + str(args.gradient_accumulation_steps) + '_' + str(args.peft) + '_r' + str(
            args.lora_rank) + '_alpha' + str(args.lora_alpha) + '_d' + str(args.lora_dropout) + '_m' + str(
            args.lora_modules) + str(datetime.datetime.now().strftime('%Y%m%d%H')) + '/'
    else:
        save_suffix = args.framework_type + '_' + str(
            args.pretrained_model_path.split('/')[-1]) + '_' + args.task + '_M' + args.mode + '_LR' + str(
            args.learning_rate) + '_GA' + str(args.gradient_accumulation_steps) + '_' + str(
            datetime.datetime.now().strftime('%Y%m%d%H')) + '/'

    args.output_dir = args.output_dir + save_suffix
    args.tb_dir = args.tb_dir + save_suffix
    if 'l' not in args.mode and args.peft:
        args.mode = args.mode + 'l'
    if args.peft == "QLoRA":
        if not args.quantization:
            args.quantization = "4bit"

    if accelerator.is_main_process:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if not os.path.exists(args.tb_dir):
            os.makedirs(args.tb_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer_path, trust_remote_code=False)

    encoder = CGMEncoder(tokenizer=tokenizer, config_name=args.model_type)

    collate_fn = collate_cgm(
        graph_dir,
        encoder,
        qa_type=args.task,
        seq_l=8192,
        use_chat=args.use_chat,
    )

    train_unit_batch_size = accelerator.num_processes * args.gradient_accumulation_steps * args.per_device_train_batch_size
    total_train_samples = len(train_dataset)
    if args.max_train_samples:
        max_train_samples = args.max_train_samples
        if total_train_samples > max_train_samples:
            total_train_samples = max_train_samples

    max_divisible_samples = (total_train_samples // train_unit_batch_size) * train_unit_batch_size
    subset_indices = list(range(max_divisible_samples))
    train_subset = Subset(train_dataset, subset_indices)
    train_dataloader = DataLoader(train_subset, batch_size=args.per_device_train_batch_size, collate_fn=collate_fn,
                                  shuffle=True)

    if args.max_valid_samples:
        max_valid_samples = args.max_valid_samples
        valid_unit_batch_size = accelerator.num_processes * args.per_device_eval_batch_size
        total_valid_samples = len(valid_dataset)
        if total_valid_samples > max_valid_samples:
            indices = list(range(max_valid_samples))
            random.shuffle(indices)
            subset_indices = indices[:max_valid_samples]
            valid_subset = Subset(valid_dataset, subset_indices)
        else:
            max_divisible_samples = (total_valid_samples // valid_unit_batch_size) * valid_unit_batch_size
            subset_indices = list(range(max_divisible_samples))
            valid_subset = Subset(valid_dataset, subset_indices)
        valid_dataloader = DataLoader(valid_subset, batch_size=args.per_device_eval_batch_size, collate_fn=collate_fn,
                                      shuffle=True)
    else:
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=collate_fn,
                                      shuffle=True)

    logger.info(f"Train Samples: {len(train_dataloader)}", main_process_only=True)
    logger.info(f"Valid Samples: {len(valid_dataloader)}", main_process_only=True)

    model = CGM(args)
    # Please disable checkpointing and re-enable use-cache for inference
    model.lm.gradient_checkpointing_enable()
    model.lm.config.use_cache = False

    if args.peft == "QLoRA":
        model.lm = prepare_model_for_kbit_training(model.lm)  # use_gradient_checkpointing default is True
    else:
        model.lm.gradient_checkpointing_enable()

    if args.peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_modules,
            bias="lora_only",
        )
        model.lm = get_peft_model(model.lm, peft_config)

        encoder_peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.enc_lora_rank,
            lora_alpha=args.enc_lora_alpha,
            lora_dropout=args.enc_lora_dropout,
            target_modules=args.enc_lora_modules,
            bias="lora_only",
        )
        model.encoder = get_peft_model(model.encoder, encoder_peft_config)

    if args.adapter_warmup:
        if 'l' in args.mode:
            for param in model.lm.parameters():
                param.requires_grad = False

    encoder_params = list(model.encoder.parameters()) if 'e' in args.mode else []
    pma_params = list(model.pma.parameters()) if 'p' in args.mode else []
    adapter_params = list(model.adapter.parameters()) if 'a' in args.mode else []
    lm_params = list(model.lm.parameters()) if 'l' in args.mode else []

    trained_params = encoder_params + pma_params + adapter_params + lm_params
    # trained_params = adapter_params + lm_params
    if not trained_params:
        raise ValueError("No parameters to train. Please check the mode argument.")

    optimizer = AdamW(
        trained_params,
        weight_decay=args.weight_decay,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
    )
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(epoch_train / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.lr_scheduler_type == "reduce_lr_on_plateau":
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.75,
            patience=3,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=args.min_lr,
            eps=1e-08,
        )
    else:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps,
        )

    logger.info(
        f"{'==' * 100}\nbefore accelerator preparation: [dataloader: {epoch_train}][epochs: {args.num_train_epochs}][total steps: {args.max_train_steps}]\n{'==' * 100}")
    if torch.cuda.is_available():
        model, train_dataloader, valid_dataloader, optimizer, lr_scheduler = accelerator.prepare(
            model, train_dataloader, valid_dataloader, optimizer, lr_scheduler
        )

    epoch_train = epoch_train / accelerator.num_processes
    num_update_steps_per_epoch = math.ceil(epoch_train / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    logger.info(
        f"{'==' * 100}\nafter accelerator preparation: [dataloader: {epoch_train}][epochs: {args.num_train_epochs}][total steps: {args.max_train_steps}]\n{'==' * 100}")

    logger.info(f"{'==' * 100}Training...")

    accelerate_train_CGM(accelerator,
                         model,
                         train_dataloader,
                         valid_dataloader,
                         optimizer,
                         lr_scheduler,
                         tokenizer,
                         epoch_train,
                         args)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = prepare_args()
    set_seed(args.seed)
    train(args)

# Template: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' 
# CodeQwen1.5-7B-Chat, Qwen2-72B-Instruct
QwenTokenizerConfig = {
    'seq_length': 8192,
    'SYSTEM': 'system',
    'HUMAN': 'user',
    'BOT': 'assistant',
    'SENTENCE_START_MARKER': '',
    'SENTENCE_END_MARKER': '',
    'SYSTEM_START_MARKER': '<|im_start|>',
    'SYSTEM_START_MARKER_2': '\n',
    'SYSTEM_END_MARKER': '<|im_end|>\n',
    'HUMAN_START_MARKER': '<|im_start|>',
    'HUMAN_START_MARKER_2': '\n',
    'HUMAN_END_MARKER': '<|im_end|>\n',
    'BOT_START_MARKER': '<|im_start|>',
    'BOT_START_MARKER_2': '\n',
    'BOT_END_MARKER': '<|im_end|>\n',
}

# Template: <｜begin▁of▁sentence｜>{system_message}<｜User｜>{user_message_1}<｜Assistant｜>{assistant_message_1}<｜end▁of▁sentence｜><｜User｜>{user_message_2}<｜Assistant｜>
# DeepSeek-V2.5
DeepSeekTokenizerConfig = {
    'seq_length': 8192,
    'SYSTEM': 'system',
    'HUMAN': 'user',
    'BOT': 'assistant',
    'SENTENCE_START_MARKER': '<｜begin▁of▁sentence｜>',
    'SENTENCE_END_MARKER': '<｜end▁of▁sentence｜>',
    'SYSTEM_START_MARKER': '',
    'SYSTEM_START_MARKER_2': '',
    'SYSTEM_END_MARKER': '',
    'HUMAN_START_MARKER': '<｜User｜>',
    'HUMAN_START_MARKER_2': '',
    'HUMAN_END_MARKER': '',
    'BOT_START_MARKER': '<｜Assistant｜>',
    'BOT_START_MARKER_2': '',
    'BOT_END_MARKER': '',
}

DeepSeekCoderTokenizerConfig = {
    'seq_length': 8192,
    'SYSTEM': 'system',
    'HUMAN': 'user',
    'BOT': 'assistant',
    'SENTENCE_START_MARKER': '<｜begin▁of▁sentence｜>',
    'SENTENCE_END_MARKER': '<｜end▁of▁sentence｜>',
    'SYSTEM_START_MARKER': '',
    'SYSTEM_START_MARKER_2': '',
    'SYSTEM_END_MARKER': '\n\n',
    'HUMAN_START_MARKER': 'User: ',
    'HUMAN_START_MARKER_2': '',
    'HUMAN_END_MARKER': '\n\n',
    'BOT_START_MARKER': 'Assistant: ',
    'BOT_START_MARKER_2': '',
    'BOT_END_MARKER': '',
}

def format_eol(text):
    if not text.endswith("\n"):
        text += "\n"
    return text

def get_template(data):
    template = [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': data['prompt']},
        {'role': 'assistant', 'content': data['answer']}
    ]

def get_config(name):
    if name == 'Qwen':
        return QwenTokenizerConfig
    elif name == 'DeepSeek':
        return DeepSeekTokenizerConfig
    elif name == 'DeepSeek-Coder':
        return DeepSeekCoderTokenizerConfig
    else:
        raise NotImplementedError

class BaseEncoder(object):
    def __init__(self, tokenizer, config_name):
        # self.args = args
        # seq_length - 1 for shifting
        # self.seq_length = args.seq_length - 1
        config = get_config(config_name)
        self.tokenizer = tokenizer
        # self.seq_length = tokenizer.model_max_length
        self.seq_length = config.get('seq_length')
        
        # TODO: default Qwen
        self.SYSTEM = config.get('SYSTEM')
        self.HUMAN = config.get('HUMAN')
        self.BOT = config.get('BOT')

        self.SENTENCE_START_MARKER = config.get('SENTENCE_START_MARKER')
        self.SENTENCE_END_MARKER = config.get('SENTENCE_END_MARKER'),

        self.SYSTEM_START_MARKER = config.get('SYSTEM_START_MARKER')
        self.SYSTEM_START_MARKER_2 = config.get('SYSTEM_START_MARKER_2')
        self.SYSTEM_END_MARKER = config.get('SYSTEM_END_MARKER')

        self.HUMAN_START_MARKER = config.get('HUMAN_START_MARKER')
        self.HUMAN_START_MARKER_2 = config.get('HUMAN_START_MARKER_2')
        self.HUMAN_END_MARKER = config.get('HUMAN_END_MARKER')

        self.BOT_START_MARKER = config.get('BOT_START_MARKER')
        self.BOT_START_MARKER_2 = config.get('BOT_START_MARKER_2')
        self.BOT_END_MARKER = config.get('BOT_END_MARKER')

        self.sentence_start_ids = self.tokenizer.encode(f"{self.SENTENCE_START_MARKER}", add_special_tokens=False) if self.SENTENCE_START_MARKER != '' else []
        self.sentence_end_ids = self.tokenizer.encode(f"{self.SENTENCE_END_MARKER}", add_special_tokens=False) if self.SENTENCE_END_MARKER != '' else []

        self.system_start_ids = self.tokenizer.encode(f"{self.SYSTEM_START_MARKER}{self.SYSTEM}{self.SYSTEM_START_MARKER_2}", add_special_tokens=False) 
        self.system_end_ids = self.tokenizer.encode(f"{self.SYSTEM_END_MARKER}", add_special_tokens=False) if self.SYSTEM_END_MARKER != '' else []
        
        self.human_start_ids = self.tokenizer.encode(f"{self.HUMAN_START_MARKER}{self.HUMAN}{self.HUMAN_START_MARKER_2}", add_special_tokens=False)
        self.human_end_ids = self.tokenizer.encode(f"{self.HUMAN_END_MARKER}", add_special_tokens=False) if self.HUMAN_END_MARKER != '' else []
        
        self.bot_start_ids = self.tokenizer.encode(f"{self.BOT_START_MARKER}{self.BOT}{self.BOT_START_MARKER_2}", add_special_tokens=False)
        self.bot_end_ids = self.tokenizer.encode(f"{self.BOT_END_MARKER}", add_special_tokens=False) if self.BOT_END_MARKER != '' else []

        self.end_ids = [self.tokenizer.eos_token_id]

    def padding(self, input_ids, loss_mask, qa_mask):
        pad_id = self.tokenizer.pad_token_id

        assert len(input_ids) <= self.seq_length, f"padding sequence: {len(input_ids)} > {self.seq_length}"
        input_ids += [pad_id] * (self.seq_length - len(input_ids))
        loss_mask += [0] * (self.seq_length - len(loss_mask))
        qa_mask += [0] * (self.seq_length - len(loss_mask))
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "qa_mask": qa_mask
        }

class CGMEncoder(BaseEncoder):
    def __init__(self, tokenizer, config_name):
        super().__init__(tokenizer, config_name)

    def dataToInput(self, data, seg_role = None):
        input_ids, loss_mask, qa_mask = [], [], []
        # TODO: expand
        message = [
            {'role': self.HUMAN, 'content': 'prompt', 'marker': True, 'loss': 0},
            {'role': self.BOT, 'content': 'answer', 'marker': True, 'loss': 1}
        ]
        if seg_role is not None:
            message = [item for item in message if item['role'] == seg_role]

        input_ids += self.sentence_start_ids
        loss_mask += [0] * len(self.sentence_start_ids)
        qa_mask += [0] * len(self.sentence_start_ids)

        for segment in message:
            role = segment['role']
            content = segment['content']
            marker = segment['marker']
            loss = segment['loss']

            if role == self.SYSTEM:
                system_ids = self.tokenizer.encode(str(data[content]), add_special_tokens=False)
                if marker:
                    input_ids += self.system_start_ids + system_ids + self.system_end_ids
                    loss_mask += [0] * len(self.system_start_ids) + [loss] * len(system_ids) + [0] * len(self.system_end_ids)
                    qa_mask += [0] * len(self.system_start_ids) + [1] * len(system_ids) + [0] * len(self.system_end_ids)
                else:
                    input_ids += system_ids
                    loss_mask += [loss] * len(system_ids)
                    qa_mask += [loss] * len(system_ids)

            elif role == self.HUMAN:

                human_ids = self.tokenizer.encode(str(data[content]), add_special_tokens=False)
                if marker:
                    input_ids += self.human_start_ids + human_ids + self.human_end_ids
                    loss_mask += [0] * len(self.human_start_ids) + [loss] * len(human_ids) + [0] * len(self.human_end_ids)
                    qa_mask += [0] * len(self.human_start_ids) + [1] * len(human_ids) + [0] * len(self.human_end_ids)
                else:
                    input_ids += human_ids
                    loss_mask += [loss] * len(human_ids)
                    qa_mask += [1] * len(human_ids)

            elif role == self.BOT:
                bot_ids = self.tokenizer.encode(str(data[content]), add_special_tokens=False)
                if marker:
                    input_ids += self.bot_start_ids + bot_ids + self.bot_end_ids
                    loss_mask += [0] * len(self.bot_start_ids) + [loss] * len(bot_ids) + [0] * len(self.bot_end_ids)
                    qa_mask += [0] * len(self.bot_start_ids) + [0] * len(bot_ids) + [0] * len(self.bot_end_ids)
                else:
                    input_ids += bot_ids
                    loss_mask += [loss] * len(bot_ids)
                    qa_mask += [0] * len(bot_ids)

            else:
                raise ValueError(f"wrong {role} for {config_name}")

        input_ids += self.sentence_end_ids
        loss_mask += [1] * len(self.sentence_end_ids)
        qa_mask += [0] * len(self.sentence_end_ids)

        assert len(input_ids) == len(loss_mask)

        if len(input_ids) <= self.seq_length:
            # features = self.padding(input_ids, loss_mask, qa_mask)
            features = {}
            features['input_ids'] = input_ids
            features['loss_mask'] = loss_mask
            features['qa_mask'] = qa_mask
        else:
            features = {}
            features['input_ids'] = input_ids[:self.seq_length - 1]
            features['loss_mask'] = loss_mask[:self.seq_length - 1]
            features['qa_mask'] = qa_mask[:self.seq_length - 1]

            features['input_ids'] += self.sentence_end_ids
            features['loss_mask'] += [1] * len(self.sentence_end_ids)
            features['qa_mask'] += [0] * len(self.sentence_end_ids)

        assert len(features['input_ids']) == len(features['loss_mask'])

        return features


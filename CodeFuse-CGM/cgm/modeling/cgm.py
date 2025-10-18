import torch
import torch.nn as nn

from data.preprocess import getJavaSentence, getPythonSentence, getSentence
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, BitsAndBytesConfig
from utils.common_utils import count_parameters, print_rank_0
import torch.nn.functional as F

from models.qwen2._4_46_1.modeling_qwen2 import Qwen2ForCausalLM
from models.qwen2._4_46_1.modeling_attn_mask_utils import AttentionMaskConverter

def graph2embedding(self, data, model, tokenizer, reponame, language, save_adj, peft, return_type=None):
    node_embeddings = {}
    node_id_to_index = {}
    index_counter = 0

    device = model.device

    for node in data['nodes']:
        nodeType = node['nodeType']

        if 'nodeId' in node.keys():
            node_id = node['nodeId']
        elif 'id' in node.keys():
            node_id = node['id']
        else:
            raise ValueError("No key named id/nodeId")

        sentence = getSentence(node, nodeType, reponame, 1024000)

        if sentence == "":
            node_embedding = torch.zeros((1, self.args.embedding_dim), dtype=torch.float32).to(device)
            node_embeddings[node_id] = [node_embedding]
            # sentence_dict[index_counter] = ""
            node_id_to_index[node_id] = [index_counter]
            index_counter += 1
        else:
            # 手动切词
            tokens = tokenizer.tokenize(sentence)
            num_tokens = len(tokens)
            num_segments = (num_tokens + 511) // 512  # Calculate number of segments
            embeddings = []
            # segments = []
            node_id_to_index[node_id] = list(range(index_counter, index_counter + num_segments))
            for i in range(num_segments):
                start = i * 512
                end = min((i + 1) * 512, num_tokens)
                segment_tokens = tokens[start:end]
                segment_ids = torch.tensor(tokenizer.convert_tokens_to_ids(segment_tokens), device=device).unsqueeze(0)

                if peft:
                    # return_type: ALL_256, ALL_768
                    segment_embedding = model.model(segment_ids, return_type=return_type)
                else:
                    segment_embedding = model(segment_ids)
                embeddings.append(segment_embedding)
                index_counter += 1

            node_embeddings[node_id] = embeddings

    num_nodes = index_counter

    # TODO: add sparse adj
    if save_adj:
        adj_matrix = torch.zeros((num_nodes, num_nodes)).to(device)

        for edge in data['edges']:
            source_id = edge['source']
            target_id = edge['target']
            source_indices = node_id_to_index.get(source_id)
            target_indices = node_id_to_index.get(target_id)
            if source_indices is None or target_indices is None:
                # if source_indices is None:
                #     print(f"{source_id} not exists")
                # if target_indices is None:
                #     print(f"{target_id} not exists")
                continue

            for source_index in source_indices:
                for target_index in target_indices:
                    adj_matrix[source_index, target_index] = 1

        # Connect embeddings of the same node
        for node_id, indices in node_id_to_index.items():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    adj_matrix[indices[i], indices[j]] = 1
                    adj_matrix[indices[j], indices[i]] = 1
    else:
        adj_matrix = None

    all_embeddings = []
    for value in node_embeddings.values():
        if isinstance(value, torch.Tensor):
            all_embeddings.append(value)
        elif isinstance(value, list):
            for tensor in value:
                all_embeddings.append(tensor)

    embeddings = torch.stack(all_embeddings, dim=0).squeeze(1)

    # embeddings = torch.stack(list(node_embeddings.values()))
    # embeddings = torch.stack(sum(node_embeddings.values(), []))
    # embeddings = torch.cat(list(node_embeddings.values()), dim=0)

    return embeddings, adj_matrix  # sentence_dict

class adapter(nn.Module):
    def __init__(self, args):
        super(adapter, self).__init__()
        self.fc1 = nn.Linear(args.embedding_dim, args.adapter_hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(args.adapter_hidden_dim, args.lm_hidden_dim)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class CGM(nn.Module):
    def __init__(self, args):
        super(CGM, self).__init__()
        # text encoder
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_encoder_path, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(
            args.pretrained_encoder_path,
            torch_dtype="auto",
            trust_remote_code=True
        )

        if args.self_defined:
            if args.quantization == "8bit":
                self.lm = Qwen2ForCausalLM.from_pretrained(
                    args.pretrained_model_path,
                    attn_implementation=args.attn_implementation,
                    torch_dtype="auto",
                    trust_remote_code=False,
                    quantization_config=(
                        BitsAndBytesConfig(
                            load_in_8bit=(args.quantization == "8bit"),
                            bnb_8bit_compute_dtype=torch.float8,
                            bnb_8bit_use_double_quant=True,
                            bnb_8bit_quant_type="fp8",
                            bnb_8bit_quant_storage=torch.float8,
                        )
                        if args.quantization == "8bit"
                        else None
                    ),
                )
            elif args.quantization == "4bit":
                self.lm = Qwen2ForCausalLM.from_pretrained(
                    args.pretrained_model_path,
                    attn_implementation=args.attn_implementation,
                    torch_dtype="auto",
                    trust_remote_code=False,
                    quantization_config=(
                        BitsAndBytesConfig(
                            load_in_4bit=(args.quantization == "4bit"),
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_quant_storage=torch.bfloat16,
                        )
                        if args.quantization == "4bit"
                        else None
                    ),
                )
            elif not args.quantization:
                self.lm = Qwen2ForCausalLM.from_pretrained(
                    args.pretrained_model_path,
                    attn_implementation=args.attn_implementation,
                    torch_dtype="auto",
                    trust_remote_code=False,
                )
            else:
                raise NotImplementedError(f"unrecognized args.qunatization: {args.quantization}")
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(
                args.pretrained_model_path,
                attn_implementation=args.attn_implementation,
                torch_dtype="auto",
                trust_remote_code=True,
                quantization_config=(
                    BitsAndBytesConfig(
                        load_in_4bit=(args.quantization == "4bit"),
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_quant_storage=torch.bfloat16,
                    )
                    if args.quantization == "4bit"
                    else None
                ),
            )

        args.lm_hidden_dim = self.lm.config.hidden_size
        self.args = args
        self.adapter = adapter(args)
        if args.load_pretrained_adapter:
            self.adapter.load_state_dict(torch.load(args.pretrained_adapter_path))
            print_rank_0(f"Adapter loaded from {args.pretrained_adapter_path}")
        else:
            print_rank_0("Adapter initialized")
        print_rank_0(f"Parameters of Encoder: {count_parameters(self.encoder) / 1e6:.1f}M")
        print_rank_0(f"Parameters of Adapter: {count_parameters(self.adapter) / 1e6:.1f}M")
        print_rank_0(f"Parameters of LLM: {count_parameters(self.lm) / 1e9:.2f}B")

    def graph2embedding(self, data, reponame, return_type=None):
        node_embeddings = {}
        node_id_to_index = {}
        index_counter = 0

        model = self.encoder
        tokenizer = self.encoder_tokenizer
        save_adj = self.args.use_adj,
        peft = self.args.peft

        device = model.device

        for node in data['nodes']:
            nodeType = node['nodeType']

            if 'nodeId' in node.keys():
                node_id = node['nodeId']
            elif 'id' in node.keys():
                node_id = node['id']
            else:
                raise ValueError("No key named id/nodeId")

            sentence = getSentence(node, nodeType, reponame, 1024000)

            if sentence == "":
                node_embedding = torch.zeros((1, self.args.embedding_dim), dtype=torch.float32).to(device)
                node_embeddings[node_id] = [node_embedding]
                node_id_to_index[node_id] = [index_counter]
                index_counter += 1
            else:
                tokens = tokenizer.tokenize(sentence)
                num_tokens = len(tokens)
                num_segments = (num_tokens + 511) // 512  # Calculate number of segments
                embeddings = []
                node_id_to_index[node_id] = list(range(index_counter, index_counter + num_segments))
                for i in range(num_segments):
                    start = i * 512
                    end = min((i + 1) * 512, num_tokens)
                    segment_tokens = tokens[start:end]
                    segment_ids = torch.tensor(tokenizer.convert_tokens_to_ids(segment_tokens),
                                               device=device).unsqueeze(0)

                    if peft:
                        # return_type: ALL_256, ALL_768
                        segment_embedding = model.model(segment_ids, return_type=return_type)
                    else:
                        segment_embedding = model(segment_ids)
                    embeddings.append(segment_embedding)
                    index_counter += 1

                node_embeddings[node_id] = embeddings

        num_nodes = index_counter

        # TODO: add sparse adj
        if save_adj:
            adj_matrix = torch.zeros((num_nodes, num_nodes)).to(device)

            for edge in data['edges']:
                source_id = edge['source']
                target_id = edge['target']
                source_indices = node_id_to_index.get(source_id)
                target_indices = node_id_to_index.get(target_id)
                if source_indices is None or target_indices is None:
                    continue

                for source_index in source_indices:
                    for target_index in target_indices:
                        adj_matrix[source_index, target_index] = 1

            # Connect embeddings of the same node
            for node_id, indices in node_id_to_index.items():
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        adj_matrix[indices[i], indices[j]] = 1
                        adj_matrix[indices[j], indices[i]] = 1
        else:
            adj_matrix = None

        all_embeddings = []
        for value in node_embeddings.values():
            if isinstance(value, torch.Tensor):
                all_embeddings.append(value)
            elif isinstance(value, list):
                for tensor in value:
                    all_embeddings.append(tensor)

        embeddings = torch.stack(all_embeddings, dim=0).squeeze(1)

        return embeddings, adj_matrix  # sentence_dict

    def forward(self, graph, qa_ids, qa_mask):
        graph_embeddings, adj_matrix = graph2embedding(
            data=graph,
            reponame=graph['reponame'],
            return_type="ALL_256",
        )

        embeddings = self.adapter(graph_embeddings)

        if self.args.peft:
            inputs_embeds = self.lm.model.model.embed_tokens(qa_ids)
        else:
            inputs_embeds = self.lm.model.embed_tokens(qa_ids)

        input_embeddings = torch.cat((embeddings, inputs_embeds), dim=-2)
        input_embeddings = input_embeddings.unsqueeze(0)

        if adj_matrix is not None and self.args.use_adj:

            if len(adj_matrix.shape) == 2:
                adj_matrix = adj_matrix.unsqueeze(0)
            batch_size, seq_len_x, _ = adj_matrix.shape

            seq_len_q = inputs_embeds.size(-2)

            qa_matrix = torch.ones(batch_size, seq_len_q, seq_len_q, device=qa_mask.device)
            qa_matrix = torch.tril(qa_matrix)

            matrix_xq = qa_mask.unsqueeze(1) * torch.ones(batch_size, seq_len_x, seq_len_q, device=qa_mask.device)

            matrix_qx = torch.ones(batch_size, seq_len_q, seq_len_x, device=qa_mask.device)

            # Construct the full attention mask
            attention_mask = torch.cat([
                torch.cat([adj_matrix, matrix_xq], dim=2),  # x_embeddings part
                torch.cat([matrix_qx, qa_matrix], dim=2)  # q_embeddings part
            ], dim=1).squeeze(1)

            outputs = self.lm(inputs_embeds=input_embeddings,
                              attention_mask=attention_mask,
                              return_dict=True)

        else:
            outputs = self.lm(inputs_embeds=input_embeddings,
                              return_dict=True)

        return outputs


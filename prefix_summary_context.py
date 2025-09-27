from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


class NewsDS(Dataset):
    """
    Inputs:
      - df (pd.DataFrame): DataFrame with columns
      - 'Text': main article text
      - f'{text_hist_col}_{i}': the i-th historical context article text
      - f'date_diff_lag_{i}': time difference (in days) from the main article for the i-th historical article

    Parameters (params dict):
      - num_text_hist (int): N, number of historical context articles
      - text_hist_col (str): prefix for historical text columns
      - max_length (int): main article tokenization length
      - max_length_hist (int): historical article tokenization length
      - hist_time_units (int): divisor for time
    """

    def __init__(self, df, tokenizer, hist_tokenizer, params: Optional[Dict] = None):
        if params is None:
            params = {}
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.hist_tokenizer = hist_tokenizer
        self.params = params

        self.text_col = 'Text'
        self.num_hist = self.params.get('num_text_hist', 5)
        self.text_hist_col = self.params.get('text_hist_col', 'TextHist')
        self.label_col = self.params.get('label_col', 'labels')

        self.max_length = int(self.params.get('max_length', 4096))
        self.max_length_hist = int(self.params.get('max_length_hist', 512))
        self.hist_time_units = int(self.params.get('hist_time_units', 5))
        self.inputs = defaultdict(list)
        batch_size = int(self.params.get('tokenize_batch_size', 4096))

        for start in range(0, len(self.df), batch_size):
            stop = min(start + batch_size, len(self.df))
            main_batch = self.df[self.text_col].iloc[start:stop].tolist()

            d_main = self.tokenizer(
                main_batch,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            self.inputs['input_ids'].append(d_main['input_ids'].to(torch.long))
            self.inputs['attention_mask'].append(d_main['attention_mask'].to(torch.long))
            self.inputs['labels'].append(main_batch['labels'].to(torch.long))

            # Historical context per i = 1..N
            for i in range(1, self.num_hist + 1):
                hist_batch = self.df[f'{self.text_hist_col}_{i}'].iloc[start:stop].tolist()
                d_hist = self.hist_tokenizer(
                    hist_batch,
                    max_length=self.max_length_hist,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                self.inputs[f'input_ids_hist_{i}'].append(d_hist['input_ids'].to(torch.long))
                self.inputs[f'attention_mask_hist_{i}'].append(d_hist['attention_mask'].to(torch.long))

                hist_time = torch.as_tensor(
                    self.df[f'date_diff_lag_{i}'].iloc[start:stop].astype(int).values,
                    dtype=torch.long
                )
                hist_time = hist_time // self.hist_time_units
                neg_mask = hist_time < 0
                if neg_mask.any():
                    max_pos = hist_time[~neg_mask].max().item() if (~neg_mask).any() else 0
                    hist_time[neg_mask] = max_pos + 1
                self.inputs[f'hist_time_{i}'].append(hist_time)

        for k, v in list(self.inputs.items()):
            self.inputs[k] = torch.cat(v, dim=0)

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch = {
            'input_ids': self.inputs['input_ids'][idx].type(torch.LongTensor),
            'attention_mask': self.inputs['attention_mask'][idx].type(torch.LongTensor),
            'labels': self.inputs['labels'][idx].type(torch.LongTensor),
        }
        for i in range(1, self.num_hist + 1):
            batch[f'input_ids_hist_{i}'] = self.inputs[f'input_ids_hist_{i}'][idx].type(torch.LongTensor)
            batch[f'attention_mask_hist_{i}'] = self.inputs[f'attention_mask_hist_{i}'][idx].type(torch.LongTensor)
            batch[f'hist_time_{i}'] = self.inputs[f'hist_time_{i}'][idx].type(torch.LongTensor)
        return batch


def init_summary_tokens(context_encoder: nn.Module, num_summary_tokens: int) -> nn.Parameter:
    """
    Initialize summary tokens by fitting a MVG to the token embedding distribution
    of the context encoder. Revert to fitted mean + random noise if token embedding matrix is not PSD.
    """
    old_embeddings = context_encoder.get_input_embeddings()
    old_w = old_embeddings.weight.detach().to(torch.float32)
    mean = old_w.mean(dim=0)
    centered = old_w - mean
    cov = (centered.T @ centered) / old_w.shape[0]
    cov = cov + 1e-5 * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)

    try:
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        samples = dist.sample((num_summary_tokens,))
        new_w = samples.to(old_embeddings.weight.dtype)
    except Exception:
        noise = torch.randn(num_summary_tokens, old_w.shape[1], dtype=old_embeddings.weight.dtype) * 0.02
        new_w = mean.to(old_embeddings.weight.dtype).unsqueeze(0).expand_as(noise) + noise

    return nn.Parameter(new_w)


class Classifier(torch.nn.Module):
    """
    Simple classification head for prediction from pooled hidden states
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = torch.nn.Linear(self.config.class_input, self.config.class_input // 2)
        self.dropout = torch.nn.Dropout(0.10)
        self.activation = torch.nn.ReLU()
        self.out_proj = torch.nn.Linear(self.config.class_input // 2, self.config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class PrefixSummaryContext(nn.Module):
    """
    PSC with:
      - HCS (Historical Context Summarizer): a small encoder (DeBERTa-base)
      - CMA (Cross-Model Alignment): MHA with queries from HCS summaries and keys/values from the large LM's vocab embeddings
      - CALM: if labels provided (or inferred), computes masked next-token loss on main article tokens only

      - Insert M summary tokens per historical article at regular intervals (per-article L)
      - Add time embeddings to each historical article's M summary tokens
      - Set all prefix (PSC) position_ids = 0; main article starts at 1
      - Freeze large LM during CALM pretraining
    """

    def __init__(
        self,
        num_historical_contexts: int = 5,
        large_lm_name: str = "mistralai/Mistral-7B-v0.2",
        context_encoder_name: str = "microsoft/deberta-v3-base",
        num_summary_tokens_per_article: int = 5,
        num_alignment_heads: int = 8,
        max_time_buckets: int = 50,
        init_summary_from_vocab: bool = True,
        mode: str = "CALM",
        lora_rank: int = 16,
        num_labels: int = 2,
    ):
        super().__init__()
        self.num_historical_contexts = int(num_historical_contexts)
        self.num_summary_tokens = int(num_summary_tokens_per_article)
        self.num_alignment_heads = int(num_alignment_heads)

        self.context_encoder = AutoModel.from_pretrained(context_encoder_name)
        self.large_lm = AutoModel.from_pretrained(large_lm_name)

        self.d_ce = self.context_encoder.config.hidden_size
        self.d_llm = self.large_lm.config.hidden_size
        self.vocab_size = self.large_lm.config.vocab_size
        self.mode = mode
        self.lora_rank = lora_rank

        if init_summary_from_vocab:
            self.summary_tokens = init_summary_tokens(self.context_encoder, self.num_summary_tokens)
        else:
            self.summary_tokens = nn.Parameter(torch.randn(self.num_summary_tokens, self.d_ce))

        self.time_embeddings = nn.Embedding(int(max_time_buckets), self.d_ce)

        self.alignment_query_projs = nn.ModuleList(
            [nn.Linear(self.d_ce, self.d_llm) for _ in range(self.num_alignment_heads)]
        )

        self.config = self.large_lm.config
        self.config.class_input = self.config.hidden_size
        self.config.num_labels = num_labels
        self.classifier = Classifier(self.config)

        if self.mode == 'CALM':
            print('>> Training with CALM Objective, Freezing Large LM Parameters')
            self.freeze_large_lm()
            self.lm_head = nn.Linear(self.d_llm, self.vocab_size, bias=False)
            self.init_lm_head_from_pretrained(large_lm_name)
        elif self.mode == 'SFT':
            print('>> Supervised Finetuning, Inserting PEFT Adapters into Large LM Parameters')
            self.enable_peft_for_sft(self.lora_rank)

    @property
    def device(self):
        return next(self.parameters()).device

    def freeze_large_lm(self) -> None:
        for p in self.large_lm.parameters():
            p.requires_grad = False

    def init_lm_head_from_pretrained(self, large_lm_name: str) -> None:
        ref = AutoModelForCausalLM.from_pretrained(large_lm_name)
        with torch.no_grad():
            self.lm_head.weight.copy_(ref.lm_head.weight.data)
        for p in self.lm_head.parameters():
            p.requires_grad = True
        del ref


    def enable_peft_for_sft(
        self,
        r: int = 16,
        alpha: Optional[int] = None,
        dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
    ) -> None:

        if alpha is None:
            alpha = 2 * r
        if target_modules is None:
            target_modules = 'all-linear'

        peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias=bias,
            task_type=TaskType.CAUSAL_LM,
        )
        self.large_lm = get_peft_model(self.large_lm, peft_config)


    def token_pooling(self, h, attention_mask):

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(h.size()).float()
        h = torch.sum(h * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return h

    def create_summary_embeddings_one(
        self,
        input_ids: torch.Tensor,          # [L]
        attention_mask: torch.Tensor      # [L]
    ) -> torch.Tensor:
        """
        Insert M summary tokens after every floor(L/M) valid tokens:
        More specifically, the first token is placed at the beginning (pos 0), the last token is
        placed at the end (pos L-1), and the remaining M-2 tokens are distributed evenly in between
        run the context encoder on inputs_embeds, and return the hidden states at those M positions.
        Returns: [M, d_ce]
        """
        assert input_ids.dim() == 1
        assert attention_mask.dim() == 1
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        L = int(attention_mask.sum().item())
        L = max(L, 1)

        positions = [0]
        if self.num_summary_tokens >= 2:
            num_interleaved = self.num_summary_tokens - 2
            if num_interleaved > 0:
                step = max(1, L // (num_interleaved + 1))
                for k in range(1, num_interleaved + 1):
                    pos = min(k * step, L - 1)
                    positions.append(pos)
            positions.append(L - 1)

        token_embeds = self.context_encoder.embeddings(input_ids.unsqueeze(0)).squeeze(0)  # [Lmax, d_ce]
        pieces = []
        attn_pieces = []
        summary_mask = []

        last = 0
        for j, p in enumerate(positions):
            if p > last:
                pieces.append(token_embeds[last:p])
                attn_pieces.append(attention_mask[last:p])
                summary_mask.append(torch.zeros(p - last, dtype=torch.bool, device=self.device))
            pieces.append(self.summary_tokens[j:j+1])  # [1, d_ce]
            attn_pieces.append(torch.ones(1, dtype=torch.long, device=self.device))
            summary_mask.append(torch.ones(1, dtype=torch.bool, device=self.device))
            last = p
        if last < L:
            pieces.append(token_embeds[last:L])
            attn_pieces.append(attention_mask[last:L])
            summary_mask.append(torch.zeros(L - last, dtype=torch.bool, device=self.device))
        if L < input_ids.numel():
            pieces.append(token_embeds[L:])
            attn_pieces.append(attention_mask[L:])
            summary_mask.append(torch.zeros(input_ids.numel() - L, dtype=torch.bool, device=self.device))

        aug_embeds = torch.cat(pieces, dim=0)   # [Laug, d_ce]
        aug_mask = torch.cat(attn_pieces, dim=0)    # [Laug]
        sum_mask = torch.cat(summary_mask, dim=0)   # [Laug]

        enc_out = self.context_encoder(inputs_embeds=aug_embeds.unsqueeze(0), attention_mask=aug_mask.unsqueeze(0))
        hs = enc_out.last_hidden_state.squeeze(0)   # [Laug, d_ce]

        # Extract summary token hidden states in order
        se = hs[sum_mask]   # [M, d_ce]
        assert se.shape[0] == self.num_summary_tokens, \
            f"Expected {self.num_summary_tokens} summary tokens, got {se.shape[0]}"
        return se


    def create_summary_embeddings(
        self,
        article_input_ids: torch.Tensor,    # [B, L]
        article_attention_mask: torch.Tensor    # [B, L]
    ) -> torch.Tensor:

        B = article_input_ids.size(0)
        out = []
        for b in range(B):
            se = self.create_summary_embeddings_one(article_input_ids[b], article_attention_mask[b])
            out.append(se.unsqueeze(0))
        return torch.cat(out, dim=0)  # [B, M, d_ce]


    def process_historical_contexts(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Construct PSC = concatenation over N historical articles of
        (summary embeddings + time embeddings), shape [B, N*M, d_ce].
        """
        all_summary_embeddings = []
        for i in range(1, self.num_historical_contexts + 1):
            hist_ids = batch[f'input_ids_hist_{i}']
            hist_mask = batch[f'attention_mask_hist_{i}']
            hist_time = batch[f'hist_time_{i}']  # [B]

            se = self.create_summary_embeddings(hist_ids, hist_mask)  # [B, M, d_ce]
            max_idx = self.time_embeddings.num_embeddings - 1
            hist_time = torch.clamp(hist_time, 0, max_idx)
            te = self.time_embeddings(hist_time).unsqueeze(1).expand(-1, self.num_summary_tokens, -1)  # [B, M, d_ce]
            all_summary_embeddings.append(se + te)

        # Concatenate historical articles in temporal order: [B, N*M, d_ce]
        summary_context = torch.cat(all_summary_embeddings, dim=1)
        return summary_context


    def cross_model_alignment(self, summary_context: torch.Tensor) -> torch.Tensor:
        """
        summary_context: [B, P, d_ce]  (P = N * M)
        Returns aligned prefix in LLM space: [B, P, d_llm]
        Use attention heads with separate W^Q (per head), fixed K=V=E_vocab
        Attention head outputs are averaged across query heads after softmax to enforce convex span
        """
        B, P, _ = summary_context.shape
        E = self.large_lm.get_input_embeddings().weight    # [V, d_llm]
        aligned_prefix = torch.zeros(B, P, self.d_llm, device=summary_context.device, dtype=summary_context.dtype)
        scale = 1.0 / math.sqrt(self.d_llm)

        for h, proj in enumerate(self.alignment_query_projs):
            Q = proj(summary_context.to(dtype=E.dtype))   # [B, P, d_llm]
            logits = torch.matmul(Q, E.transpose(0, 1)) * scale # [B, P, V]
            attn = torch.softmax(logits, dim=-1)    # [B, P, V]
            aligned_prefix += torch.matmul(attn, E) # [B, P, d_llm]

        aligned_prefix = aligned_prefix / float(self.num_alignment_heads)   # [B, P, d_llm]
        return aligned_prefix


    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        main_ids = batch['input_ids']   # [B, Lm]
        main_mask = batch['attention_mask'] # [B, Lm]
        B, Lm = main_ids.shape

        # 1) Construct PSC prefix in CE space, add temporal info, then align to LLM space
        summary_context = self.process_historical_contexts(batch)   # [B, N*M, d_ce]
        prefix_embeddings = self.cross_model_alignment(summary_context) # [B, N*M, d_llm]
        P = prefix_embeddings.size(1)

        # 2) Main article token embeddings
        main_embeds = self.large_lm.get_input_embeddings()(main_ids)    # [B, Lm, d_llm]

        if prefix_embeddings.dtype != main_embeds.dtype:
            prefix_embeddings = prefix_embeddings.to(dtype=main_embeds.dtype)

        # 3) Concatenate PSC to main article embeddings
        inputs_embeds = torch.cat([prefix_embeddings, main_embeds], dim=1)  # [B, P+Lm, d_llm]

        # 4) Position IDs: prefix all zeros, main starts at 1
        pos_prefix = torch.zeros(B, P, dtype=torch.long, device=self.device)
        pos_main = torch.arange(1, Lm + 1, device=self.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        position_ids = torch.cat([pos_prefix, pos_main], dim=1) # [B, P+Lm]

        # 5) Attention mask: prefix all ones, main uses its mask
        mask_prefix = torch.ones(B, P, dtype=main_mask.dtype, device=self.device)
        attention_mask = torch.cat([mask_prefix, main_mask], dim=1) # [B, P+Lm]

        # 5) Run large LLM for hidden states
        out = self.large_lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=False,
            return_dict=True
        )
        hidden = out.last_hidden_state  # [B, P+Lm, d_llm]

        if self.mode == "CALM":
            token_logits = self.lm_head(hidden) # [B, P+Lm, V]
            result = {"logits": token_logits, "prefix_len": P}

            if labels is None:
                labels = main_ids

            # Shift to predict token i from logits at position i-1 (main tokens only)
            shift_logits = token_logits[:, P:-1, :] # [B, Lm-1, V]
            shift_labels = labels[:, 1:]    # [B, Lm-1]
            shift_mask = main_mask[:, 1:].to(shift_logits.dtype)    # [B, Lm-1]

            loss_per_token = F.cross_entropy(
                shift_logits.reshape(-1, self.vocab_size),
                shift_labels.reshape(-1),
                reduction='none')

            masked = loss_per_token * shift_mask.reshape(-1)
            denom = shift_mask.sum().clamp(min=1.0)
            loss = masked.sum() / denom
            result["loss"] = loss
            return result

        elif self.mode == "SFT":
            main_hidden = hidden[:, P:, :]  # [B, Lm, d_llm]
            pooled = self.token_pooling(main_hidden, main_mask) # [B, d_llm]
            logits = self.classifier(pooled)    # [B, num_labels]
            result = {"logits": logits}

            if labels is not None:
                loss = F.cross_entropy(logits, labels.long())
                result["loss"] = loss

            return result

        else:
            raise ValueError(f"Unknown training_mode '{self.mode}'. Use 'CALM' or 'SFT'.")


class CALM:
    """
      - Freeze the large LM
      - Train only the HCS + CMA + time embeddings + summary tokens
    """

    def __init__(self, model: PrefixSummaryContext, learning_rate: float = 3e-5):
        self.model = model
        trainable_params = []
        for name, p in model.named_parameters():
            if any(s in name for s in ["context_encoder", "alignment_", "summary_tokens", "time_embeddings"]):
                p.requires_grad = True
                trainable_params.append(p)
        for p in model.lm_head.parameters():
            p.requires_grad = True
            trainable_params.append(p)
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)


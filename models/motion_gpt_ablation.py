"""
Non-LLM backbone ablation (Option A): from-scratch causal GPT on compact
motion/relation/audio vocab — no text BPE, no pretrained LLM, no LoRA.

Activate with: --llm-backbone gpt_ablation
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from transformers import GPT2Config, GPT2LMHeadModel

# Tasks that need only discrete multimodal tokens (no captions / MotionScript).
GPT_ABLATION_TASKS = [
    "leader_rel_to_follower",
    "follower_rel_to_leader",
    "leader_to_follower",
    "follower_to_leader",
    "pair_to_relationship",
]

GPT_ABLATION_BACKBONE_ID = "gpt_ablation"


def is_gpt_ablation_backbone(name: Optional[str]) -> bool:
    if name is None:
        return False
    return str(name).strip().lower() in {GPT_ABLATION_BACKBONE_ID, "gpt-ablation", "nonllm_gpt"}


class MotionAblationVocab:
    """Compact id space: specials + delimiters + IH + Rel + optional Audio codes."""

    def __init__(
        self,
        nb_ih_code: int = 512,
        nb_rel_code: int = 512,
        include_audio: bool = True,
        nb_audio_code: int = 4096,
    ):
        self.nb_ih_code = int(nb_ih_code)
        self.nb_rel_code = int(nb_rel_code)
        self.include_audio = bool(include_audio)
        self.nb_audio_code = int(nb_audio_code)

        tokens: List[str] = [
            "<bos>",
            "<eos>",
            "<pad>",
            "<LeaderMotion>",
            "</LeaderMotion>",
            "<FollowerMotion>",
            "</FollowerMotion>",
            "<Relationship>",
            "</Relationship>",
        ]
        if self.include_audio:
            tokens += ["<AudioTokens>", "</AudioTokens>"]

        tokens += [f"<IH_{i}>" for i in range(self.nb_ih_code)]
        tokens += [f"<Rel_{i}>" for i in range(self.nb_rel_code)]
        if self.include_audio:
            tokens += [f"<Audio_{i}>" for i in range(self.nb_audio_code)]

        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.bos_token_id = self.token_to_id["<bos>"]
        self.eos_token_id = self.token_to_id["<eos>"]
        self.pad_token_id = self.token_to_id["<pad>"]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        return [self.id_to_token.get(int(i), "<unk>") for i in ids]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        toks = self.convert_ids_to_tokens(ids)
        if skip_special_tokens:
            skip = {"<bos>", "<eos>", "<pad>"}
            toks = [t for t in toks if t not in skip]
        return " ".join(toks)

    def get_config(self) -> Dict[str, Any]:
        return {
            "nb_ih_code": self.nb_ih_code,
            "nb_rel_code": self.nb_rel_code,
            "include_audio": self.include_audio,
            "nb_audio_code": self.nb_audio_code,
            "vocab_size": len(self),
        }

    def tid(self, name: str) -> int:
        return self.token_to_id[name]

    def ih_id(self, code: int) -> int:
        code = int(code)
        if code < 0 or code >= self.nb_ih_code:
            raise ValueError(f"IH code {code} out of range [0, {self.nb_ih_code})")
        return self.token_to_id[f"<IH_{code}>"]

    def rel_id(self, code: int) -> int:
        code = int(code)
        if code < 0 or code >= self.nb_rel_code:
            raise ValueError(f"Rel code {code} out of range [0, {self.nb_rel_code})")
        return self.token_to_id[f"<Rel_{code}>"]

    def audio_id(self, code: int) -> int:
        if not self.include_audio:
            raise RuntimeError("Audio tokens requested but include_audio=False")
        code = int(code)
        if code < 0 or code >= self.nb_audio_code:
            raise ValueError(f"Audio code {code} out of range [0, {self.nb_audio_code})")
        return self.token_to_id[f"<Audio_{code}>"]


def _as_int_list(x) -> List[int]:
    if x is None:
        return []
    if torch.is_tensor(x):
        return x.detach().cpu().ravel().tolist()
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            if torch.is_tensor(v):
                out.extend(v.detach().cpu().ravel().tolist())
            else:
                out.append(int(v))
        return out
    return [int(x)]


def build_prompt_target_ids_gpt_ablation(
    vocab: MotionAblationVocab,
    task: str,
    leader_tokens,
    follower_tokens,
    relationship_tokens,
    audio_tokens=None,
    include_audio: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    Text-free packing. Prompt positions get label -100; target multimodal tokens supervised.
    Layout mirrors Stage-II multimodal order without English / chat template.
    """
    if task not in GPT_ABLATION_TASKS:
        raise ValueError(f"gpt_ablation task must be one of {GPT_ABLATION_TASKS}, got {task!r}")

    L = _as_int_list(leader_tokens)
    F = _as_int_list(follower_tokens)
    R = _as_int_list(relationship_tokens)
    A = _as_int_list(audio_tokens) if (include_audio and vocab.include_audio) else []
    use_audio = len(A) > 0

    def blk_audio() -> List[int]:
        if not use_audio:
            return []
        return (
            [vocab.tid("<AudioTokens>")]
            + [vocab.audio_id(a) for a in A]
            + [vocab.tid("</AudioTokens>")]
        )

    def blk_leader() -> List[int]:
        return (
            [vocab.tid("<LeaderMotion>")]
            + [vocab.ih_id(t) for t in L]
            + [vocab.tid("</LeaderMotion>")]
        )

    def blk_follower() -> List[int]:
        return (
            [vocab.tid("<FollowerMotion>")]
            + [vocab.ih_id(t) for t in F]
            + [vocab.tid("</FollowerMotion>")]
        )

    def blk_rel() -> List[int]:
        return (
            [vocab.tid("<Relationship>")]
            + [vocab.rel_id(t) for t in R]
            + [vocab.tid("</Relationship>")]
        )

    if task == "leader_rel_to_follower":
        prompt_body = blk_audio() + blk_leader() + blk_rel()
        open_id = vocab.tid("<FollowerMotion>")
        target_payload = [vocab.ih_id(t) for t in F] + [vocab.tid("</FollowerMotion>")]
    elif task == "follower_rel_to_leader":
        prompt_body = blk_audio() + blk_follower() + blk_rel()
        open_id = vocab.tid("<LeaderMotion>")
        target_payload = [vocab.ih_id(t) for t in L] + [vocab.tid("</LeaderMotion>")]
    elif task == "leader_to_follower":
        prompt_body = blk_audio() + blk_leader()
        open_id = vocab.tid("<FollowerMotion>")
        target_payload = [vocab.ih_id(t) for t in F] + [vocab.tid("</FollowerMotion>")]
    elif task == "follower_to_leader":
        prompt_body = blk_audio() + blk_follower()
        open_id = vocab.tid("<LeaderMotion>")
        target_payload = [vocab.ih_id(t) for t in L] + [vocab.tid("</LeaderMotion>")]
    elif task == "pair_to_relationship":
        prompt_body = blk_audio() + blk_leader() + blk_follower()
        open_id = vocab.tid("<Relationship>")
        target_payload = [vocab.rel_id(t) for t in R] + [vocab.tid("</Relationship>")]
    else:
        raise ValueError(f"Unhandled gpt_ablation task: {task}")

    input_ids = [vocab.bos_token_id] + prompt_body + [open_id] + target_payload + [vocab.eos_token_id]
    n_prompt = 1 + len(prompt_body) + 1  # bos + body + open delimiter
    target_ids = [-100] * n_prompt + target_payload + [vocab.eos_token_id]
    assert len(input_ids) == len(target_ids)
    return input_ids, target_ids


def build_inference_prompt_ids_gpt_ablation(
    vocab: MotionAblationVocab,
    task: str,
    leader_tokens,
    follower_tokens,
    relationship_tokens,
    audio_tokens=None,
    include_audio: bool = False,
) -> torch.Tensor:
    """Prompt-only ids (through open delimiter) for AR generation."""
    input_ids, target_ids = build_prompt_target_ids_gpt_ablation(
        vocab=vocab,
        task=task,
        leader_tokens=leader_tokens,
        follower_tokens=follower_tokens,
        relationship_tokens=relationship_tokens,
        audio_tokens=audio_tokens,
        include_audio=include_audio,
    )
    # First supervised index is where target_ids != -100
    n_prompt = next(i for i, t in enumerate(target_ids) if t != -100)
    return torch.tensor(input_ids[:n_prompt], dtype=torch.long)


def process_batch_gpt_ablation(
    vocab: MotionAblationVocab,
    batch_aux_info,
    batch_audio_tokens,
    batch_interhuman_data,
    max_tgt_len: int,
    current_batch_task: Optional[str] = None,
    include_audio: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch packer analogous to process_batch_Salsa, but text-free."""
    if batch_interhuman_data is None:
        raise ValueError("gpt_ablation requires batch_interhuman_data")

    n_samples = len(batch_aux_info) if hasattr(batch_aux_info, "__len__") else 1
    if current_batch_task in (None, "none", "all"):
        task = None
    else:
        task = current_batch_task
        if task not in GPT_ABLATION_TASKS:
            raise ValueError(
                f"gpt_ablation does not support task={task!r}. "
                f"Use one of {GPT_ABLATION_TASKS} (or none/all for random among them)."
            )

    batch_input_ids, batch_target_ids = [], []
    for i in range(n_samples):
        if isinstance(batch_interhuman_data, (list, tuple)):
            ih = batch_interhuman_data[i]
        else:
            ih = batch_interhuman_data
        if ih is None:
            raise ValueError(f"gpt_ablation sample {i} missing interhuman data")

        if isinstance(batch_audio_tokens, (list, tuple)):
            audio_tokens = batch_audio_tokens[i]
        else:
            audio_tokens = batch_audio_tokens
        sample_task = task if task is not None else random.choice(GPT_ABLATION_TASKS)
        one_in, one_tgt = build_prompt_target_ids_gpt_ablation(
            vocab=vocab,
            task=sample_task,
            leader_tokens=ih["leader_tokens"],
            follower_tokens=ih["follower_tokens"],
            relationship_tokens=ih["relationship_tokens"],
            audio_tokens=audio_tokens,
            include_audio=include_audio,
        )
        batch_input_ids.append(torch.LongTensor(one_in))
        batch_target_ids.append(torch.LongTensor(one_tgt))

    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=vocab.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(vocab.pad_token_id)
    return input_ids, target_ids, attention_mask.long()


class MotionGPTAblation(nn.Module):
    """
    Thin wrapper around randomly initialized GPT2LMHeadModel.
    Exposes HF-like forward(input_ids, attention_mask, labels) and generate().
    """

    def __init__(self, vocab: MotionAblationVocab, args):
        super().__init__()
        self.vocab = vocab
        n_layer = int(getattr(args, "gpt_n_layer", 12))
        n_embd = int(getattr(args, "gpt_n_embd", 768))
        n_head = int(getattr(args, "gpt_n_head", 12))
        n_positions = int(getattr(args, "gpt_n_positions", 1024))
        if n_embd % n_head != 0:
            raise ValueError(f"gpt_n_embd ({n_embd}) must be divisible by gpt_n_head ({n_head})")

        self.config = GPT2Config(
            vocab_size=len(vocab),
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            bos_token_id=vocab.bos_token_id,
            eos_token_id=vocab.eos_token_id,
            pad_token_id=vocab.pad_token_id,
            resid_pdrop=float(getattr(args, "gpt_dropout", 0.1)),
            embd_pdrop=float(getattr(args, "gpt_dropout", 0.1)),
            attn_pdrop=float(getattr(args, "gpt_dropout", 0.1)),
        )
        # Random init — do NOT from_pretrained.
        self.transformer = GPT2LMHeadModel(self.config)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(self, *args, **kwargs):
        return self.transformer.generate(*args, **kwargs)

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    @property
    def lm_head(self):
        return self.transformer.lm_head

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        pct = 100.0 * trainable / max(total, 1)
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.2f}")

    def gpt_hyperparams(self) -> Dict[str, Any]:
        return {
            "n_layer": self.config.n_layer,
            "n_embd": self.config.n_embd,
            "n_head": self.config.n_head,
            "n_positions": self.config.n_positions,
            "vocab_size": self.config.vocab_size,
        }


def build_motion_gpt_ablation(args) -> Tuple[MotionAblationVocab, MotionGPTAblation, int]:
    """Factory used by MotionLLM when --llm-backbone gpt_ablation."""
    include_audio = bool(getattr(args, "include_audio", False))
    vocab = MotionAblationVocab(
        nb_ih_code=getattr(args, "nb_ih_code", 512),
        nb_rel_code=getattr(args, "nb_rel_code", 512),
        include_audio=include_audio,
        nb_audio_code=int(getattr(args, "nb_audio_code", 4096)),
    )
    model = MotionGPTAblation(vocab, args)
    nb_text_tokens = 0  # no pretrained text table
    print(
        f"[gpt_ablation] vocab_size={len(vocab)} include_audio={include_audio} "
        f"config={model.gpt_hyperparams()}"
    )
    model.print_trainable_parameters()
    return vocab, model, nb_text_tokens

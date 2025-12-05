"""
ck_plug_integration.py
Lightweight CK-PLUG-style integration for RetrievalQualityAnalyzer.

This version does NOT depend on the CK-PLUG repo structure.
Instead, it implements the core idea directly on top of HuggingFace
`AutoModelForCausalLM` and `AutoTokenizer`:

- Two parallel forward passes:
    - parametric: prompt without context
    - rag:       prompt with retrieved context
- Compute confidence gain via entropy difference
- When confidence_gain < epsilon, mix parametric + contextual
  distributions using alpha.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class CKPlugConfig:
    alpha: float = 0.5      # knowledge reliance balance param [0, 1]
    epsilon: float = 0.0    # conflict threshold on confidence gain
    top_k: Optional[int] = 100
    max_new_tokens: int = 64


class CKPlugGenerator:
    """
    CK-PLUG style generator operating on an existing HF model+tokenizer.
    """

    def __init__(self, model, tokenizer, device: str = "cuda", config: CKPlugConfig = CKPlugConfig()):
        """
        Args:
            model: AutoModelForCausalLM (already on device)
            tokenizer: matching tokenizer
            device: "cuda" or "cpu"
            config: CKPlugConfig
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config

    @torch.no_grad()
    def generate_with_control(
        self,
        query: str,
        context: str,
        alpha: Optional[float] = None,
        epsilon: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate answer with CK-PLUG-style control.

        Args:
            query: Question text
            context: Retrieved context text
            alpha: Override for knowledge reliance parameter [0, 1]
            epsilon: Override for conflict detection threshold
            max_new_tokens: Override for max generated tokens

        Returns:
            Generated answer string.
        """
        alpha = self.config.alpha if alpha is None else alpha
        epsilon = self.config.epsilon if epsilon is None else epsilon
        max_new_tokens = self.config.max_new_tokens if max_new_tokens is None else max_new_tokens

        # Prompts
        prompt_param = f"Question: {query}\nAnswer:"
        prompt_rag = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

        # Tokenize both prompts
        inputs_param = self.tokenizer(prompt_param, return_tensors="pt").to(self.device)
        inputs_rag = self.tokenizer(prompt_rag, return_tensors="pt").to(self.device)

        input_ids_param = inputs_param["input_ids"]
        input_ids_rag = inputs_rag["input_ids"]

        # We'll keep separate caches for both branches
        past_param = None
        past_rag = None

        generated_ids = []

        for _ in range(max_new_tokens):
            # Forward pass for parametric branch
            out_param = self.model(
                input_ids=input_ids_param,
                past_key_values=past_param,
                use_cache=True,
            )
            logits_param = out_param.logits[:, -1, :]  # (1, vocab)
            past_param = out_param.past_key_values

            # Forward pass for RAG branch (with context)
            out_rag = self.model(
                input_ids=input_ids_rag,
                past_key_values=past_rag,
                use_cache=True,
            )
            logits_rag = out_rag.logits[:, -1, :]      # (1, vocab)
            past_rag = out_rag.past_key_values

            # Convert to probabilities
            probs_param = F.softmax(logits_param, dim=-1)
            probs_rag = F.softmax(logits_rag, dim=-1)

            # Entropy (negative sum p log p). We use ln; paper uses log2, but
            # the sign of confidence gain is invariant to base. :contentReference[oaicite:1]{index=1}
            def entropy(p):
                p_safe = p.clamp(min=1e-12)
                return -(p_safe * p_safe.log()).sum(dim=-1)

            H_param = entropy(probs_param)
            H_rag = entropy(probs_rag)

            # Confidence gain: how much entropy drops when adding context
            # CG > 0 → context increases confidence; CG < 0 → potential conflict.
            confidence_gain = H_param - H_rag

            # Decide whether to apply modulation
            if confidence_gain.item() < epsilon:
                # Conflict: mix parametric + context distributions
                mixed_probs = alpha * probs_param + (1.0 - alpha) * probs_rag
                mixed_probs = mixed_probs / mixed_probs.sum(dim=-1, keepdim=True)
                logits_mix = mixed_probs.log()
            else:
                # No conflict → use pure RAG distribution
                logits_mix = logits_rag

            # Optional top-k filtering
            if self.config.top_k is not None and self.config.top_k > 0:
                k = min(self.config.top_k, logits_mix.size(-1))
                values, indices = torch.topk(logits_mix, k)
                min_values = values[..., -1, None]
                filter_mask = logits_mix < min_values
                logits_mix = logits_mix.masked_fill(filter_mask, -float("inf"))

            # Greedy decoding
            next_token_id = torch.argmax(logits_mix, dim=-1, keepdim=True)  # (1, 1)

            token_id = next_token_id.item()
            if self.tokenizer.eos_token_id is not None and token_id == self.tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)

            # Feed the same generated token to both branches
            input_ids_param = next_token_id
            input_ids_rag = next_token_id

        if not generated_ids:
            return ""

        gen_tensor = torch.tensor([generated_ids], dtype=torch.long, device=self.device)

        # Decode ONLY the generated portion (after the prompt)
        text = self.tokenizer.decode(gen_tensor[0], skip_special_tokens=True)
        return text.strip()


def integrate_ck_plug_with_analyzer(analyzer):
    """
    Monkey-patch a RetrievalQualityAnalyzer *instance* so that
    analyzer.generate_with_ck_plug uses CKPlugGenerator.

    Usage:
        analyzer = RetrievalQualityAnalyzer(...)
        analyzer = integrate_ck_plug_with_analyzer(analyzer)
    """
    ck_config = CKPlugConfig()  # you can pass custom defaults if you like
    ck_gen = CKPlugGenerator(
        model=analyzer.model,
        tokenizer=analyzer.tokenizer,
        device=analyzer.device,
        config=ck_config,
    )

    def generate_with_ck_plug_impl(
        self,
        query: str,
        context: str,
        alpha: float = 0.5,
        epsilon: float = 0.0,
        max_new_tokens: int = 64,
    ) -> str:
        return ck_gen.generate_with_control(
            query=query,
            context=context,
            alpha=alpha,
            epsilon=epsilon,
            max_new_tokens=max_new_tokens,
        )

    # Bind as an instance method
    analyzer.generate_with_ck_plug = generate_with_ck_plug_impl.__get__(analyzer, analyzer.__class__)
    return analyzer

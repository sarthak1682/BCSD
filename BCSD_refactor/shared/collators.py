"""Shared data collators for training datasets."""

import numpy as np
import torch
from shared.data_utils import asm_to_text
from shared.nova_utils import make_bidirectional_nova_mask


class TranslationCollator:
    """Causal-LM collator: source tokens get labels=-100, target tokens carry IDs."""

    def __init__(self, nova_tokenizer, max_length: int) -> None:
        self.tok = nova_tokenizer
        self.maxL = max_length
        self.pad = nova_tokenizer.tokenizer.pad_token_id or 0

    def __call__(self, batch) -> dict:
        ids_l, lbl_l, msk_l = [], [], []
        for src, tgt in batch:
            r = self.tok.encode(src, tgt, "1" * (len(src) + len(tgt)))
            L = min(len(r["input_ids"]), self.maxL)
            ids_l.append(r["input_ids"][:L])
            lbl_l.append(r["labels"][:L])
            msk_l.append(r["nova_attention_mask"][:L, :L])

        maxL = max(len(x) for x in ids_l)
        B = len(batch)
        p_ids = np.full((B, maxL), self.pad, dtype=np.int64)
        p_lbls = np.full((B, maxL), -100, dtype=np.int64)
        p_msk = np.zeros((B, maxL, maxL), dtype=np.float32)
        for i, (ids, lbls, msk) in enumerate(zip(ids_l, lbl_l, msk_l)):
            n = len(ids)
            p_ids[i, :n] = ids
            p_lbls[i, :n] = lbls
            p_msk[i, :n, :n] = msk
        return {
            "input_ids": torch.tensor(p_ids, dtype=torch.long),
            "labels": torch.tensor(p_lbls, dtype=torch.long),
            "nova_attention_mask": torch.tensor(p_msk, dtype=torch.bfloat16),
        }


class MNTPCollator:
    def __init__(self, nova_tokenizer, mask_id, mask_prob=0.15, max_length=1024):
        self.nova_tokenizer = nova_tokenizer
        self.base_tokenizer = nova_tokenizer.tokenizer
        self.mask_id = mask_id
        self.mask_prob = mask_prob
        self.label_ids = nova_tokenizer.labels
        self.max_length = max_length

    def __call__(self, batch):
        all_input_ids, all_labels, all_masks = [], [], []

        for item in batch:
            text = asm_to_text(item["asm"])
            char_types = "1" * len(text)
            result = self.nova_tokenizer.encode("", text, char_types)

            input_ids = result['input_ids'].copy()
            labels = np.full_like(result['labels'], -100)

            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]
                result['nova_attention_mask'] = result['nova_attention_mask'][
                                                :self.max_length, :self.max_length]

            for i in range(len(input_ids)):
                if input_ids[i] not in self.label_ids:
                    if np.random.random() < self.mask_prob:
                        labels[i] = input_ids[i]
                        input_ids[i] = self.mask_id

            bidir_mask = make_bidirectional_nova_mask(result['nova_attention_mask'])
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_masks.append(bidir_mask)

        max_len = min(max(len(x) for x in all_input_ids), self.max_length)
        pad_id = self.base_tokenizer.pad_token_id or 0

        padded_ids = np.full((len(batch), max_len), pad_id, dtype=np.int64)
        padded_labels = np.full((len(batch), max_len), -100, dtype=np.int64)
        padded_masks = np.zeros((len(batch), max_len, max_len), dtype=np.float32)

        for i, (ids, labs, mask) in enumerate(zip(all_input_ids, all_labels, all_masks)):
            L = len(ids)
            padded_ids[i, :L] = ids
            padded_labels[i, :L] = labs
            padded_masks[i, :L, :L] = mask

        return {
            "input_ids": torch.tensor(padded_ids),
            "labels": torch.tensor(padded_labels),
            "nova_attention_mask": torch.tensor(padded_masks, dtype=torch.bfloat16),
        }


class PairCollator:
    def __init__(self, nova_tokenizer, max_length=1024):
        self.nova_tokenizer = nova_tokenizer
        self.max_length = max_length
        self.label_ids = nova_tokenizer.labels
        self.pad_id = nova_tokenizer.tokenizer.pad_token_id or 0

    def __call__(self, batch):
        flat_texts = []
        func_ids = []
        has_func_ids = len(batch[0]) > 2
        
        instruct_template = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "
        for item in batch:
            if has_func_ids:
                query_asm, positive_asm, func_id = item
                flat_texts.extend([instruct_template + query_asm, positive_asm])
                func_ids.extend([func_id, func_id])
            else:
                query_asm, positive_asm = item
                flat_texts.extend([query_asm, positive_asm])

        all_ids, all_masks, all_label_positions = [], [], []

        for text in flat_texts:
            if text.startswith(instruct_template):
                asm_len = len(text) - len(instruct_template)
                char_types = "0" * len(instruct_template) + "1" * asm_len
            else:
                char_types = "1" * len(text)
            result = self.nova_tokenizer.encode("", text, char_types)
            ids = result['input_ids'][:self.max_length]
            raw_mask = result['nova_attention_mask']
            L = len(ids)
            mask = np.maximum(raw_mask[:L, :L], raw_mask[:L, :L].T)

            label_pos = [i for i, tid in enumerate(ids) if tid in self.label_ids]

            all_ids.append(ids)
            all_masks.append(mask)
            all_label_positions.append(label_pos)

        max_len = max(len(x) for x in all_ids)
        pad_ids = np.full((len(flat_texts), max_len), self.pad_id, dtype=np.int64)
        pad_masks = np.zeros((len(flat_texts), max_len, max_len), dtype=np.float32)

        for i, (ids, mask) in enumerate(zip(all_ids, all_masks)):
            L = len(ids)
            pad_ids[i, :L] = ids
            pad_masks[i, :L, :L] = mask

        res = {
            "input_ids": torch.tensor(pad_ids),
            "nova_attention_mask": torch.tensor(pad_masks, dtype=torch.bfloat16),
            "label_positions": all_label_positions,
        }
        if has_func_ids:
            res["func_ids"] = torch.tensor(func_ids, dtype=torch.long)
        return res


class DistillCollator:
    def __init__(self, nova_tokenizer, max_length=1024):
        self.nova_tokenizer = nova_tokenizer
        self.max_length = max_length
        self.pad_id = nova_tokenizer.tokenizer.pad_token_id or 0
        self.instruct_template = "Instruct: Retrieve the functionally equivalent assembly code.\nQuery: "
        # Precompute instruction token length for pool_mask construction (mirrors NV-Embed's
        # instruction_lens used to zero out instruction tokens from mean pooling).
        self.instruct_token_len = len(nova_tokenizer.tokenizer.tokenize(self.instruct_template))

    def __call__(self, batch):
        flat_texts = []
        func_ids = []
        for (q, p, fid) in batch:
            if not q.startswith(self.instruct_template):
                q = self.instruct_template + q
            flat_texts.extend([q, p])
            func_ids.extend([fid, fid])

        all_ids, all_masks, is_query = [], [], []
        for text in flat_texts:
            if text.startswith(self.instruct_template):
                asm_len = len(text) - len(self.instruct_template)
                char_types = "0" * len(self.instruct_template) + "1" * asm_len
                is_query.append(True)
            else:
                char_types = "1" * len(text)
                is_query.append(False)
            result = self.nova_tokenizer.encode("", text, char_types)
            ids = result['input_ids'][:self.max_length]
            raw_mask = result['nova_attention_mask']

            L = len(ids)
            mask = np.maximum(raw_mask[:L, :L], raw_mask[:L, :L].T)

            all_ids.append(ids)
            all_masks.append(mask)

        max_len = max(len(x) for x in all_ids)
        n = len(flat_texts)
        pad_ids = np.full((n, max_len), self.pad_id, dtype=np.int64)
        pad_masks = np.zeros((n, max_len, max_len), dtype=np.float32)

        for i, (ids, mask) in enumerate(zip(all_ids, all_masks)):
            L = len(ids)
            pad_ids[i, :L] = ids
            pad_masks[i, :L, :L] = mask

        key_padding_mask = (pad_ids == self.pad_id)

        # pool_mask: True = exclude token from LAL mean pooling.
        # Excludes padding positions AND instruction prefix token positions (for query texts),
        # mirroring NV-Embed's pool_mask that zeroes out instruction tokens before mean pooling.
        pool_mask = key_padding_mask.copy()
        for i, query in enumerate(is_query):
            if query:
                excl = min(self.instruct_token_len, len(all_ids[i]))
                pool_mask[i, :excl] = True

        return {
            "input_ids": torch.tensor(pad_ids),
            "nova_attention_mask": torch.tensor(pad_masks, dtype=torch.bfloat16),
            "key_padding_mask": torch.tensor(key_padding_mask),
            "pool_mask": torch.tensor(pool_mask),
            "func_ids": torch.tensor(func_ids, dtype=torch.long),
        }

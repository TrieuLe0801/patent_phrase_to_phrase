import os
from typing import Dict

import torch
import torch.nn as nn
from config import Config
from transformers import AutoConfig, AutoModel, AutoTokenizer


class ModelForCPC(nn.Module):
    def __init__(
        self,
        model_name: str = None,
        max_len: int = Config.MAX_LENGTH,
        batch_size: int = Config.BATCH_SIZE,
        device: torch.device = torch.cuda.current_device(),
        fc_dropout: float = Config.DROPOUT_RATE,
        normalize: bool = True,
        config: Dict = {},
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = device
        self.config = AutoConfig.from_pretrained(self.model_name, **config)
        self.model = AutoModel.from_pretrained(self.model_name, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.normalize = normalize
        self.fc_dropout = fc_dropout
        self.dropout = nn.Dropout(self.fc_dropout)
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.out_proj = nn.Linear(self.config.hidden_size, 1)

    def mean_pooling(
        self, model_output: object, attention_mask: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Create mean pool
        :params model_output:
        :params attention_mask: list attention mask
        :returns:
        """
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs["attention_mask"])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def save_pretrained(
        self,
        output_dir: str = None,
        checkpoint_name: str = None,
    ) -> None:
        """
        Save model
        :params output_dir: directory to save model
        :params checkpoint_name: checkpoint name
        """
        self.tokenizer.save_pretrained(output_dir)
        self.model.config.save_pretrained(output_dir)
        torch.save(self.model.state_dict(), os.path.join(output_dir, checkpoint_name))

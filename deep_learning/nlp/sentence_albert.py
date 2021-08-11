import json
import os
import torch
import torch.nn as nn
from transformers import AlbertConfig

from deep_learning.nlp.albert import AlbertEmbeddings, AlbertTransformer

__all__ = ['SentenceAlbertPP', 'SAlbertConfig']


class SAlbertConfig(AlbertConfig):
    def __init__(self, albert_path):
        self._load_albert_config(albert_path)
        super(SAlbertConfig, self).__init__(**self.model_config)

    def _load_albert_config(self, albert_path):
        if not os.path.isdir(albert_path):
            raise NotADirectoryError(f'Model path {albert_path} should be a directory.')
        config_path = os.path.join(albert_path, 'config.json')
        config = json.load(open(config_path, 'r'))
        self.model_config = config['model_param']


class SentenceAlbertPP(nn.Module):
    ''' Sentence Albert Pytorch Production '''
    def __init__(self, config: AlbertConfig):
        super(SentenceAlbertPP, self).__init__()
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoder_output = self.encoder(embedding_output)
        pooled_output = torch.sum(encoder_output, dim=1) / encoder_output.size()[-1]
        return pooled_output

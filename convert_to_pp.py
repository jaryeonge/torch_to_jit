import os
import json
import torch
import importlib

from logger import default_logger
from mapping import MODEL_MAPPING

convert_logger = default_logger()


class ConverterForJit:
    def __init__(self, model_path: str, save_name: str):
        self.save_name = save_name
        if not os.path.isdir(model_path):
            raise NotADirectoryError(f'Model path {model_path} should be a directory.')
        self.model_path = model_path
        self.config_path = os.path.join(model_path, 'config.json')
        self.state_path = os.path.join(model_path, 'state.pt')
        convert_logger.info(f'Model path: {model_path}\nSave name: {save_name}')

        self._load_config()
        self._load_model()
        convert_logger.info('Model is successfully loaded')

        self._convert()
        convert_logger.info('Model is converted to Jit Model')

        self._save_jit_model()

    def _load_config(self):
        config = json.load(open(self.config_path, 'r'))
        self.mapping = MODEL_MAPPING[config['model_import_path'][1]]
        self.model_config = config['model_param']

    def _load_model(self):
        state_dict = torch.load(self.state_path, map_location=torch.device('cpu'))
        model_state = state_dict['model_state']
        convert_state_fn = self.mapping.get('convert_state_fn')
        if convert_state_fn is not None:
            fn = getattr(importlib.import_module('model_state_converter'), convert_state_fn)
            model_state = fn(model_state)
        module = getattr(importlib.import_module(self.mapping['module_path']), self.mapping['module_name'])
        config = getattr(importlib.import_module(self.mapping['module_path']), self.mapping['config_name'])
        self.model = module(config(**self.model_config))
        self.model.load_state_dict(model_state)

    def _convert(self):
        inputs = self.mapping['example_input']
        self.jit_model = torch.jit.trace(self.model, inputs)

    def _save_jit_model(self):
        folder_path = f'./model_pp/{self.mapping["module_name"]}'
        save_path = os.path.join(folder_path, f'{self.save_name}.pt')
        os.makedirs(folder_path, exist_ok=True)
        torch.jit.save(self.jit_model, save_path)
        convert_logger.info(f'Jit model is saved in {save_path}')

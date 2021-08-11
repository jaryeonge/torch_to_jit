import torch

MODEL_MAPPING = {
    'AlbertForPreTraining': {
        'module_path': 'deep_learning.nlp.albert',
        'module_name': 'AlbertModelPP',
        'config_name': 'AlbertConfig',
        'convert_state_fn': 'albert_for_pretraining',
        'example_input': torch.randint(0, 30000, (1, 512))
    },
    'SAlbert': {
        'module_path': 'deep_learning.nlp.sentence_albert',
        'module_name': 'SentenceAlbertPP',
        'config_name': 'SAlbertConfig',
        'convert_state_fn': 'salbert_ft_model',
        'example_input': torch.randint(0, 30000, (1, 512))
    }
}
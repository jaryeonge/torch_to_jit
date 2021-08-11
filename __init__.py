import sys
import argparse


def main(config):
    if config.pp_type == 'jit':
        from convert_to_pp import ConverterForJit
        ConverterForJit(config.model_path, config.name)
    else:
        return f'pp_type {config.pp_type} is not defined.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch model to Production model')
    parser.add_argument('-p', '--pp_type', default='jit', type=str, help='Save type of the model')
    parser.add_argument('-m', '--model_path', type=str, help='Path of the model')
    parser.add_argument('-n', '--name', type=str, help='name of the model to be saved')
    args = parser.parse_args()
    main(args)

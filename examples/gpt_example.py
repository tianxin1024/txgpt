from __future__ import print_function

import os
import sys
import argparse
import configparser

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_num', type=int, default=24, help='number of layers')
    parser.add_argument('--ckpt_path', type=str, default='../models/megatron-models/c-model/345m/1-gpu', 
                        help='path to the checkpoint file.')
    parser.add_argument('--lib_path', type=str, default='./lib/libth_transformer.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--vocab_file', type=str, default='../models/gpt2-vocab.json',
                        help='vocabulary file.')
    parser.add_argument()

    args = parser.parse_args()

    ckpt_config = configparser.ConfigParser()
    ckpt_config_path = os.path.join(args.ckpt_path, 'config.ini')

    if os.path.isfile(ckpt_config_path):
        ckpt_config.read(ckpt_config_path)

    if 'gpt' in ckpt_config.keys():
        for args_key, config_key, func in [
            ('layer_num', 'num_layer', ckpt_config.getint),
            ('max_seq_len', 'max_pos_seq_len', ckpt_config.getint),
            ('weights_data_type', 'weight_data_type', ckpt_config.get),
        ]:
            if config_key in ckpt_config['gpt'].keys():
                prev_val = args.__dict__[args_key]
                args.__dict__[args_key] = func('gpt', config_key)


if __name__ == "__main__":
    main()

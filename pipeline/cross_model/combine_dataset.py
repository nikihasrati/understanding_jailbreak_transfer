import argparse
import os

from pipeline.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Combine cross-model transfer chunks.')
    parser.add_argument('--source-model-path', required=True)
    parser.add_argument('--target-model-path', required=True)
    parser.add_argument('--num-chunks', type=int, default=1)
    parser.add_argument('--delete-input-chunks', action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    source_cfg = Config(args.source_model_path)
    target_cfg = Config(args.target_model_path)
    save_dir = os.path.join(target_cfg.cross_model_transfer_generations_dir(), f'{source_cfg.model_alias}_to_{target_cfg.model_alias}')
    from pipeline.utils import utils
    input_dir = os.path.join(save_dir, 'evaluation_chunks')
    if not os.path.isdir(input_dir):
        input_dir = save_dir
    df = utils.concat_json_files_in_dir(input_dir)
    out_path = os.path.join(save_dir, 'cross_model_transfer_generations.json')
    df.to_json(out_path, orient='records', indent=2)
    if args.delete_input_chunks:
        for filename in os.listdir(save_dir):
            if filename == os.path.basename(out_path):
                continue
            full_path = os.path.join(save_dir, filename)
            if os.path.isfile(full_path):
                os.remove(full_path)


if __name__ == '__main__':
    main()

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

from pipeline.config import Config
from pipeline.utils import utils


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Set up data sets for cross-model analysis.')
    parser.add_argument('--source-model-path', '--source_model_path', dest='source_model_path', type=str, required=True, help='Path to the source model')
    parser.add_argument('--target-model-path', '--target_model_path', dest='target_model_path', type=str, required=True, help='Path to the target model')
    return parser.parse_args(argv)

def plot_success_matrix(success_matrix_df, source_cfg, target_cfg):
    save_dir = Path(source_cfg.FIGURES) / f'{source_cfg.model_alias}_to_{target_cfg.model_alias}'
    file_name = "cross_model_transfer_success_matrix.png"
    save_path = save_dir / file_name
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(25, 25))
    cmap = ListedColormap(['white', '#4C72B0'])
    ax.imshow(success_matrix_df, cmap=cmap, interpolation='none')
    # ax.set_title(f"Jailbreak Success Matrix for Transfer from {source_cfg.model_name()} to {target_cfg.model_name()}")
    ax.set_xticks(range(len(success_matrix_df.columns)))
    # ax.set_xticklabels(success_matrix_df.columns)
    ax.set_xticklabels([])
    ax.set_yticks(range(len(success_matrix_df.index)))
    # ax.set_yticklabels(success_matrix_df.index)
    ax.set_yticklabels([])
    ax.set_xlabel('Suffix', fontsize=150)
    ax.set_ylabel('Prompt', fontsize=150)
    plt.savefig(save_path)
    plt.close()


def transfer_generations_path(source_cfg, target_cfg) -> Path:
    return (
        Path(target_cfg.cross_model_transfer_generations_dir())
        / f'{source_cfg.model_alias}_to_{target_cfg.model_alias}'
        / 'cross_model_transfer_generations.json'
    )


def load_previously_refused_transfer_df(source_cfg, target_cfg) -> pd.DataFrame:
    transfer_df = pd.read_json(transfer_generations_path(source_cfg, target_cfg))
    target_previously_refused_indices = utils.get_previously_refused_indices(target_cfg)
    source_previously_refused_suffix_indices = utils.get_previously_refused_suffix_indices(source_cfg)
    return transfer_df[
        transfer_df['prompt_id'].isin(target_previously_refused_indices)
        & transfer_df['suffix_id'].isin(source_previously_refused_suffix_indices)
    ]


def data_analysis(source_cfg, target_cfg):
    transfer_df = load_previously_refused_transfer_df(source_cfg, target_cfg)
    success_matrix_df = utils.get_jailbreak_success_matrix_df(transfer_df)
    plot_success_matrix(success_matrix_df, source_cfg, target_cfg)

def main(argv=None):
    args = parse_args(argv)
    source_cfg = Config(model_path=args.source_model_path)
    target_cfg = Config(model_path=args.target_model_path)
    
    print("Source model path", args.source_model_path)
    print("Target model path", args.target_model_path)

    data_analysis(source_cfg, target_cfg)

if __name__ == "__main__":
    main()

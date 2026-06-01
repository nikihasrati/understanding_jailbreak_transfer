import argparse
import random
from pathlib import Path

def parse_args(argv=None):
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Parse arguments.")
    parser.add_argument('--model-path', '--model_path', dest='model_path', type=str, required=True, help='Path to the model')
    parser.add_argument(
        '--sample-random-state',
        type=int,
        default=0,
        help='RNG seed used to choose one generated suffix seed per source prompt for the sampled transfer plot.',
    )
    return parser.parse_args(argv)


parse_arguments = parse_args

def plot_with_error_bars(layers, mean_values, std_values, color):
    plt.plot(layers, mean_values, label=None, color=color)
    plt.fill_between(layers, mean_values - std_values, mean_values + std_values, alpha=0.2, color=color)

def plot_cosine_similarity_across_layers(cosine_similarities, color, num_layers):
    mean_values = cosine_similarities.mean(dim=0).cpu().detach().numpy()
    std_values = cosine_similarities.std(dim=0).cpu().detach().numpy()
    layers = np.arange(num_layers)

    plot_with_error_bars(layers, mean_values, std_values, color)

def plot_cosine_similarity_with_refusal(cosine_similarities, prompt_cosine_sim_with_refusal, num_layers, cfg, most_successful_suffix_ids, least_successful_suffix_ids):
    plot_cosine_similarity_across_layers(
        prompt_cosine_sim_with_refusal,
        color="blue",
        num_layers=num_layers
    )

    for suffix_id in most_successful_suffix_ids:
        plot_cosine_similarity_across_layers(
            cosine_similarities[:, suffix_id, :], 
            color="green",
            num_layers=num_layers)
    
    for suffix_id in least_successful_suffix_ids:
        plot_cosine_similarity_across_layers(
            cosine_similarities[:, suffix_id, :], 
            color="orange",
            num_layers=num_layers)
        
    plt.xlabel("Layer number", fontsize=24)
    plt.ylabel("Cosine sim w refusal", fontsize=24)
    # plt.title(f"{cfg.model_name().capitalize()} Average Cosine Similarity with Refusal Direction Across Layers (Multi-Seed)", fontsize=10)
    # plt.legend(handles=[
        # plt.Line2D([0], [0], color='blue', label='Harmful Prompt'),
        # plt.Line2D([0], [0], color='green', label='Harmful Prompt + Top 3 Most Successful Suffixes'),
        # plt.Line2D([0], [0], color='orange', label='Harmful Prompt + Top 3 Least Successful Suffixes')
    # ])
    save_path = Path(cfg.figures_dir()) / 'cosine_similarity_across_layers_suffixes_direction_multi_seed.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_random_suffix_per_source_prompt_transfer_matrix(transfer_df, cfg, random_state=0):
    """Build a transfer matrix after sampling one generated seed per source prompt."""
    df = transfer_df.copy()
    suffixes_per_prompt = cfg.num_suffixes_per_prompt
    df["source_prompt_id"] = df["suffix_id"] // suffixes_per_prompt

    expected_seed = df["suffix_id"] % suffixes_per_prompt
    if not expected_seed.eq(df["seed"]).all():
        raise ValueError(
            "Expected suffix_id to encode source prompt and seed as "
            "source_prompt_id * num_seeds + seed."
        )

    refused_prompt_ids = set(utils.get_previously_refused_indices(cfg).tolist())
    df = df[df["source_prompt_id"].isin(refused_prompt_ids)]

    rng = random.Random(random_state)
    suffix_seed_df = df[["source_prompt_id", "seed", "suffix_id"]].drop_duplicates()
    selected_suffix_ids = []
    selected_seeds_by_source_prompt = {}

    for source_prompt_id, group in suffix_seed_df.groupby("source_prompt_id", sort=True):
        seeds = sorted(group["seed"].unique().tolist())
        selected_seed = rng.choice(seeds)
        selected_suffixes = group.loc[group["seed"] == selected_seed, "suffix_id"].unique()
        if len(selected_suffixes) != 1:
            raise ValueError(f"Expected one suffix for source prompt {source_prompt_id} and seed {selected_seed}.")

        selected_suffix_ids.append(selected_suffixes[0])
        selected_seeds_by_source_prompt[int(source_prompt_id)] = int(selected_seed)

    sampled_df = df[df["suffix_id"].isin(selected_suffix_ids)]
    success_matrix = sampled_df.pivot(index="prompt_id", columns="source_prompt_id", values="jailbroken")
    return success_matrix.astype(int), selected_seeds_by_source_prompt

def data_analysis(cfg, sample_random_state=0):
    ### No Transfer ###
    no_transfer_multi_seed_df = utils.get_multi_seed_df(cfg, transfer=False)

    no_transfer_success_matrix_df = utils.get_multi_seed_jailbreak_success_matrix_no_transfer_df(no_transfer_multi_seed_df)
    utils.plot_jailbreak_success_matrix(no_transfer_success_matrix_df, cfg, x_label="Seed", y_label="Prompt", multi_seed=True)

    num_prompts, num_seeds = no_transfer_success_matrix_df.shape

    number_of_successful_jailbreaks = no_transfer_success_matrix_df.sum(axis=1).sum(axis=0)
    fraction_of_successful_jailbreaks = number_of_successful_jailbreaks / (num_prompts * num_seeds)
    print(f"Fraction of successful jailbreaks: {fraction_of_successful_jailbreaks:.2}")

    number_of_prompts_with_at_least_one_successful_jailbreak = (no_transfer_success_matrix_df.sum(axis=1) > 0).sum()
    fraction_of_prompts_with_at_least_one_successful_jailbreak = number_of_prompts_with_at_least_one_successful_jailbreak / num_prompts
    print(f"Fraction of prompts with at least one successful jailbreak: {fraction_of_prompts_with_at_least_one_successful_jailbreak:.2}")

    ### Transfer ###
    n = 3

    transfer_multi_seed_df = utils.get_multi_seed_df(cfg, transfer=True)
    transfer_success_matrix_df = utils.get_jailbreak_success_matrix_df(transfer_multi_seed_df)
    num_prompts, num_suffixes = transfer_success_matrix_df.shape

    sampled_transfer_success_matrix_df, selected_seeds = get_random_suffix_per_source_prompt_transfer_matrix(
        transfer_multi_seed_df,
        cfg,
        random_state=sample_random_state,
    )
    sampled_transfer_figure_path = cfg.multi_seed_random_suffix_transfer_matrix_figure_path(sample_random_state)
    utils.plot_jailbreak_success_matrix(
        sampled_transfer_success_matrix_df,
        cfg,
        x_label="Suffix",
        y_label="Prompt",
        save_path=sampled_transfer_figure_path,
    )
    print(f"Saved sampled multi-seed transfer matrix to {sampled_transfer_figure_path}")
    print(f"Selected seeds by source prompt: {selected_seeds}")

    most_successful_suffixes = transfer_success_matrix_df.sum(axis=0).nlargest(n).index.tolist()
    least_successful_suffixes = transfer_success_matrix_df.sum(axis=0).nsmallest(n).index.tolist()

    print(f"Most successful suffixes: {most_successful_suffixes}")
    print(f"Least successful suffixes: {least_successful_suffixes}")

    # refactor into own function
    consine_sim_with_refusal = torch.load(cfg.cosine_similarity_with_refusal_path(), weights_only=True)
    consine_sim_with_refusal = consine_sim_with_refusal.reshape(cfg.num_prompts, num_suffixes, -1)
    prev_refused_indices = utils.get_previously_refused_indices(cfg).tolist()
    consine_sim_with_refusal = consine_sim_with_refusal[prev_refused_indices, :, :]
    _, _, num_layers = consine_sim_with_refusal.shape

    # Plot the cosine similarity with refusal direction across layers for the most and least successful suffixes
    refusal_direction = utils.get_refusal_direction(cfg.arditi_et_al_refusal_direction_dir()).half()
    prompt_activations = get_prompt_activations(cfg)

    prompt_cosine_sim_with_refusal = torch.nn.functional.cosine_similarity(prompt_activations, refusal_direction, dim=-1)
    print(prompt_cosine_sim_with_refusal.shape)

    plot_cosine_similarity_with_refusal(
        consine_sim_with_refusal,
        prompt_cosine_sim_with_refusal,
        num_layers, 
        cfg, 
        most_successful_suffix_ids=most_successful_suffixes, 
        least_successful_suffix_ids=least_successful_suffixes
    )

def main(argv=None):
    args = parse_args(argv)
    global torch, pd, np, plt, utils, Config, get_prompt_activations
    import torch
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pipeline.utils import utils
    from pipeline.config import Config
    from pipeline.utils.activation_utils import get_prompt_activations
    cfg = Config(model_path=args.model_path)
    data_analysis(cfg, sample_random_state=args.sample_random_state)


if __name__ == "__main__":
    main()

import argparse
import torch
import torch.nn.functional as F

from pipeline.config import Config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Inspect a model with its stored refusal direction.')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--prompt', default='Write a short explanation of machine learning.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = Config(args.model_path)
    from pipeline.model_utils.model_factory import construct_model_base
    from pipeline.submodules.generate_activations import get_activations
    model = construct_model_base(cfg.model_path)
    direction = torch.load(cfg.arditi_et_al_refusal_direction_path(), map_location='cpu', weights_only=True)
    layer = cfg.arditi_et_al_refusal_direction_layer()
    acts = get_activations(model, [args.prompt], model.model_block_modules, batch_size=1)[0, layer, :].cpu()
    print('layer:', layer)
    print('dot:', torch.dot(acts, direction).item())
    print('cosine:', F.cosine_similarity(acts, direction, dim=0).item())


if __name__ == '__main__':
    main()

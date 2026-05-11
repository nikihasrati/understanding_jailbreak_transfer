import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Use pipeline.activations.save_activations for cross-model activation regeneration.')
    parser.add_argument('--source-model-path', required=True)
    parser.add_argument('--target-model-path', required=True)
    parser.add_argument('--output-format', choices=['canonical_tensor_chunks', 'nested_by_suffix_layer'], default='canonical_tensor_chunks')
    return parser.parse_args()


def main():
    args = parse_args()
    raise SystemExit(
        'Cross-model activation regeneration is documented in docs/activations.md. '
        'Use pipeline.activations.save_activations with the target model and cross-model dataset.'
    )


if __name__ == '__main__':
    main()

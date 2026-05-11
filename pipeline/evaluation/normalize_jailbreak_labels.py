import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Normalize jailbreak labels to booleans in a JSON generations file.')
    parser.add_argument('--path', required=True)
    parser.add_argument('--column', default='jailbroken')
    return parser.parse_args()


def to_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {'true', '1', 'yes'}:
        return True
    if text in {'false', '0', 'no'}:
        return False
    return value


def main():
    args = parse_args()
    df = pd.read_json(args.path)
    if args.column not in df.columns:
        raise KeyError(args.column)
    df[args.column] = df[args.column].map(to_bool)
    df.to_json(args.path, orient='records', indent=2)


if __name__ == '__main__':
    main()

import argparse

import pandas as pd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Normalize jailbreak labels to booleans in a JSON generations file.')
    parser.add_argument('--path', required=True)
    parser.add_argument('--output', default=None, help='Output JSON path. Defaults to overwriting --path.')
    parser.add_argument('--column', default='jailbroken')
    return parser.parse_args(argv)


def to_bool(value):
    if value is None:
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {'true', '1', 'yes'}:
        return True
    if text in {'false', '0', 'no'}:
        return False
    return value


def main(argv=None):
    args = parse_args(argv)
    df = pd.read_json(args.path)
    if args.column not in df.columns:
        raise KeyError(args.column)
    df[args.column] = df[args.column].map(to_bool)
    df.to_json(args.output or args.path, orient='records', indent=2)


if __name__ == '__main__':
    main()

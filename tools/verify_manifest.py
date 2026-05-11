import argparse

from pipeline.artifacts import verify_manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Verify chunk checksums in a manifest.')
    parser.add_argument('--manifest', required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        verify_manifest(args.manifest)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(f'OK: {args.manifest}')


if __name__ == '__main__':
    main()

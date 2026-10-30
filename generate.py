"""Generate an EVA continuation with the bundled reference checkpoint."""
import argparse
from server import DEFAULT_LAB, Voynich, request_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', default='qokedy chedy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.8)
    args = parser.parse_args()
    engine = Voynich(DEFAULT_LAB)
    options = request_options(dict(model='voynich-experimental',
        messages=[dict(role='user', content=args.prompt)], seed=args.seed,
        max_tokens=args.max_tokens, temperature=args.temperature))
    print(engine.complete(options)[0])


if __name__ == '__main__':
    main()

import json
import argparse
import tqdm


def main(args):
    f = open(args.group_json, 'r')
    group_dict = json.load(f)
    f.close()

    f = open(args.input_index, 'r')
    input_dict = json.load(f)
    f.close()

    output_dict = {}

    for ls in tqdm.tqdm(group_dict.values()):
        for c in ls:
            tmp = []
            if not c in input_dict.keys():
                continue
            else:
                for cc in ls:
                    output_dict[cc] = input_dict[c]
                break

    print(len(output_dict))
    f = open(args.output_index, 'w+')
    json.dump(output_dict, f)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group_json", type=str, 
        help="""Path to precomputed alignment directory, with one subdirectory 
                per chain."""
    )
    parser.add_argument("--input_index", type=str)
    parser.add_argument("--output_index", type=str)

    args = parser.parse_args()

    main(args)

import argparse
import json,sys
from datasets import load_from_disk,load_dataset
import textwrap

def parse_argument(sys_argv):
    """Parses arguments from command line.
    Args:
        sys_argv: the list of arguments (strings) from command line.
    Returns:
        A struct whose member corresponds to the required (optional) variable.
        For example,
        ```
        args = parse_argument(['main.py' '--input', 'a.txt', '--num', '10'])
        args.input       # 'a.txt'
        args.num         # 10
        ```
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)

    # Training parameters
    parser.add_argument(
        "--dataset_path", type=str,
        default=None,
        help=textwrap.dedent("input dataset path, reads from stdin by default")
    )
    parser.add_argument(
        "--option", choices=["lm2hf","hf2lm"],
        default="lm2hf",
    )
    parser.add_argument(
        "--output_path", type=str,
        default=None,
        help=textwrap.dedent("output dataset path, writes to stdout by default")
    )

    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args

if __name__ == "__main__":
    args = parse_argument(sys.argv)
    if args.option == "lm2hf":
        ds = load_dataset('json', data_files=args.dataset_path, field="instances",split="train",
                use_auth_token=None,)
        ds.save_to_disk(args.output_path)
    else:
        data_dict = load_from_disk(args.dataset_path)
        out = {}
        out["type"] = "text2text"
        out["instances"] = []
        for item in data_dict:
            out["instances"].append({"instruction":item["instruction"],"input":item["input"],"output":item["output"]})
        with open(args.output_path, "w") as fout:
            json.dump(out, fout, indent=4, ensure_ascii=False)
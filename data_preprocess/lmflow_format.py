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
    parser.add_argument(
        "--language", type=str,
        default="chinese",
        help=textwrap.dedent("output dataset path, writes to stdout by default")
    )

    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args

if __name__ == "__main__":
    args = parse_argument(sys.argv)
    with open("data_preprocess/prompt_template.json","r") as file:
        prompt_template = json.load(file)
    if args.option == "lm2hf":
        ds = load_dataset('json', data_files=args.dataset_path, field="instances",split="train",
                use_auth_token=None,)
        ds.save_to_disk(args.output_path)
    else:
        data_dict = load_from_disk(args.dataset_path)
        out_dataset = {}
        out_dataset["type"] = "text2text"
        out_dataset["instances"] = []
        for item in data_dict:
            instruction_key = "instruction"
            if "prompt" in item.keys():
                instruction_key = "prompt"
            if "context" in item.keys():
                context_key = "context"
            elif "reformulations" in item.keys():
                instruction_key = "instruction_with_input"
                item = item["instances"][0]
                context_key = "context"
            else:
                context_key = "input"
            if context_key in item.keys() and item[context_key] != "":
                instruction = prompt_template[args.language+"_prompt_input"].format(input=item[context_key],instruction=item[instruction_key])
            else:
                instruction = prompt_template[args.language+"_prompt_no_input"].format(instruction=item[instruction_key])
            if "response" in item.keys():
                output = item["response"]
            else:
                output = item["output"]
            out_dataset["instances"].append({"instruction":instruction,"input":"","output":output})
        print(len(out_dataset["instances"]))
        with open(args.output_path, "w") as fout:
            json.dump(out_dataset, fout, indent=4, ensure_ascii=False)
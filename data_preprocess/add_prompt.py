#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""
Adds prompt structure to a text2text dataset.
"""
from __future__ import absolute_import

import argparse
import json
import textwrap
import sys

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
        "--output_path", type=str,
        default=None,
        help=textwrap.dedent("output dataset path, writes to stdout by default")
    )
    parser.add_argument(
        "--language", type=str,
        default="chinese",
        help=textwrap.dedent("the language of the data")
    )

    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args


def main():
    args = parse_argument(sys.argv)
    if args.dataset_path is not None:
        with open(args.dataset_path, "r") as fin:
            data_dict = json.load(fin)
    else:
        data_dict = json.load(sys.stdin)

    if data_dict["type"] != "text2text":
        raise NotImplementedError(
            "only support text2text prompt augmentation"
        )

    with open("data_preprocess/prompt_template.json","r") as file:
        prompt_template = json.load(file)
    format_data = []
    for instance in data_dict["instances"]:
        if instance["input"] == "":
            prompt_structure = prompt_template[args.language+"_prompt_no_input"]
            idata = prompt_structure.format(instruction=instance["instruction"])
        else:
            prompt_structure = prompt_template[args.language+"_prompt_input"]
            idata = prompt_structure.format(instruction=instance["instruction"],input=instance["input"])
        format_data.append(
            {
                "instruction": idata,
                "input": "",
                "output": instance["output"],
            }
        )
    data_dict["instances"] = format_data
    if args.output_path is not None:
        with open(args.output_path, "w") as fout:
            json.dump(data_dict, fout, indent=4, ensure_ascii=False)
    else:
        json.dump(data_dict, sys.stdout, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()

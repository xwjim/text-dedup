import argparse
import json,sys,os,random
from datasets import load_from_disk,load_dataset
import textwrap
import re

instruction_set = {
    "chinese_summary": [
        "根据已知信息生成一个简洁的摘要。",
        "这个新闻的摘要是什么？",
        "为这个新闻生成一个简洁的摘要。",
        "请生成这个已知信息的摘要。",
        "已知信息的主要内容的主要内容是什么？",
    ],
    "english_summary": [
        "give the summary of the input.",
        "what's the summary of the input?",
        "summary the input.",
        "give the main point of the input",
    ],
    "chinese_qa": [
        "根据下面已知信息，简洁和专业地来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”。",
        "根据内容回答问题。",
        "根据已知信息回复问题。",
        "根据内容回复问题。",
    ],
    "english_qa": [
        "please answer the question according to the content. If the content does not contain the information of the question, say we can not get the answer according to the content.",
        "answer the quesion according to the content.",
        "write a response that appropriately completes the question."
    ],
}

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
        "--option", choices=["lm2hf","hf2lm","pair2lm","line2lm"],
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
        help=textwrap.dedent("the language of the data")
    )
    parser.add_argument(
        "--instruction_key", type=str,
        default="instruction",
        help=textwrap.dedent("instruction key")
    )
    parser.add_argument(
        "--input_key", type=str,
        default="input",
        help=textwrap.dedent("input key")
    )
    parser.add_argument(
        "--output_key", type=str,
        default="output",
        help=textwrap.dedent("output key")
    )
    parser.add_argument(
        "--task", choices=["qa","summary"],
        default="summary",
    )
    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args

def format_hf2lm(args):
    data_dict = load_from_disk(args.dataset_path)
    out_dataset = {}
    out_dataset["type"] = "text2text"
    out_dataset["instances"] = []
    print(data_dict[0]) 
    for item in data_dict:
        if "instances" in item:
            item = item["instances"][0]
        if args.input_key in item.keys():
            input_text = item[args.input_key]
        else:
            input_text = ""
        out_dataset["instances"].append({"instruction":item[args.instruction_key],"input":input_text,"output":item[args.output_key]})
    print(len(out_dataset["instances"]))
    with open(args.output_path, "w") as fout:
        json.dump(out_dataset, fout, indent=4, ensure_ascii=False)

def format_lm2hf(args):
    ds = load_dataset('json', data_files=args.dataset_path, field="instances",split="train",
                use_auth_token=None,)
    ds.save_to_disk(args.output_path)

def format_pair2lm(args):
    """
    language
    task
    """
    with open(os.path.join(args.dataset_path,"test.src"),"r") as file:
        src = file.readlines()
    with open(os.path.join(args.dataset_path,"test.tgt"),"r") as file:
        tgt = file.readlines()
    
    out_dataset = {}
    out_dataset["type"] = "text2text"
    out_dataset["instances"] = []
    
    for it_s,it_t in zip(src,tgt):
        if args.language == "chinese":
            it_s = re.sub("\s","",it_s)
            it_t = re.sub("\s","",it_t)
        out_dataset["instances"].append(
            {
                "instruction":instruction_set[args.language+"_"+args.task][random.randint(0,len(instruction_set)-1)],
                "input":it_s,
                "output":it_t,
            }
        )
    print(len(out_dataset["instances"]))
    with open(args.output_path, "w") as fout:
        json.dump(out_dataset, fout, indent=4, ensure_ascii=False)

def format_line2lm(args):
    """
    language
    task
    """
    with open(args.dataset_path,"r") as file:
        data = [json.loads(line) for line in file.readlines()]
    
    out_dataset = {}
    out_dataset["type"] = "text2text"
    out_dataset["instances"] = []
    
    for item in data:
        out_dataset["instances"].append(
            {
                "instruction":item[args.instruction_key],
                "input":item[args.input_key],
                "output":item[args.output_key],
            }
        )
    print(len(out_dataset["instances"]))
    with open(args.output_path, "w") as fout:
        json.dump(out_dataset, fout, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    args = parse_argument(sys.argv)

    format_func = {
        "lm2hf": format_lm2hf,
        "hf2lm": format_hf2lm,
        "pair2lm": format_pair2lm,
        "line2lm":format_line2lm,
    }

    function = format_func[args.option]
    function(args)
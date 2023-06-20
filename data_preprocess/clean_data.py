from datasets import load_from_disk




if __name__ == "__main__":

    dataset = load_from_disk("../LMFlow/data/BELLE/train/train.arrow/")

    print(len(dataset))
    for item in dataset:
        if "如果你需要任何帮助" in item["instruction"]:
            print(item)
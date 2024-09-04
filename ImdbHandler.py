from datasets import load_dataset


class ImdbDataset:
    def __init__(self):
        self.dataset = load_dataset("stanfordnlp/imdb")
        self.train_dataset = self.dataset["train"]
        self.eval_dataset = self.dataset["test"]

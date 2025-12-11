import os,re
from transformers import AutoTokenizer
import lightning as L
from torch.utils.data import Dataset, DataLoader



class Seq2SeqNERDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.read_data()

    def read_data(self):
        data_list = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for data_item in f.readlines():
                data_list.append(data_item.strip().split(' ')[1])
        return data_list
    
    def __getitem__(self, index):
        label = self.data[index]
        input = re.sub(r'[<>\[\]()]', '', label)
        return input, label

    def __len__(self):
        return len(self.data)
    

class Seq2SeqNERDataModule(L.LightningDataModule):
    def __init__(self, **hyperargs):
        super().__init__()
        self.data_path = hyperargs["data_path"]
        self.batch_size = hyperargs["batch_size"]
        self.max_length = hyperargs["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(hyperargs["model_name_or_path"])
    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = Seq2SeqNERDataset(os.path.join(self.data_path, 'train.txt'))
        if stage == "test":
            self.test_dataset = Seq2SeqNERDataset(os.path.join(self.data_path, 'test.txt'))
        if stage == "predict":
            self.predict_dataset = Seq2SeqNERDataset(os.path.join(self.data_path, 'test.txt'))

        self.dev_dataset = Seq2SeqNERDataset(os.path.join(self.data_path, 'dev.txt'))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.my_collate_fn)

    def dev_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, collate_fn=self.my_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.my_collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, collate_fn=self.my_collate_fn)
    
    def my_collate_fn(self, data):
        batch_inputs_list = []
        batch_labels_list = []
        for data_item in data:
            batch_inputs_list.append(data_item[0])
            batch_labels_list.append(data_item[1])
        tokenizer_result = self.tokenizer(text = batch_inputs_list,\
                                         text_target = batch_labels_list,\
                                         padding = "max_length",\
                                         max_length = self.max_length,\
                                         truncation = True,\
                                         return_tensors = "pt")
        labels = tokenizer_result["labels"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        tokenizer_result["labels"] = labels
        return tokenizer_result
    

if __name__ == "__main__":
    class Hyperargs:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # 使用示例
    hyperargs = Hyperargs(data_path="data/aishell_ner", batch_size=8, model_name_or_path = "cache/bart-large",max_length = 50)
    datamodule = Seq2SeqNERDataModule(hyperargs)
    datamodule.setup(stage = "test")

    test_dataloader = datamodule.test_dataloader()

    tokenizer = datamodule.tokenizer
    for item in test_dataloader:
        print(item['input_ids'][0], item['attention_mask'][0], item['labels'][0])
        print(tokenizer.decode(item['input_ids'][0]), '\n', tokenizer.decode(item['attention_mask'][0]), '\n', tokenizer.decode(item['labels'][0]))
        break
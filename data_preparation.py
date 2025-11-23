import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import spacy
from collections import Counter
import requests
import zipfile
import os
import io

# Bước 1: Chuẩn bị dữ liệu cho mô hình Encoder-Decoder LSTM
# Bao gồm: Tải Multi30K, Tokenizer với SpaCy, Xây dựng Vocabulary, Tạo DataLoader với Padding/Packing

# 1.1: Tải và giải nén Multi30K dataset
# Multi30K không còn được hỗ trợ trực tiếp trong torchtext mới, nên tải raw files từ GitHub
def download_multi30k():
    """
    Tải và giải nén dataset Multi30K từ GitHub.
    Dataset bao gồm các file train.en, train.fr, val.en, val.fr, test_2016.en, test_2016.fr, v.v.
    """
    url = "https://github.com/multi30k/dataset/archive/refs/heads/master.zip"
    response = requests.get(url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    zip_file.extractall("multi30k_raw")
    print("Đã tải và giải nén Multi30K dataset vào thư mục 'multi30k_raw'")

# Gọi hàm tải dataset (chỉ chạy một lần)
# download_multi30k()

# Giả sử dataset đã được tải vào thư mục 'multi30k_raw/dataset-master/data/task1/tok'
# Các file: train.lc.norm.tok.en, train.lc.norm.tok.fr, val.lc.norm.tok.en, val.lc.norm.tok.fr, test_2016_flickr.lc.norm.tok.en, test_2016_flickr.lc.norm.tok.fr

# 1.2: Tokenizer với SpaCy
# Tải và sử dụng spaCy models cho tiếng Anh và tiếng Pháp
try:
    spacy_en = spacy.load("en_core_web_sm")
    spacy_fr = spacy.load("fr_core_news_sm")
except OSError:
    # Nếu chưa cài, cài đặt models
    os.system("python -m spacy download en_core_web_sm")
    os.system("python -m spacy download fr_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")
    spacy_fr = spacy.load("fr_core_news_sm")

def tokenize_en(text):
    """
    Tokenize văn bản tiếng Anh sử dụng spaCy.
    """
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_fr(text):
    """
    Tokenize văn bản tiếng Pháp sử dụng spaCy.
    """
    return [tok.text.lower() for tok in spacy_fr.tokenizer(text)]

# 1.3: Xây dựng Vocabulary
class Vocab:
    def __init__(self, freqs, max_size=10000):
        """
        Xây dựng vocabulary từ counter của tần suất từ.
        Giữ lại top max_size từ phổ biến nhất + special tokens.
        """
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        words = [word for word, _ in freqs.most_common(max_size)]
        self.itos.update({i+4: word for i, word in enumerate(words)})
        self.stoi = {word: idx for idx, word in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def numericalize(self, tokens):
        """
        Chuyển list tokens thành list indices.
        """
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]

# Hàm xây dựng vocab từ file
def build_vocab(filepath, tokenizer, max_size=10000):
    """
    Xây dựng vocab từ file văn bản.
    """
    counter = Counter()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = tokenizer(line.strip())
            counter.update(tokens)
    return Vocab(counter, max_size)

# Paths đến files (sử dụng file .lc.norm.tok)
train_en_path = "multi30k_raw/dataset-master/data/task1/tok/train.lc.norm.tok.en"
train_fr_path = "multi30k_raw/dataset-master/data/task1/tok/train.lc.norm.tok.fr"

# Xây dựng vocab cho EN và FR
en_vocab = build_vocab(train_en_path, tokenize_en)
fr_vocab = build_vocab(train_fr_path, tokenize_fr)

print(f"English Vocab size: {len(en_vocab)}")
print(f"French Vocab size: {len(fr_vocab)}")

# 1.4: Tạo Dataset và DataLoader với Padding/Packing
class TranslationDataset(Dataset):
    def __init__(self, src_path, trg_path, src_tokenizer, trg_tokenizer, src_vocab, trg_vocab):
        """
        Dataset cho translation: src (EN) -> trg (FR)
        """
        self.src_data = []
        self.trg_data = []
        with open(src_path, 'r', encoding='utf-8') as f_src, open(trg_path, 'r', encoding='utf-8') as f_trg:
            for src_line, trg_line in zip(f_src, f_trg):
                src_tokens = src_tokenizer(src_line.strip())
                trg_tokens = trg_tokenizer(trg_line.strip())
                # Thêm <sos> và <eos>
                src_indices = [src_vocab.stoi["<sos>"]] + src_vocab.numericalize(src_tokens) + [src_vocab.stoi["<eos>"]]
                trg_indices = [trg_vocab.stoi["<sos>"]] + trg_vocab.numericalize(trg_tokens) + [trg_vocab.stoi["<eos>"]]
                self.src_data.append(torch.tensor(src_indices))
                self.trg_data.append(torch.tensor(trg_indices))

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.trg_data[idx]

# Hàm collate_fn để padding và packing
def collate_fn(batch):
    """
    Collate function: Sắp xếp batch theo độ dài src giảm dần (để giữ alignment giữa src và trg), pad sequences, và pack.
    """
    # Zip cặp (src, trg) để giữ alignment
    paired_batch = list(zip(*batch))  # paired_batch[0] = src_batch, paired_batch[1] = trg_batch
    src_batch, trg_batch = paired_batch
    # Sắp xếp cả cặp dựa trên độ dài src giảm dần
    sorted_indices = sorted(range(len(src_batch)), key=lambda i: len(src_batch[i]), reverse=True)
    src_batch = [src_batch[i] for i in sorted_indices]
    trg_batch = [trg_batch[i] for i in sorted_indices]
    # Pad sequences
    src_padded = pad_sequence(src_batch, padding_value=en_vocab.stoi["<pad>"])
    trg_padded = pad_sequence(trg_batch, padding_value=fr_vocab.stoi["<pad>"])
    # Lấy lengths
    src_lengths = torch.tensor([len(seq) for seq in src_batch])
    trg_lengths = torch.tensor([len(seq) for seq in trg_batch])
    # Pack padded sequences
    src_packed = pack_padded_sequence(src_padded, src_lengths, enforce_sorted=False)
    trg_packed = pack_padded_sequence(trg_padded, trg_lengths, enforce_sorted=False)
    return src_packed, trg_packed

# Tạo datasets
train_dataset = TranslationDataset(train_en_path, train_fr_path, tokenize_en, tokenize_fr, en_vocab, fr_vocab)
val_en_path = "multi30k_raw/dataset-master/data/task1/tok/val.lc.norm.tok.en"
val_fr_path = "multi30k_raw/dataset-master/data/task1/tok/val.lc.norm.tok.fr"
val_dataset = TranslationDataset(val_en_path, val_fr_path, tokenize_en, tokenize_fr, en_vocab, fr_vocab)

# Tạo DataLoaders
batch_size = 32  # Có thể điều chỉnh
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print("Đã chuẩn bị xong DataLoader cho training và validation.")

import torch
from data_preparation import en_vocab, fr_vocab, train_loader, val_loader, tokenize_en, tokenize_fr

# Thorough testing cho data preparation

print("=== Thorough Testing cho Data Preparation ===")

# 1. Kiểm tra vocab sizes
print(f"English Vocab size: {len(en_vocab)}")
print(f"French Vocab size: {len(fr_vocab)}")

# 2. Kiểm tra special tokens
print(f"English <pad> index: {en_vocab.stoi['<pad>']}")
print(f"English <sos> index: {en_vocab.stoi['<sos>']}")
print(f"English <eos> index: {en_vocab.stoi['<eos>']}")
print(f"English <unk> index: {en_vocab.stoi['<unk>']}")

print(f"French <pad> index: {fr_vocab.stoi['<pad>']}")
print(f"French <sos> index: {fr_vocab.stoi['<sos>']}")
print(f"French <eos> index: {fr_vocab.stoi['<eos>']}")
print(f"French <unk> index: {fr_vocab.stoi['<unk>']}")

# 3. Kiểm tra tokenizer
sample_en = "Hello world!"
sample_fr = "Bonjour le monde!"

tokens_en = tokenize_en(sample_en)
tokens_fr = tokenize_fr(sample_fr)

print(f"English tokens: {tokens_en}")
print(f"French tokens: {tokens_fr}")

# 4. Kiểm tra numericalize
indices_en = en_vocab.numericalize(tokens_en)
indices_fr = fr_vocab.numericalize(tokens_fr)

print(f"English indices: {indices_en}")
print(f"French indices: {indices_fr}")

# 5. Kiểm tra một batch từ train_loader
for src_packed, trg_packed in train_loader:
    print(f"Source packed data shape: {src_packed.data.shape}")
    print(f"Source batch sizes: {src_packed.batch_sizes}")
    print(f"Target packed data shape: {trg_packed.data.shape}")
    print(f"Target batch sizes: {trg_packed.batch_sizes}")

    # Decode một cặp mẫu để kiểm tra alignment
    # Lấy sample đầu tiên (độ dài lớn nhất sau sort)
    src_sample = src_packed.data[:src_packed.batch_sizes[0]]
    trg_sample = trg_packed.data[:trg_packed.batch_sizes[0]]

    # Chuyển về text (bỏ <sos> và <eos>)
    src_text = [en_vocab.itos[idx.item()] for idx in src_sample if idx.item() not in [0,1,2]]
    trg_text = [fr_vocab.itos[idx.item()] for idx in trg_sample if idx.item() not in [0,1,2]]

    print(f"Sample English: {' '.join(src_text)}")
    print(f"Sample French: {' '.join(trg_text)}")

    # Kiểm tra thêm một vài samples
    print("\nAdditional samples:")
    for i in range(min(3, len(src_packed.batch_sizes))):
        start = sum(src_packed.batch_sizes[:i])
        end = start + src_packed.batch_sizes[i]
        src_sample_i = src_packed.data[start:end]
        trg_sample_i = trg_packed.data[start:end]

        src_text_i = [en_vocab.itos[idx.item()] for idx in src_sample_i if idx.item() not in [0,1,2]]
        trg_text_i = [fr_vocab.itos[idx.item()] for idx in trg_sample_i if idx.item() not in [0,1,2]]

        print(f"Sample {i+1} English: {' '.join(src_text_i)}")
        print(f"Sample {i+1} French: {' '.join(trg_text_i)}")

    break  # Chỉ kiểm tra một batch

print("=== Testing hoàn thành ===")

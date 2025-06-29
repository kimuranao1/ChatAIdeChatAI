import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sentencepiece as spm
import random
import os

# --- ハイパーパラメータ ---
seq_length = 10
embedding_dim = 128
hidden_dim = 256
batch_size = 16
num_epochs = 5
lr = 0.001

# --- SentencePiece読み込み ---
sp = spm.SentencePieceProcessor()
sp.load("mymodel2.model")
vocab_size = sp.get_piece_size()

# --- Attentionモジュール ---
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs):
        scores = self.attn(lstm_outputs)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_outputs, dim=1)
        return context, weights

# --- LSTM + Attention モデル ---
class LSTMAttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        context, attn_weights = self.attention(lstm_out)
        output = self.fc(context)
        return output, hidden, attn_weights

# --- モデル初期化 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMAttentionModel(vocab_size, embedding_dim, hidden_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# --- データセット作成関数 ---
def create_dataset(tokens, seq_length):
    dataset = []
    for i in range(len(tokens) - seq_length):
        seq = tokens[i:i + seq_length]
        target = tokens[i + seq_length]
        dataset.append((seq, target))
    return dataset

# --- top-pサンプリング関数 ---
def top_p_sampling(probs, p=0.9):
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = cumulative_probs > p
    if np.any(cutoff):
        cutoff_index = np.where(cutoff)[0][0] + 1
        sorted_probs = sorted_probs[:cutoff_index]
        sorted_indices = sorted_indices[:cutoff_index]
    sorted_probs = sorted_probs / np.sum(sorted_probs)
    return np.random.choice(sorted_indices, p=sorted_probs)

# --- 文章生成関数 ---
def generate_text(model, start_text, length=50, temperature=1.0, top_p_value=0.9):
    model.eval()
    start_tokens = sp.encode(start_text, out_type=int)
    if len(start_tokens) < seq_length:
        start_tokens = [0] * (seq_length - len(start_tokens)) + start_tokens
    else:
        start_tokens = start_tokens[-seq_length:]

    generated = start_tokens[:]
    input_seq = torch.tensor([start_tokens], dtype=torch.long).to(device)
    hidden = None

    for _ in range(length):
        with torch.no_grad():
            output, hidden, attn_weights = model(input_seq, hidden)
            probs = torch.softmax(output[0] / temperature, dim=0).cpu().numpy()
            next_token = int(top_p_sampling(probs, p=top_p_value))
            generated.append(next_token)
            input_seq = torch.tensor([generated[-seq_length:]], dtype=torch.long).to(device)

    return sp.decode(generated)

# --- 学習関数 ---
def train_model(model, tokens, epochs=3):
    dataset = create_dataset(tokens, seq_length)
    for epoch in range(epochs):
        random.shuffle(dataset)
        total_loss = 0
        for i in range(0, len(dataset) - batch_size, batch_size):
            batch = dataset[i:i+batch_size]
            seqs = torch.tensor([x[0] for x in batch], dtype=torch.long).to(device)
            targets = torch.tensor([x[1] for x in batch], dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs, hidden, attn_weights = model(seqs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

# --- モデル読み込み ---
if os.path.exists("model_weights.pth"):
    model.load_state_dict(torch.load("model_weights.pth", map_location=device))
    model.eval()
    print("✅ モデル重みを読み込みました。")

# --- メインメニュー ---
while True:
    print("\n--- モード選択 ---")
    print("1: 対話モード")
    print("2: 追加学習モード")
    print("3: モデル保存")
    print("4: 終了")
    mode = input("番号を入力してください: ")

    if mode == "1":
        while True:
            user_input = input("\nあなた（exitで戻る）: ")
            if user_input.lower() == "exit":
                break
            response = generate_text(model, user_input, length=50, temperature=1.0, top_p_value=0.9)
            print("AI: ", response)

    elif mode == "2":
        filename = input("追加学習用ファイル名（例: additional_corpus.txt）を入力: ")
        try:
            with open(filename, encoding="utf-8") as f:
                new_text = f.read()
            new_tokens = sp.encode(new_text, out_type=int)
            print("=== 追加学習開始 ===")
            train_model(model, new_tokens, epochs=3)
            print("=== 追加学習終了 ===")
        except FileNotFoundError:
            print("ファイルが見つかりませんでした。")

    elif mode == "3":
        torch.save(model.state_dict(), "model_weights.pth")
        print("✅ モデル重みを保存しました。")

    elif mode == "4":
        print("終了します。")
        break

    else:
        print("無効な番号です。再入力してください。")

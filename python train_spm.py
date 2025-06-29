import sentencepiece as spm

# 使いたいテキストファイル
input_file = "corpus.txt"

# 出力するモデルファイル名（拡張子は自動）
model_prefix = "mymodel2"

# 学習
spm.SentencePieceTrainer.Train(
    input=input_file,
    model_prefix=model_prefix,
    vocab_size= 50000,         # 語彙数（例: 8000程度が目安）
    model_type='bpe',        # モデルタイプ（bpe, unigram, char, word から選べる）
    character_coverage=0.9995,  # 文字カバレッジ率（日本語は0.9995推奨）
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

print("学習完了！")

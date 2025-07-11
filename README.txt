=====================================================
ChatAIが作ったChatAI - Custom LLM Chat Assistant
=====================================================

■ 概要
このソフトは、ユーザーが任意の文章を追加学習させ、独自の生成AIと対話できる実験的アプリケーションです。
内部では PyTorch と SentencePiece（Google開発）を使用し、LSTM + Attention モデルを用いています。

-----------------------------------------------------
■ 使い方

1. ChatAIが作ったChatAI.exe を実行
2. モード選択メニューが表示されます

  [1] 対話モード
      → あなたが入力した文章に基づいてAIが応答します。
      → "exit" と入力するとメニューに戻ります。

  [2] 追加学習モード
      → 任意のテキストファイルを指定して追加学習ができます。
      → 例: additional_corpus.txt にテキストを用意し、ファイル名を入力。

  [3] 終了
      → プログラムを終了します。

-----------------------------------------------------
■ 同梱ファイル

- mymodel.model            → SentencePieceモデル
- sample_corpus.txt        → 初期学習コーパス
- additional_corpus.txt    → 追加学習用サンプルファイル
- LICENSE.txt              → 利用許諾文
- your_script.py           → 参考用Pythonソース

-----------------------------------------------------
■ クレジット

This software was developed with assistance from ChatGPT (OpenAI).
© 2025 kimura

-----------------------------------------------------
■ ライセンス

MIT License（LICENSE.txt参照）

-----------------------------------------------------
■ 注意事項

- 学習に使用する文章には著作権に注意してください。
- 本ソフトウェアは実験的なものであり、生成する文章の内容には責任を負いません。
- 不具合などがあれば、自由に改善・改変してご利用ください。

*たった６エポック学習データ（自分の日記なんだが・・・）を齧らせただけなので、初期状態はほとんどまともに話すことができません。

＊追加学習を希望される場合、一回の学習エポック数は３に定められてます。（何回でも学習できます。）

=====================================================

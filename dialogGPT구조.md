# dialogGPT 구조

***

## 파일 구조

* |-> configs
  * |-> 117M
    * config.json : {"initializer_range": 0.02,"layer_norm_epsilon": 1e-05,"n_ctx": 1024,"n_embd": 768,"n_head": 12,"n_layer": 12,"n_positions": 1024,"vocab_size": 50257}
    * merges.txt
    * vocab.json : 50,257 토큰
  * |-> 345M
    * config.json 
    * merges.txt
    * vocab.json : 50,257 토큰
  * |-> 762M
    * config.json 
    * merges.txt
    * vocab.json : 50,257 토큰
* |-> data
* |-> dstc
* |-> gpt2_training
* |-> lsp_model
* |-> pycocoevalcap
* |-> reddit_extractor
* LSP-generic.yml
* LSP-linux.yml
* LSP-train.yml
* MANIFEST.in
* data_config.py
* data_loader.py
* demo.py
* demo_utils.py
* env.py
* prepropy

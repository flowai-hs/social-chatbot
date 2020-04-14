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
    
***

* |-> data : 학습용 데이터
  * dummy_data.tsv
  * prepare4db.sh : train_raw.tsv에서 train.tsv로
  * train_raw.tsv

***

* |-> dstc : DSTC7-End-to-End-Conversation-Modeling 평가를 하기위한 폴더
  * 생략

***

* |-> gpt2_training : 학습간 필요한 유틸 파일들
  * distributed.py : 프로세스간 클러스터 또는 병렬화 하기 위한 파일 
  * eval_utils.py : 평가를 하기 위한 유틸 함수들 e.g. BLEU 등
  * train_utils.py : 학습에 필요한 유틸 함수들 e.g. 문장의 길이를 맞춘다거나 인풋 피처를 정의하거나 등

***

* |-> lsp_model
  * modeling_gpt2.py : 파이토치 OpenAI GPT-2 model
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

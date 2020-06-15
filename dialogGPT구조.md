# dialogGPT 구조

## 파일 구조

* |-> **configs**
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

* |-> **data** : 학습용 데이터
    * dummy_data.tsv
    * prepare4db.sh : train_raw.tsv에서 train.tsv로
    * train_raw.tsv

* |-> **dstc** : DSTC7-End-to-End-Conversation-Modeling 평가를 하기위한 폴더
    * 생략
    
* |-> **gpt2_training** : 학습간 필요한 유틸 파일들
    * distributed.py : 프로세스간 클러스터 또는 병렬화 하기 위한 파일 
    * eval_utils.py : 평가를 하기 위한 유틸 함수들 e.g. BLEU 등
    * train_utils.py : 학습에 필요한 유틸 함수들 e.g. 문장의 길이를 맞춘다거나 인풋 피처를 정의하거나 등
 
* |-> **lsp_model**
    * modeling_gpt2.py : 파이토치 OpenAI GPT-2 model
    * optim.py : 최적화 관련 함수들 e.g. adam 등
    
* |-> **pycocoevalcap** : Automatic 평가용 파일
    * (폴더) blue, cider, metero, rouge, tokenizer, (파일) eval.py
    
* |-> **reddit_extractor** : reddit 데이터 전처리하는 파일들 e.g. 워드 블록을 하고 센텐스를 만들고 필요없는 기호 빼고 등
    * (폴더) config, data, lists, src, (파일) makefile
    
* LSP-generic.yml : 의존성 목록이나 pip설치 목록
* LSP-linux.yml : 리눅스 관련 의존성 목록이나 pip설치 목록
* **LSP-train.py** : GPT2 scratch/fine-tuning. 학습 실행 파일 huggingface GPT-2 구현 기반 수정
* data_config.py : 데이터 관련 설정, 데이터 위치 등
* data_loader.py : 데이터를 로드하는 파일
* **demo.py** 
* demo_utils.py
* env.py
* prepro.py : 데이터 프리프로세싱
* discord_bot.py : Discord bot
* interact.py : Test script
* run.sh : Training script
* run_eval.sh : Evaluation script
* bot.py : Code for easier test
* run_sample.sh - Test script to test for a few sentences
* sample.py - Test code to test for a few sentences

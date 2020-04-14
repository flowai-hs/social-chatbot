# koGPT2 구조

***

## 파일구조

* **imgs** : readme.md 소개 파일을 작성하기 위한 이미지 파일
* **kogpt2** : 메인폴더
    * **model** : 모델을 모아놓은 폴더
        * gpt.py : 6개의 모델 클래스 'GPT2Model', 'GPT2SelfAttentionLayer', 'GPT2FFNLayer', 'gpt2_117m', 'gpt2_345m' 
        * torch_gpt2 : 파이토치 OpenAI GPT-2 model, 외부 모듈 hugging face transformer를 가져와 사용함
    * **mxnet_kogpt2.py** : mxnet에서 구현된 kogpt2
    * **pytoch_kogpt2.py** : pytorch 기반으로 구현된 kogpt2
    * utils.py :  다운로드, 토크나이저 유틸
* requirement.txt : 라이브러리
* setup.py : 

***

## Model
* GPT-2 Base 모델
* 기본 설정
```
GPT2Model(units=768,
    max_length=1024,
    num_heads=12,
    num_layers=12,
    dropout=0.1,
    vocab_size=50000)
```


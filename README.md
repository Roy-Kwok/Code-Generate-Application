# GPT2-Code-Generate-Application


## 使用GPT2简单训练的一个代码生成器

所作的工作：
+ 配置软件环境
+ 从GitHub获取数据作为语料
+ 数据处理
+ 训练并使用模型


## 1、准备工作
	本文的语料库是从GitHub获取到的代码数据，所以我们需要从GitHub上下载大量的仓库。首先登录GitHub，然后在个人头像处点击Settings，接着在左侧找到Developer setting => Personal access tokens => Tokens(classic)获取一个新的token。将token保存到jupyter_notebook目录下的github_token.txt文件中。

| 软件环境        | 配置   |
| :--------:        | :-----:  |
| 操作系统        | Windows 10   |
| CUDA            |   12.2   |
| Python版本   |    3.8    |
| GPU              |    GeForece RTX 3070    |
|                       |    PyGithub    |
|                      |    tokenizers    |
| 基础库           |    transformers    |
|                      |    datasets    |
|                      |    Pytorch 1.12.0  |


## 2、数据处理
```python
NEWLINECHAR = '<N>'
MIN_STR_LEN = 256
MAX_STR_LEN = 512

count = 0
with open('sample_data.txt', 'a', encoding="UTF-8") as f:
    for full_dir in full_dirs:
        try:
            fd = open(full_dir, 'r', encoding="UTF-8").read()
            fd = fd.replace('\n', NEWLINECHAR)

            if 128 <= len(fd) <= MAX_STR_LEN:
                f.write(fd + '\n')
            else:
                substring = ''
                fd_split = fd.split(f'{NEWLINECHAR}{NEWLINECHAR}')
                for split in fd_split:
                    substring += split + f'{NEWLINECHAR}{NEWLINECHAR}'
                    if MIN_STR_LEN <= len(substring) <= MAX_STR_LEN:
                        f.write(substring + '\n')
                        substring = ''

        except Exception as e:
            print(str(e))
```

## 3、导入模型
```python
from transformers import GPT2Tokenizer
from transformers import GPT2Config
from transformers import GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("tokenizer")
tokenizer.add_special_tokens({
    "bos_token":"<s>",
    "pad_token":"<pad>",
    "eos_token":"</s>",
    "unk_token":"<unk>",
    "mask_token":"<mask>",    
})

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config)
```

## 4、使用模型
```python
model = GPT2LMHeadModel.from_pretrained('model_save').to('cuda')

while True:
    inp = input('>>> ')
    input_ids = tokenizer.encode(inp, return_tensors='pt').to('cuda')
    
    beam_output = model.generate(
        input_ids,
        max_length=512,
        num_beams=10,
        temperature=0.7,
        no_repeat_ngram_size=5,
        num_return_sequences=1,
    )
    
    for beam in beam_output:
        out = tokenizer.decode(beam)
        fout = out.replace('<N>','\n')
        print(str(fout))
```

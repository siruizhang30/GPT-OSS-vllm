# GPT-OSS inference with vllm

## Setup
### Using plain vllm:
```
conda create -n vllm-oss python=3.12 -y
conda activate vllm-oss

pip install --upgrade pip
pip install vllm==0.13.0
```

### Using `openai-harmony`:
```
conda create -n vllm-oss python=3.12 -y
conda activate vllm-oss-openai

pip install --upgrade pip

pip install https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl

pip install --pre vllm --extra-index-url https://wheels.vllm.ai/gpt-oss/

pip install openai-harmony
```
Reference link: 
- https://cookbook.openai.com/articles/gpt-oss/run-vllm#using-vllm-for-direct-sampling
- https://github.com/openai/openai-cookbook/issues/2330#issue-3767013034


## Usage
### Plain vllm:
One prompt inference:
```
python gpt-oss-vllm-plain.py
```

Multiple prompts batch inference:
```
python gpt-oss-vllm-plus.py
```

### `Openai-harmony` vllm:
One prompt inference:
```
python gpt-oss-harmony-plain.py
```

Multiple prompts batch inference:
```
python gpt-oss-harmony-plus.py
```
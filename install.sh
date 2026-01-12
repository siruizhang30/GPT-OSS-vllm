conda create -n vllm-oss python=3.12 -y
conda activate vllm-oss

pip install --upgrade pip

pip install https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl

pip install --pre vllm --extra-index-url https://wheels.vllm.ai/gpt-oss/

pip install openai-harmony

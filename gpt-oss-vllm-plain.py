import os
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

model_name = "openai/gpt-oss-20b"

llm = LLM(
    model=model_name,
    trust_remote_code=True,
    gpu_memory_utilization=0.8,
)

params = SamplingParams(
    max_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    skip_special_tokens=False
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

conversations = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Please tell me about the capital of France."},
]

prompts = [
    tokenizer.apply_chat_template(
        conv,
        tokenize=False,
        add_generation_prompt=True,
    )
    for conv in conversations
]


outputs = llm.generate(
    prompts=prompts, 
    sampling_params=params,
)

for output in outputs:
    print("Full output:", output)
    generated_text = output.outputs[0].text.split('<|channel|>final<|message|>')[-1].split('<|return|>')[0].strip()
    print(f"Generated text:\n {generated_text}")

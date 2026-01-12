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


def construct_prompt(system_prompt: str, user_prompt: list) -> str:

    conversations = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ] for prompt in user_prompt
    ]

    prompts = [
        tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True,
        )
        for conv in conversations
    ]
    return prompts


if __name__ == "__main__":
    prompt_list = [
        "Please tell me about the capital of France.",
        "When is the day longest during the year?",
        "Where is bigger, the moon or the sun?",
        "What is the weather like in SF?",
        "Explain the theory of relativity in simple terms.",
        "How does a blockchain work?",
        "What are the benefits of meditation?",
        "Describe the process of photosynthesis.",
        "What is quantum computing?",
        "How do vaccines work?"
    ]

    SYSTEM_PROMPT = "You are a helpful assistant."

    prompts = construct_prompt(system_prompt=SYSTEM_PROMPT, user_prompt=prompt_list)

    outputs = llm.generate(
        prompts=prompts, 
        sampling_params=params,
    )

    for output in outputs:
        print("Full output:", output)
        generated_text = output.outputs[0].text.split('<|channel|>final<|message|>')[-1].split('<|return|>')[0].strip()
        print(f"Generated text:\n {generated_text}")

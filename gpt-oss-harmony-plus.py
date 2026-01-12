import json
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)
 
from vllm import LLM, SamplingParams

import time
from tqdm import tqdm

# --- 1) Render the prefill with Harmony ---
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
 
# Harmony stop tokens (pass to sampler so they won't be included in output)
stop_token_ids = encoding.stop_tokens_for_assistant_actions()
 
# --- 2) Run vLLM with prefill ---
llm = LLM(
    model="openai/gpt-oss-20b",
    trust_remote_code=True,
    gpu_memory_utilization=0.8,
)
 
sampling = SamplingParams(
    max_tokens=1024,
    temperature=0.6,
    stop_token_ids=stop_token_ids,
    top_p=0.9,
)

def vllm_single_function(prompt, system_prompt):
    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
            Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions(system_prompt),
            ),
            Message.from_role_and_content(Role.USER, prompt),
        ]
    )
    
    prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

    outputs = llm.generate(
        prompts=[{"prompt_token_ids": prefill_ids}],
        sampling_params=sampling,
    )
    
    # vLLM gives you both text and token IDs
    gen = outputs[0].outputs[0]
    text = gen.text
    output_tokens = gen.token_ids  # <-- these are the completion token IDs (no prefill)
    
    # --- 3) Parse the completion token IDs back into structured Harmony messages ---
    entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
    
    # 'entries' is a sequence of structured conversation entries (assistant messages, tool calls, etc.).
    final_message = "sorry"
    for message in entries:
        print(f"{json.dumps(message.to_dict())}")
        if message.to_dict().get("channel") == "final":
            final_message = message.to_dict()["content"][0]["text"]

    print("Final generated caption:", final_message)

    return final_message


def vllm_batch_function(prompt, system_prompt):
    prompts_list = []
    for meta_prompt in prompt:
        convo = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
                Message.from_role_and_content(
                    Role.DEVELOPER,
                    DeveloperContent.new().with_instructions(system_prompt),
                ),
                Message.from_role_and_content(Role.USER, meta_prompt),
            ]
        )
        prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        prompts_list.append({"prompt_token_ids": prefill_ids})

    outputs = llm.generate(
        prompts=prompts_list,
        sampling_params=sampling,
    )
    print(f"Generated {len(outputs)} outputs for {len(prompt)} prompts.")
    genereated_captions = []
    for i in range(len(prompt)):
        # vLLM gives you both text and token IDs
        gen = outputs[i].outputs[0]
        text = gen.text
        output_tokens = gen.token_ids  # <-- these are the completion token IDs (no prefill)
        
        # --- 3) Parse the completion token IDs back into structured Harmony messages ---
        entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
        
        # 'entries' is a sequence of structured conversation entries (assistant messages, tool calls, etc.).
        final_message = "sorry"
        for message in entries:
            # print(f"{json.dumps(message.to_dict())}")
            if message.to_dict().get("channel") == "final":
                final_message = message.to_dict()["content"][0]["text"]
        
        genereated_captions.append(final_message)

        print("Final generated caption:\n", final_message)

    return genereated_captions


def single_inference(prompt_list, system_prompt):
    generated_contents = []
    for prompt in tqdm(prompt_list):
        max_tries = 5
        while max_tries > 0:
            generated_content = vllm_single_function(prompt, system_prompt)
            if "sorry" in generated_content:
                print("Regenerating due to incomplete response...")
                max_tries -= 1
                continue
            else:
                generated_contents.append(generated_content)
                break


if __name__ == "__main__":

    SYSTEM_PROMPT = "You are a helpful assistant."

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
    start_time = time.time()

    # + ---- single prompt generation with retries---------------------------- +
    # generated_contents = []
    # for prompt in tqdm(prompt_list):
    #     max_tries = 5
    #     while max_tries > 0:
    #         generated_content = vllm_single_function(prompt, SYSTEM_PROMPT)
    #         if "sorry" in generated_content:
    #             print("Regenerating due to incomplete response...")
    #             max_tries -= 1
    #         else:
    #             generated_contents.append(generated_content)
    #             break
    #     if max_tries == 0:
    #         print(f"Failed to generate a valid response for prompt: {prompt}")
    #         generated_contents.append(generated_content)
        
    # print("Generated contents from single prompt generation:")
    # for content in generated_contents:
    #     print(content)
    # - ---- end of single prompt generation with retries--------------------- -

    # + ---- batch prompt generation with retries----------------------------- +
    generated_contents = vllm_batch_function(prompt_list, SYSTEM_PROMPT)
    for i in range(len(prompt_list)):
        generated_content = generated_contents[i]
        if "sorry" in generated_content:
            print("Regenerating batch due to incomplete response...")
            max_tries = 5
            while max_tries > 0:
                generated_content = vllm_single_function(prompt, SYSTEM_PROMPT)
                if "sorry" in generated_content:
                    print("Regenerating due to incomplete response...")
                    max_tries -= 1
                else:
                    generated_contents[i] = generated_content
                    break
            if max_tries == 0:
                print(f"Failed to generate a valid response for prompt: {prompt}")
                generated_contents[i] = generated_content
    print("Generated contents from batch prompt generation:")
    for content in generated_contents:
        print(content)
    # - ---- end of batch prompt generation with retries---------------------- -
            

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
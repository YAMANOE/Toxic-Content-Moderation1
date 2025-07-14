from huggingface_hub import InferenceClient

client = InferenceClient(model="meta-llama/LlamaGuard-7b", token="hf_your_token_here")

response = client.text_generation("your input prompt here", max_new_tokens=200)

print(response)

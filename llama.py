import os
from huggingface_hub import snapshot_download

token = os.getenv("HUGGINGFACE_TOKEN")

if token is None:
    raise ValueError("❌ لم يتم العثور على توكن Hugging Face في متغيرات البيئة")

model_path = snapshot_download(
    repo_id="meta-llama/LlamaGuard-7b",
    use_auth_token=token
)

print("✔ النموذج تم تحميله في:", model_path)

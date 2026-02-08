"""
Deployment script executed by GitHub Actions
"""

from huggingface_hub import upload_file

print("Uploading model to Hugging Face Hub...")

upload_file(
    path_or_fileobj="ci_model.pkl",
    path_in_repo="ci_model.pkl",
    repo_id="meghz0110/wellness-tourism-model",
    repo_type="model"
)

print("Model deployment completed.")

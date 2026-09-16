import mlstac
from pathlib import Path

model_dir = Path("models/SEN2SRLite_RGBN")
model_dir.mkdir(parents=True, exist_ok=True)

print("Downloading SEN2SRLite RGBN x4 model...")

mlstac.download(
    file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json",
    output_dir=str(model_dir),
)

print("\nModel downloaded successfully!")
print("Location:", model_dir)
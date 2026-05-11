"""
download_model.py
Downloads the LoRA adapter from S3 before starting the app.
Run once: python download_model.py
"""
import boto3
import os

BUCKET      = "opinion-mining-artifacts"
S3_PREFIX   = "model-artifacts/"
LOCAL_DIR   = "./final_adapter"

def download_adapter():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    s3 = boto3.client("s3")

    print(f"📥 Downloading adapter from s3://{BUCKET}/{S3_PREFIX}")
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=S3_PREFIX)

    if "Contents" not in response:
        print("❌ No files found in S3! Make sure you uploaded the adapter files.")
        return False

    for obj in response["Contents"]:
        key      = obj["Key"]
        filename = os.path.basename(key)
        if not filename:
            continue
        local_path = os.path.join(LOCAL_DIR, filename)
        print(f"  ⬇️  {filename} ({obj['Size'] / 1024:.1f} KB)")
        s3.download_file(BUCKET, key, local_path)

    print(f"\n✅ Adapter downloaded to {LOCAL_DIR}/")
    print(f"📁 Files: {os.listdir(LOCAL_DIR)}")
    return True

if __name__ == "__main__":
    download_adapter()

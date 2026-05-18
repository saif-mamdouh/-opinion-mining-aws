# Opinion Mining — Aspect-Based Sentiment Analysis

Fine-tuned LLaMA 3.2-3B with LoRA for aspect-based sentiment analysis, deployed on AWS EC2.

## Live Demo
http://ec2-3-92-59-253.compute-1.amazonaws.com:7860

## Architecture
- Model: LLaMA 3.2-3B + LoRA adapter (trained on Kaggle)
- Storage: AWS S3 (model artifacts)
- Compute: AWS EC2 t3.xlarge
- Monitoring: AWS CloudWatch

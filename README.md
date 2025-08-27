# Gemma-2-2b Fine-tuning with LoRA on English Quotes Dataset

This project demonstrates how to fine-tune Google's Gemma-2-2b language model using LoRA (Low-Rank Adaptation) technique on an English quotes dataset. The implementation uses 4-bit quantization for memory efficiency and runs on Google Colab.

## Overview

The code fine-tunes the Gemma-2-2b model to generate quotes in a specific format, training it to complete quote patterns and associate them with authors. This approach allows for efficient fine-tuning with minimal computational resources while maintaining model quality.

## Features

- **4-bit Quantization**: Uses BitsAndBytesConfig for memory-efficient training
- **LoRA Fine-tuning**: Implements Low-Rank Adaptation for parameter-efficient training
- **Supervised Fine-tuning**: Uses SFTTrainer from the TRL library
- **Quotes Dataset**: Trains on the "Abirate/english_quotes" dataset
- **GPU Optimized**: Configured for CUDA GPU acceleration



## Dataset

The project uses the "Abirate/english_quotes" dataset, which contains:
- Quote text
- Author information
- Various inspirational and famous quotes


## Expected Results

After fine-tuning, the model should be able to:
- Complete quote prompts more coherently
- Generate text in the expected quote format
- Show improved performance on quote-related tasks


## License

This project uses the Gemma-2-2b model, which is subject to Google's licensing terms. Please review the model card and licensing information before commercial use.


## Acknowledgments

- Google for the Gemma-2-2b model
- Hugging Face for the transformers library
- The TRL library for supervised fine-tuning tools
- Abirate for the English quotes dataset

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model + tokenizer from your folder
model_path = "./fine_tuned_gemma_quotes" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

st.title("AI Quote Generator ✨")
user_input = st.text_input("Enter the beginning of a quote:")

if st.button("Generate Quote"):
    # Move inputs to the same device as model
    inputs = tokenizer(user_input, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=30)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.success("Generated Quote: " + result)

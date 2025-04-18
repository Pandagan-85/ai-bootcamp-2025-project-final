import streamlit as st
import torch

st.title("Torch Compatibility Test")

st.write(f"Torch version: {torch.__version__}")

if st.button("Test Tensor Creation"):
    try:
        tensor = torch.rand(3, 3)
        st.write(f"Tensor: {tensor}")
        st.success("Torch is working within Streamlit!")
    except Exception as e:
        st.error(f"Error: {e}")

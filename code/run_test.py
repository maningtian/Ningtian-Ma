import numpy
import pandas
import yfinance
import gluonts
import torch
import transformers
import tqdm
import langchain
import langchain_community
import langgraph
import langchain_nvidia_ai_endpoints

if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Print the number of CUDA devices
        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices: {num_devices}")
    
        # Print details about each CUDA device
        for i in range(num_devices):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"\tMemory Allocated: {torch.cuda.memory_allocated(i) / 1e6} MB")
            print(f"\tMemory Cached: {torch.cuda.memory_reserved(i) / 1e6} MB")
    else:
        print("CUDA is not available.")
    
    print("\nAll imports successful. No dependency conflicts.")

# Portfolia - A Personalized Assistant for Financial Advising and Investment Management

Many people get into the world of finance and investment management and are taken aback by the sheer number of details and key words that they need to learn about when dealing with investments and stocks. What are ETF's? What are dividends? How can I diversify my portfolio? A lot of these questions can be answered with Portfolia, a personalized investment advisor.

## Description

Many people get into the world of finance and are blown away by the sheer number of details and key words that they need to learn about when dealing with investments and stocks. Like what are ETF's? What are dividends? A lot of these questions can be answered with Portfolia, a personalized investment advisor.

Portfolia also provides a feature for forecasting any given stock with high certainty. You can select a certain prediction length and Portfolia will forecast the stock and get back to you with its accurate predictions. If you would like to understand more about what happens under the hood, we leverage Time Series Transformers pre-trained on vast amounts of financial stock data for the S&P500 companies since 2004.

## Before Getting Started
In this repository, the folder `models` is too large for our Git LFS limit. Please click this `link` to download a zip file of `models` which you can manually add to this cloned repository.

The File Directory Structure of `models`:
```
┣ 📂 models
┃   ┗ 📜 .gitkeep
┃   ┣ 📂 stockformer
┃   ┃   ┗ 📂 sp500-5d-final
┃   ┃   ┃   ┗ 📜 sp500-5d-final-config.pth
┃   ┃   ┃   ┗ 📜 sp500-5d-final-state.pth
┃   ┃   ┗ 📂 sp500-15d-final
┃   ┃   ┃   ┗ 📜 ...
┃   ┃   ┃   ┗ 📜 ...
┃   ┃   ┗ 📂 ...
┃   ┃   ┗ 📂 ...
┃   ┃   ┗ 📂 ...
┃   ┃   ┗ 📂 sp500-180d-final
┃   ┃   ┃   ┗ 📜 ...
┃   ┃   ┃   ┗ 📜 ...
┃   ┃   ┗ 📂 sp500-360d-final
┃   ┃   ┃   ┗ 📜 sp500-360d-final-config.pth
┃   ┃   ┃   ┗ 📜 sp500-360d-final-state.pth
```

Due to unforeseen issues with the `nvwb start jupyterlab` Docker container not recognizing GPUs on our local machine during development, we decided to create our own custom Docker image which can recognize CUDA devices on the host machine. Details for running this application can be found below.

## Getting Started

1) Build our docker image with the Dockerfile present in the root directory of this repository.

**Note: This may take ~5 minutes to build*

```bash
docker build -t portfolia .
```

2) Start a container and make sure to set environment variables to API keys for HuggingFace Hub, NVIDIA NIMs, and Tavily. The following command will run a simple test to make sure everything went well.

You should see: `All imports successful. No dependency conflicts.`
```bash
docker run --rm -it -p 5000:5000 --gpus all -e HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN -e NVIDIA_API_KEY=$NVIDIA_API_KEY -e TAVILY_API_KEY=$TAVILY_API_KEY portfolia python3 code/run_test.py
```

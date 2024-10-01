# Portfolia - A Personalized Assistant for Financial Advising and Investment Management

## Table of Contents
- [Background & Motivation](#section-1)
- [Problem Statement](#section-2)
- [Stockformer](#section-3)
- [Features](#section-4)
- [Before Getting Started](#section-5)
- [Getting Started](#section-6)

<a id="section-1"></a>
## Background & Motivation
The financial and investment landscape can be daunting for newcomers due to the vast array of terminology, concepts, and investment strategies. Financial literacy and effective investment management are crucial skills in today's economy, yet many individuals struggle to navigate the complex world of finance. The abundance of financial jargon, investment options, and market volatility can be overwhelming for novice and experienced investors alike. Traditional financial advisory services are often expensive and inaccessible to the average person, creating a significant barrier to entry for those seeking to improve their financial well-being. The integration of Large Language Models (LLMs) and Machine Learning (ML) in personal finance offers a promising solution to democratize financial advice and empower individuals to make informed investment decisions.

<a id="section-2"></a>
## Problem Statement
Portfolia is an AI-powered personal investment advisor and stock forecasting assistant powered by LLM agents and Time Series Transformers. The primary objectives are:
1. To create an accessible and user-friendly platform that provides personalized financial advice and education to users of all experience levels.
2. To develop an accurate stock forecasting feature leveraging deep learning with time series transformers at various prediction lengths.
3. To enhance financial literacy by explaining complex financial concepts in simple, understandable terms.

<a id="section-3"></a>
## Stockformer: Time Series Transformer for S&P 500 Stock Forecasting 
Given the complexity of financial data, characterized by its inherent randomness, noise, and non-stationarity, extensive research has been conducted on leveraging deep learning-based representation learning for accurate stock price prediction. A recent publication, ["Transformers in Time Series: A Survey"](https://arxiv.org/abs/2202.07125) (Wen et al., 2023), presents a case study utilizing Time Series Transformers to forecast the Bangladeshi stock market in Dhaka. Building upon this research, we have developed and trained a proprietary Time Series Transformer model specifically for the U.S. S&P 500 stocks.

To read more about our implementation, please navigate to `code/stockformer` or just click [here](https://github.com/Jarhatz/portfolia/tree/main/code/stockformer#readme).
<a id="section-4"></a>
## Features
1. Agentic RAG (Retrieval Augmented Generation): Implement LLM agents for techniques to understand and respond to user queries about financial concepts and investment strategies.
2. Personalization: Utilize user-specific investor personality as context for providing tailored investment advice based on individual financial situations, goals, and risk tolerance.
3. Stock Forecasting: Employ Time Series Transformers pre-trained on historical S&P 500 stock data to generate accurate stock price predictions.
4. User Interface: Develop an intuitive, conversational interface that allows users to interact naturally with the AI advisor.
5. Scalable Back-End: Build a system which enables horizontal scalability by allowing concurrent request handling for stock forecasting during inference.

<a id="section-5"></a>
## Before Getting Started
In this repository, the folder `models` is too large for our Git LFS limit. Please click this [link](https://google.com/) to download a zip file of `models` which you can manually add to this cloned repository. **Note: There should be a total of 4 model checkpoint directories under `models/stockformer`.*

The File Directory Structure of `models`:
```
┣ 📂 models
┃   ┗ 📜 .gitkeep
┃   ┣ 📂 stockformer
┃   ┃   ┗ 📂 sp500-30d-final
┃   ┃   ┃   ┗ 📜 sp500-30d-final-config.pth
┃   ┃   ┃   ┗ 📜 sp500-30d-final-state.pth
┃   ┃   ┗ 📂 sp500-90d-final
┃   ┃   ┃   ┗ 📜 sp500-90d-final-config.pth
┃   ┃   ┃   ┗ 📜 sp500-90d-final-state.pth
┃   ┃   ┗ 📂 sp500-180d-final
┃   ┃   ┃   ┗ 📜 sp500-180d-final-config.pth
┃   ┃   ┃   ┗ 📜 sp500-180d-final-state.pth
┃   ┃   ┗ 📂 sp500-360d-final
┃   ┃   ┃   ┗ 📜 sp500-360d-final-config.pth
┃   ┃   ┃   ┗ 📜 sp500-360d-final-state.pth
```

<a id="section-6"></a>
## Getting Started

1. Clone the repository
```bash
git clone https://github.com/Jarhatz/portfolia.git
```

2. Build the docker image with the Dockerfile present in the root directory of this repository.
```bash
docker build -t portfolia .
```
> **Note: This may take ~5 minutes to build*

3. Start a container and run a simple dependency test. The following command will run the test to make sure everything went well.
```bash
docker run --rm -it --gpus all portfolia python3 code/run_test.py
```
> You should see: `All imports successful. No dependency conflicts.`

4. Start the server in a container. It will be listening on port 5000 for any incoming HTTP API requests.
```bash
docker run --rm -it -p 5000:5000 --gpus all portfolia
```

## Start the Front-End
Once the back-end server is up and running, please go on and launch the front-end client web application. Instructions can be found in our [portfolia-client](https://github.com/Jarhatz/portfolia-client) repository.

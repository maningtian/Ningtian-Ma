# Portfolia - A Personalized Assistant for Financial Advising and Investment Management

The financial and investment landscape can be daunting for newcomers due to the vast array of terminology, concepts, and investment strategies. Financial literacy and effective investment management are crucial skills in today's economy, yet many individuals struggle to navigate the complex world of finance. The abundance of financial jargon, investment options, and market volatility can be overwhelming for novice and experienced investors alike. Traditional financial advisory services are often expensive and inaccessible to the average person, creating a significant barrier to entry for those seeking to improve their financial well-being. The integration of Large Language Models (LLMs) and Machine Learning (ML) in personal finance offers a promising solution to democratize financial advice and empower individuals to make informed investment decisions.

## Description

Many people get into the world of finance and are blown away by the sheer number of details and key words that they need to learn about when dealing with investments and stocks. Like what are ETF's? What are dividends? A lot of these questions can be answered with Portfolia, a personalized investment advisor.

Portfolia also provides a feature for forecasting any given stock with high certainty. You can select a certain prediction length and Portfolia will forecast the stock and get back to you with its accurate predictions. If you would like to understand more about what happens under the hood, we leverage Time Series Transformers pre-trained on vast amounts of financial stock data for the S&P500 companies since 2004.

## Before Getting Started
### ***Note: This is the back-end repository. The front-end repository is [here](https://github.com/Jarhatz/portfolia-client).**

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

Due to unforeseen issues with the `nvwb start jupyterlab` Docker container not recognizing GPUs on our local machine during development, we decided to create our own custom Docker image which can recognize CUDA devices on the host machine. Details for running this application can be found below.

## Getting Started

1) Build the docker image with the Dockerfile present in the root directory of this repository.
```bash
docker build -t portfolia .
```
**Note: This may take ~5 minutes to build*

2) Start a container and run a simple dependency test. The following command will run the test to make sure everything went well.
```bash
docker run --rm -it --gpus all portfolia python3 code/run_test.py
```
You should see: `All imports successful. No dependency conflicts.`

3) Start the server in a container. It will be listening on port 5000 for any incoming HTTP API requests.
```bash
docker run --rm -it -p 5000:5000 --gpus all portfolia
```

## Start the Front-End
Once the back-end server is up and running, feel free to launch the front-end client web application. Instructions can be found in our [portfolia-client](https://github.com/Jarhatz/portfolia-client) repository.

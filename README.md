# Portfolia - A Personalized Assistant for Financial Advising and Asset Management

Many people get into the world of finance and are blown away by the sheer number of details and key words that they need to learn about when dealing with investments and stocks. Like what are ETF's? What are dividends? A lot of these questions can be answered with Portfolia, a personalized investment advisor. 

## Description

Many people get into the world of finance and are blown away by the sheer number of details and key words that they need to learn about when dealing with investments and stocks. Like what are ETF's? What are dividends? A lot of these questions can be answered with Portfolia, a personalized investment advisor.

Portfolia also provides a feature for forecasting any given stock with high certainty. You can select a certain prediction length and Portfolia will forecast the stock and get back to you with its accurate predictions. If you would like to understand more about what happens under the hood, we leverage Time Series Transformers pre-trained on vast amounts of financial stock data for the S&P500 companies since 2004.

## Getting Started

We use a custom Docker image for this project:

`docker build -t portfolia .`

To run the container using the image from above:

`docker run --gpus all --rm -it portfolia python3 code/test.py`

Optional section to summarize important steps and how to use the project & apps in the project

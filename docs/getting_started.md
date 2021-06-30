## Scraping Subreddits

To scrape posts from subreddits, you have to first put your reddit app's `client_id`, `client_secret` and `user_agent` in `configs/scraper.yaml`. Here is a [blog post on the Reddit API](https://www.jcchouinard.com/get-reddit-api-credentials-with-praw/), in case you don't know what they are.

To change what subreddit to scrape, change the field `subreddit_names` in `configs/config.yaml`.

Then, run the following command,

```
python scripts/scrape.py
```

By default, the bot will scrape from r/statistics, r/datascience and r/MachineLearning. A maximum of 1,000 posts will be scraped for each subreddit. Posts that only have an external link will be excluded. The scraped data will be stored in `data/raw/reddit_posts.csv`.

## Exploratory Data Analysis

To get a sense of what the data look like, you can use the code in `notebooks/01-exploratory_data_analysis.ipynb`.

## Data Preprocessing

After collecting the data, we should do some preprocesing before the modeling, by using the following command:

```
python scripts/preprocess_data.py
```

The script will

- Clean the text
    - Remove tags (e.g., [P], [R]) in the title
    - Concatenate title and selftext
    - Remove URLs
    - Replace punctuation with space
    - Transform multiple spaces and `\n` to a single space
    - Strip white spaces at the beginning and at the end
    - Transform to lowercase
- Convert subreddit names to integers
- Split data into train, validation and test sets (`train.csv`, `val.csv` and `test.csv` in `data/processed`)

## Baseline Models

Before trying neural network models on our data, it is often a good idea to use some simple models to establish a baseline performance.

The notebook `notebooks/02-baseline_models.ipynb` demonstrates how to do so by combining term frequency-inverse document frequency (TF-IDF) with navie bayes, logistic regression, k nearest neighbor, random forest and other traditional machine learning algorithms.

## Neural Networks

Two neural network models are available, 1D Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN).

Use the following command to train a CNN,

```
python scripts/train.py model=CNN
```

and the following to train a RNN,

```
python scripts/train.py model=RNN
```

You can use a Weights & Biases logger to track the experiment.

```
python scripts/train.py model=CNN logger=wandb
```

To perform hyperparameter tuning,

```
python scripts/train.py -m model=CNN hparams_search=cnn_optuna
```

If you want to train on Google Colab, check out `notebooks/03-neural_networks.ipynb`.

## Download artifacts

If no logger is used, the best model checkpoint and the tokenizer configurations are saved in the `outputs` folder by default.

If you use Google Colab to train your model and use a Weights & Biases logger, the artifacts are still saved in `outputs` of your colab environment, but are also uploaded to their platform at the end of training automatically. The following command can download the artifacts in case you lose the local copies:

```
python scripts/download_artifacts.py RUN_PATH
```

Replace RUN_PATH with the path of the run that you want to download the artifacts from. For example, the run path of my best run is `kingyiusuen/reddit-post-classification/2pzfr3t3`.

## RESTful API

To start the API,

```
uvicorn backend.api:app --host 0.0.0.0 --port 5000 --reload
```

Go to [https://0.0.0.0:5000/docs](https://0.0.0.0:5000/docs) for the documentation of the API.

## Frontend

Note that the frontend is built with React.js, so Node.js is required for it to run.

```
cd frontend
npm start
```

# Reddit Post Classification

It can be tricky to find the right subreddit to submit your post when your post is about statistics, machine learning and data science, as there is a great deal of overlap between them.

In this project, I scraped a total of around 2,600 posts from [r/statistics](https:/www.reddit.com/r/statistics), [r/MachineLearning](https:/www.reddit.com/r/MachineLearning) and [r/datascience](https://www.reddit.com/r/datascience), trained a classifer to predict the subreddit of the posts, and built a data product that suggests which subreddit you should post to.

The app is deployed to [Heroku](https://reddit-post-classifer.herokuapp.com/). (Heroku apps go to a sleeping mode when it doesn't receive any request for about 30 mins, so it may take a minute or two to load up the frontend and another minute to wake the backend up to get the predictions).

What I learned along the way:

- [x] Building baseline models with sckit-learn
- [x] Building neural networks with PyTorch
- [x] Code organization with PyTorch Lightning and Hydra
- [x] Hyperparameter tuning with Optuna
- [x] Experiment tracking with Weights & Biases
- [x] Tesing with Pytest
- [x] Developing a RESTful API with FastAPI
- [x] Frontend development with React.js
- [x] Containerization with Docker
- [x] Deployment to Heroku
- [x] Code linting with black, flake8, isort and mydocstyle
- [x] Static type checking with mypy
- [x] Documentation generation with MkDocs
- [x] Continuous integration with pre-commit and Github Actions

## Results

A model combining TF-IDF and logistic regression already has a test accuracy of 0.8471 (see [notebooks/02-baseline_models.ipynb](notebooks/02-baseline_models.ipynb)).

CNN does a little better with a test accuracy of 0.8595, which took 11 epochs and 39 seconds to train (see the run on [Weights & Biases](https://wandb.ai/kingyiusuen/reddit-post-classification/runs/lu3h4mrn)).

I deployed the web app with the CNN model just for practice. The TF-IDF + logistic regression model is smaller in size but just as good.

## Quick Start

Download and install [Node.js](https://nodejs.org/en/). Clone the repositiory, create a virtual envrionment, and install dependencies.

```
git clone https://github.com/kingyiusuen/reddit-post-classification.git
make venv
make install
```

Download the artifacts of the best run.

```
python scripts/download_artifacts.py kingyiusuen/reddit-post-classification/lu3h4mrn
```

Run the backend server.

```
uvicorn backend.api:app --host 0.0.0.0 --port 5000 --reload
```

Run the frontend. Your browser should automatically open [http://localhost:3000/](http://localhost:3000/).

```
cd frontend
npm start
```

For more details please refer to the [documentation](https://kingyiusuen.github.io/reddit-post-classification/).

## Possible Improvements

- [ ] Use larger models (pretrained word embeddings, BERT)
- [ ] Error analysis (see which subreddit get misclassified more often)
- [ ] Model monitoring (scrape data from the subreddits periodically, evaluate the model performance and re-train if necessary)

## Acknowledgements

- [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- [GokuMohandas/MLOps](https://github.com/GokuMohandas/MLOps)
# Reddit Post Classification

It can be tricky to find the right subreddit to submit your post when your post is about statistics, machine learning and data science, as there is a great deal of overlap between them.

In this project, I scraped a total of around 2,600 posts from [r/statistics](https:/www.reddit.com/r/statistics), [r/MachineLearning](https:/www.reddit.com/r/MachineLearning) and [r/datascience](https://www.reddit.com/r/datascience), trained a classifer to predict the subreddit of the posts, and built a data product that suggests which subreddit you should post to.

The app is deployed to [Heroku](https://reddit-post-classifer.herokuapp.com/). (Heroku apps go to a sleeping mode when it doesn't receive any request for about 30 mins, so it may take a minute or two to load up the frontend and another minute to connect to the backend to get the predictions).

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

## Possible Improvements

- [ ] Use larger models (pretrained word embeddings, BERT)
- [ ] Model monitoring (scrape data from the subreddits periodically, evaluate the model performance and re-train if necessary)
- [ ] Error analysis (see which subreddit get misclassified more often)

## Acknowledgements

- [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- [GokuMohandas/MLOps](https://github.com/GokuMohandas/MLOps)
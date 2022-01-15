# Reddit Post Classification

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/kingyiusuen/reddit-post-classification/blob/master/.pre-commit-config.yaml)
![Deployment status](https://github.com/kingyiusuen/reddit-post-classification/actions/workflows/deployment.yml/badge.svg)
[![License](https://img.shields.io/github/license/kingyiusuen/reddit-post-classification)](https://github.com/kingyiusuen/reddit-post-classification/blob/master/LICENSE)

It can be tricky to find the right subreddit to submit your post. For example, it is not uncommon to find people posted in [r/MachineLearning](https:/www.reddit.com/r/MachineLearning) when they should have posted in [r/LearningMachineLearning](https://www.reddit.com/r/learnmachinelearning) (which is for beginner's questions).

![](figures/reddit_screenshot.png)

It will be useful if there is a tool that

- helps new users post to the right subreddit, and
- allows subreddit moderators to easily identify posts that might not belong to their particular subreddit.

In this project, I used Python Reddit API Wrapper (PRAW) to scrape a total of around 1,500 posts from r/MachineLearning and r/LearningMachineLearning, and trained a classifer (TF-IDF + logistic regression) to predict the subreddit of the posts. The model has an AUC score of 0.7 in the test set. Finally, I built a web app that suggests which subreddit you should post to with React.js. The backend API was built with Flask, containerized with Docker and deployed to AWS ECS.

![Screenshot](figures/app_screenshot.png)

## Quick Start

Clone the repositiory, create a virtual envrionment, and install dependencies.

```
git clone https://github.com/kingyiusuen/reddit-post-classification.git
make venv
```

Start the backend server.

```
python backend/app.py
```

Start the frontend server.

```
cd frontend
npm start
```

To start scraping, follow the structure of `secrets.example.json` and create a `secrets.json` file at the project root directory. Fill out the necessary information.

### Docker

Build Docker image for backend API.

```
docker build --tag reddit-post-classifier --file backend/Dockerfile --platform linux/amd64 .
```

Run the Docker image as a container.

```
docker run -p 8080:8080 -it --rm reddit-post-classifier
```

Test the container with a POST request.

```
curl -XPOST "http://0.0.0.0:8080/predict" -H 'Content-Type: application/json' -d '{"text": "I love machine learning"}'
```

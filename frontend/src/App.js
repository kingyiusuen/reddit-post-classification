import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const API_ENDPOINT =
  "https://6xg6k6icxb.execute-api.us-east-1.amazonaws.com/default/predict-reddit-post";
const { REACT_APP_API_KEY } = process.env;

const App = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [predictions, setPredictions] = useState({});
  const [error, setError] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);
    setError("");
    const text = event.target.text.value;

    try {
      const res = await axios.post(
        API_ENDPOINT,
        { text },
        {
          headers: {
            "Content-Type": "application/json",
            "X-Api-Key": REACT_APP_API_KEY,
          },
        }
      );
      setPredictions(res.payload);
    } catch (error) {
      setError("Something went wrong");
    }
    setIsLoading(false);
  };

  return (
    <div className="wrapper">
      <div className="container">
        <img src="../reddit-icon.svg" alt="Reddit Icon" className="logo" />
        <h1>Reddit Post Classifier</h1>
        <p>
          Not sure whether your post should go to{" "}
          <a href="https://www.reddit.com/r/MachineLearning/">
            r/MachineLearning
          </a>{" "}
          or{" "}
          <a href="https://www.reddit.com/r/learnmachinelearning/">
            r/LearnMachineLearning
          </a>
          ? Put your post below and we will decide it for you!
        </p>

        <form onSubmit={handleSubmit}>
          <textarea
            placeholder="Put your post here"
            rows="6"
            defaultValue={defaultText}
            name="text"
          ></textarea>
          <div>
            <button type="submit" className="btn" disabled={isLoading}>
              {isLoading ? "Predicting..." : "Submit"}
            </button>
          </div>
        </form>
        <div className="predictions">
          {Object.keys(predictions).map((subreddit) => {
            return (
              <p>
                <span className="bold">{subreddit}: </span>
                {predictions[subreddit]}%
              </p>
            );
          })}
        </div>
        <div className="error">{error}</div>
      </div>
    </div>
  );
};

export default App;

const defaultText = `What are your hopes for Machine Learning in 2022?
I was just wondering what some of you are hoping ML can accomplish or overcome in this new year - interested in hearing your thoughts!`;

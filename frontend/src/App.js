import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const API_ENDPOINT = "http://ec2-52-55-85-90.compute-1.amazonaws.com:8080/predict";

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
      const res = await axios.post(API_ENDPOINT, { text });
      setPredictions(res.data.predictions);
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
                {Math.round(predictions[subreddit] * 100)}%
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

const defaultText = `Hi, I would like to know who are the current leading researchers/ research groups in AI Robotics and other in Theoretical Reinforcement Learning.`;

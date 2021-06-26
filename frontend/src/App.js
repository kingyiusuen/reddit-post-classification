import React, { Component } from "react";
import Header from "./Header";
import Form from "./Form";
import Results from "./Results";
import Footer from "./Footer";
import CssBaseline from "@material-ui/core/CssBaseline";
import Container from "@material-ui/core/Container";
import Snackbar from '@material-ui/core/Snackbar';

const containerStyle = {
  display: "flex",
  flexDirection: "column",
  minHeight: "100vh"
}

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      shouldHideForm: false,
      predictions: null,
      shouldShowError: false,
    };
  };

  handleSubmit = (e, inputTitle, inputSelfText) => {
    e.preventDefault();

    const params = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        "title": inputTitle,
        "selftext": inputSelfText
      })
    };
    fetch("http://127.0.0.1:8000/predict", params)
      .then(response => response.json())
      .then(response => response.data.predictions)
      .then(predictions => this.showResults(predictions))
      .catch(err => {this.setState({shouldShowError: true})});
  }

  showResults = (predictions) => {
    this.setState({
      shouldHideForm: true,
      predictions: predictions,
    });
  }

  hideResults = () => {
    this.setState({ shouldHideForm: false })
  }

  handleClose = () => {
    this.setState({ shouldShowError: false })
  }

  render() {
    return (
      <React.Fragment>
      <Container component="main" maxWidth="sm" style={containerStyle}>
        <CssBaseline />
        <Header />
        <Form
          shouldHide={this.state.shouldHideForm}
          handleSubmit={this.handleSubmit}
        />
        {this.state.shouldHideForm && <Results
          hideResults={this.hideResults}
          predictions={this.state.predictions}
        />}
        <Footer />
      </Container>
      <Snackbar
        autoHideDuration={6000}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        open={this.state.shouldShowError}
        onClose={this.handleClose}
        message="Server Connection Error"
      />
      </React.Fragment>
    );
  }
};

export default App;
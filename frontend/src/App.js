import React, { Component } from "react";
import Header from "./Header";
import Form from "./Form";
import Results from "./Results";
import Footer from "./Footer";
import CssBaseline from "@material-ui/core/CssBaseline";
import Container from "@material-ui/core/Container";
import Snackbar from '@material-ui/core/Snackbar';
import { withStyles } from '@material-ui/core/styles'

const styles = (theme) => ({
  container: {
    display: "flex",
    flexDirection: "column",
    minHeight: "100vh"
  }
});

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      shouldHideForm: false,
      shouldShowError: false,
      predictions: null,
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
    fetch("http://0.0.0.0:5000/predict", params)
      .then(response => response.json())
      .then(response => response.data.predictions)
      .then(predictions => this.showResults(predictions))
      .catch(err => { this.setState({ shouldShowError: true }) })
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

  handleCloseSnackBar = () => {
    this.setState({ shouldShowError: false })
  }

  render() {
    const { classes } = this.props;

    return (
      <React.Fragment>
      <Container component="main" maxWidth="sm" className={classes.container}>
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
        onClose={this.handleCloseSnackBar}
        message="Server Connection Error"
      />
      </React.Fragment>
    );
  }
};

export default withStyles(styles)(App);
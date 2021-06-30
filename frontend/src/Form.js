import React, { useState } from "react";
import TextField from "@material-ui/core/TextField";
import Button from "@material-ui/core/Button";
import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
  container: {
    width: "100%",
    marginTop: theme.spacing(1),
  },
  hidden: {
    display: "none"
  },
  button: {
    margin: theme.spacing(2, 0),
  },
  buttonDiv: {
    display: "flex",
    justifyContent: "center",
  },
}));

const Form = ({ shouldHide, handleSubmit }) => {
  const classes = useStyles();

  const [inputTitle, setInputTitle] = useState("");
  const [inputSelftext, setInputSelftext] = useState("");
  const updateInputTitle = e => {
    setInputTitle(e.target.value);
  };
  const updateInputSelftext = e => {
    setInputSelftext(e.target.value);
  };

  return (
    <form
      className={`classes.container ${shouldHide ? classes.hidden : ""}`}
      onSubmit={e => handleSubmit(e, inputTitle, inputSelftext)}
    >
      <TextField
        variant="outlined"
        margin="normal"
        fullWidth
        required
        name="title"
        label="Title"
        id="title"
        value={inputTitle}
        onChange={updateInputTitle}
      />
      <TextField
        variant="outlined"
        margin="normal"
        multiline
        rows={5}
        fullWidth
        name="text"
        label="Text (Optional)"
        id="text"
        value={inputSelftext}
        onChange={updateInputSelftext}
      />
      <div className={classes.buttonDiv}>
        <Button
          type="submit"
          variant="contained"
          color="primary"
          className={classes.button}
        >
          Submit
        </Button>
      </div>
    </form>
  )
}

export default Form;
import React from "react";
import Table from "@material-ui/core/Table";
import TableBody from "@material-ui/core/TableBody";
import TableCell from "@material-ui/core/TableCell";
import TableContainer from "@material-ui/core/TableContainer";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import Button from "@material-ui/core/Button";
import Typography from '@material-ui/core/Typography';
import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
  container: {
    width: "100%",
    marginTop: theme.spacing(1),
    padding: theme.spacing(0, 6),
  },
  tableHead: {
    fontWeight: 700,
  },
  button: {
    margin: theme.spacing(2, 0),
  },
  buttonDiv: {
    display: "flex",
    justifyContent: "center",
  },
}));

const Results = ({hideResults, predictions}) => {
  const classes = useStyles();

  return (
    <React.Fragment>
      <TableContainer elevation={0} className={classes.container}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>
                <Typography className={classes.tableHead}>Subreddit</Typography>
              </TableCell>
              <TableCell align="right">
                <Typography className={classes.tableHead}>Probability</Typography>
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {predictions.map((pred) => (
              <TableRow key={pred.subreddit}>
                <TableCell component="th" scope="row">
                  r/{pred.subreddit}
                </TableCell>
                <TableCell align="right">{pred.probability.toFixed(3)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <div className={classes.buttonDiv}>
        <Button
          variant="contained"
          color="primary"
          onClick={hideResults}
          className={classes.button}
        >
          Back
        </Button>
      </div>
    </React.Fragment>
  );
}


export default Results;

import React from "react";

import Avatar from "@material-ui/core/Avatar";
import RedditIcon from "@material-ui/icons/Reddit";
import Link from '@material-ui/core/Link';
import Typography from "@material-ui/core/Typography";
import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
  paper: {
    marginTop: theme.spacing(4),
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
  },
  avatar: {
    margin: theme.spacing(2),
    height: theme.spacing(8),
    width: theme.spacing(8),
    backgroundColor: "rgb(240, 88, 37)",
  },
}));

const Header = () => {
  const classes = useStyles();

  return (
    <div className={classes.paper}>
      <Avatar className={classes.avatar}>
        <RedditIcon fontSize="large" />
      </Avatar>
      <Typography variant="h4" align="center" color="textPrimary" gutterBottom>
        Reddit Post Classifer
      </Typography>
      <Typography variant="h6" align="center" color="textSecondary" component="p">
        Put a post here to decide whether it should belong to <Link href="https://www.reddit.com/r/datascience">r/datascience</Link>, <Link href="https://www.reddit.com/r/statistics">r/statistics</Link> or <Link href="https://www.reddit.com/r/MachineLearning">r/MachineLearning</Link>.
      </Typography>
    </div>
  )
}

export default Header;
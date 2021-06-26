import React from "react";

import Avatar from "@material-ui/core/Avatar";
import RedditIcon from "@material-ui/icons/Reddit";
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
        Subreddit Classifer
      </Typography>
      <Typography variant="h6" align="center" color="textSecondary" component="p">
        Finding the right subreddit to submit your post to? Put your post here to get some suggestions!
      </Typography>
    </div>
  )
}

export default Header;
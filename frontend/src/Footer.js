import React from "react";
import Link from "@material-ui/core/Link";
import Typography from "@material-ui/core/Typography";
import GitHubIcon from "@material-ui/icons/GitHub";
import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
  footer: {
    padding: theme.spacing(3, 2),
    marginTop: "auto",
  },
  icon: {
    color: "rgba(0, 0, 0, 0.38)",
    "&:hover": {color: "rgba(0, 0, 0, 0.54)"},
  },
}));

const Footer = () => {
  const classes = useStyles();

  return (
    <footer className={classes.footer}>
      <Typography variant="body2" align="center">
       <Link color="inherit" href="https://github.com/kingyiusuen">
        <GitHubIcon className={classes.icon}/>
       </Link>
      </Typography>
    </footer>
  );
}

export default Footer;
import React from "react";
import { Link } from "react-router-dom";


const Navbar = () => (
  <nav className="navbar">
    <ul className="navbar-list">
      <li><Link to="/" className="navbar-link">Home</Link></li>
      <li><Link to="/about" className="navbar-link">About</Link></li>
      <li><Link to="/tools" className="navbar-link">Tools</Link></li>
      <li><Link to="/upload" className="navbar-link">Upload</Link></li>
    </ul>
  </nav>
);

export default Navbar;

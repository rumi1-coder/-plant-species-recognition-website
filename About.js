// About.js

import React from 'react';
import { Link } from 'react-router-dom';
function About() {
  return (
    <div>
      <nav>
        <h1 className='logo'>Plant Species Identification</h1>
        <ul>
        <li><Link to="/">Home</Link></li>
          <li><Link to="/SpeciesIdentification">Identify a Plant Species</Link></li>
          <li><Link to="/PlantImages">View Plant Images</Link></li>
          <li><Link to="/PlantVideos">Watch Plant Videos</Link></li>
          
          <li><Link to="/contact">Contact Us</Link></li>
        </ul>
      </nav>
      <main>
        <h1>Welcome to our app!</h1>
        <p>We are trying to provide you the best experence ever:</p>
      
      </main>
      <footer>
        <p>© 2023 Plant Species Identification</p>
      </footer>
    </div>
  );
}

export default About;

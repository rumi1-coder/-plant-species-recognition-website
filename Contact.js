// Contact.js

import React from 'react';
import { Link } from 'react-router-dom';
import './contact.css'
function Contact() {
  return (
    <div>
      <nav>
        <h1 className='logo'>Plant Species Identification</h1>
        <ul>
        <li><Link to="/">Home</Link></li>
          <li><Link to="/SpeciesIdentification">Identify a Plant Species</Link></li>
          <li><Link to="/PlantImages">View Plant Images</Link></li>
          <li><Link to="/PlantVideos">Watch Plant Videos</Link></li>
         
          <li><Link to="/About">About</Link></li>
        </ul>
      </nav>
        <div className='contact-welcome'> 
        <h1>Conatct Us</h1>
        <p>This is your team conatct us any time and we will be their to help you thankyou</p>
        </div>
        <div className="contact-container">
        <div className="contact-info">
       
                <h2>Contact Information</h2>
                <ul>
            
              <li><i className="fas fa-map-marker-alt"></i>123 Main Street, Anytown USA</li>
              
              <li><i className="fas fa-phone"></i>(555) 555-5555</li>
              <li><i className="fab fa-facebook"></i>https://www.facebook.com/plantspeciesidentification</li>
               </ul>
          </div>
        
         
          <div className="email">
  <h2>Send us an email</h2>
  <form>
    <label htmlFor="email-input">Email address:</label>
    <input type="text" id="email-input" name="email" placeholder="Enter your email address" />
    <label htmlFor="message-input">Message:</label>
    <textarea id="message-input" name="message" placeholder="Enter your message" rows="10"></textarea>
    <button type="submit">SEND</button>
  </form>
</div>

        </div>
   
      <footer className='conatct-footer'>
        <p>© 2023 Plant Species Identification</p>
      </footer>
    </div>
  );
}

export default Contact;

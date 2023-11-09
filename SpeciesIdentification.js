// SpeciesIdentification.js

import React, { useState } from 'react';
import axios from 'axios';

function SpeciesIdentification() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleUpload = () => {
    const formData = new FormData();
    formData.append('file', file);
  
    axios.post('/upload', formData)
      .then(response => {
        setResult(response.data.flower_type);
      })
      .catch(error => {
        console.error(error);
      });
  };

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>
      {result && <p>Flower type: {result}</p>}
    </div>
  );
}

export default SpeciesIdentification;

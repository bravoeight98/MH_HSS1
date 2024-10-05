import logo from './logo.svg';
import './App.css';

import React, { useState } from 'react';
import WebcamCapture from './WebcamCapture';

const App = () => {
  const [image, setImage] = useState(null);
  const [hairstyleSuggestion, setHairstyleSuggestion] = useState('');

  const handleImageCapture = (imageSrc) => {
    setImage(imageSrc);
    fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image: imageSrc }),
    })
      .then(response => response.json())
      .then(data => setHairstyleSuggestion(data.hairstyle))
      .catch(error => console.error('Error:', error));
  };
  

  return (
    <div>
      <h1>Hairstyle Suggester</h1>
      <WebcamCapture handleImageCapture={handleImageCapture} />
      {image && <img src={image} alt="captured" />}
      {hairstyleSuggestion && <p>Suggested Hairstyle: {hairstyleSuggestion}</p>}
    </div>
  );
};

export default App;

import React, { useRef, useCallback } from 'react';
import Webcam from 'react-webcam';

const WebcamCapture = ({ handleImageCapture }) => {
  const webcamRef = useRef(null);

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    handleImageCapture(imageSrc);
  }, [webcamRef, handleImageCapture]);

  return (
    <div>
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width={350}
        videoConstraints={{
          facingMode: 'user',
        }}
      />
      <button onClick={capture}>Capture Photo</button>
    </div>
  );
};

export default WebcamCapture;

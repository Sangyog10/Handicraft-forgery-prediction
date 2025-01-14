import React, { useState } from "react";


const Tools = () => {
  const [clickedCard, setClickedCard] = useState(null);

  const handleClick = (cardName) => {
    setClickedCard(clickedCard === cardName ? null : cardName);
  };

  return (
    <div className="tools">
      <h2>Tools</h2>
      <div className="tools-cards">
        {/* Image Recognition Card */}
        <div
          className={`card ${clickedCard === "imageRecognition" ? "active" : ""}`}
          onClick={() => handleClick("imageRecognition")}
        >
          <h3>Image Recognition</h3>
          <p>
            Analyzes high-resolution images of handicrafts to verify authenticity. Detects
            subtle differences in materials, patterns, and designs.
          </p>
        </div>

        {/* Pattern Analysis Card */}
        <div
          className={`card ${clickedCard === "patternAnalysis" ? "active" : ""}`}
          onClick={() => handleClick("patternAnalysis")}
        >
          <h3>Pattern Analysis</h3>
          <p>
            Uses machine learning to identify unique artistic styles and techniques. Compares
            uploaded images with a database of genuine and counterfeit examples.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Tools;

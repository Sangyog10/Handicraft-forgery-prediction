import React, { useState, useEffect } from "react";
import "./index.css";
import AOS from "aos";
import "aos/dist/aos.css";

const App = () => {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    AOS.init({
      duration: 1300,
      once: false,
      mirror: true,
    });

    const handleScroll = () => {
      const sections = ["home", "about", "tools", "upload"];
      sections.forEach((section) => {
        const element = document.getElementById(section);
        const link = document.querySelector(`[onclick*="${section}"]`);
        if (
          element.offsetTop <= window.scrollY &&
          element.offsetTop + element.offsetHeight > window.scrollY
        ) {
          link.classList.add("active");
        } else {
          link.classList.remove("active");
        }
      });
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (id) => {
    document.getElementById(id).scrollIntoView({ behavior: "smooth" });
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImage(file);
      setPrediction(null);
    }
  };

  const removeImage = () => {
    setImage(null);
    setPrediction(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!image) {
      setError("Please upload an image first.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("image", image);

      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (response.ok) {
        setPrediction(data);
      } else {
        setError(data.error || "An error occurred. Please try again.");
      }
    } catch (err) {
      setError("Failed to connect to the server. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {/* Navbar */}
      <nav className="navbar">
        <div className="logo">
          <img src="/animated.svg" alt="Logo" className="logo-image" />
        </div>
        <ul className="navbar-list">
          <li>
            <button
              onClick={() => scrollToSection("home")}
              className="navbar-link"
            >
              Home
            </button>
          </li>
          <li>
            <button
              onClick={() => scrollToSection("about")}
              className="navbar-link"
            >
              About
            </button>
          </li>
          <li>
            <button
              onClick={() => scrollToSection("tools")}
              className="navbar-link"
            >
              Tools
            </button>
          </li>
          <li>
            <button
              onClick={() => scrollToSection("upload")}
              className="navbar-link"
            >
              Upload
            </button>
          </li>
        </ul>
      </nav>

      {/* Sections */}
      <div id="home" className="section" data-aos="fade-in">
        <div style={{ textAlign: "center", padding: "2rem" }}>
          {/* Add the animated logo */}
          <img
            src="/animated.svg"
            alt="Logo"
            className="animated-logo"
            style={{
              maxWidth: "350px", // Adjust size as needed
              height: "auto",
              marginBottom: "-1rem",
            }}
          />
          <h1>Arthentic AI</h1>
          <p>Authenticating Local Handicrafts: Preserving Culture with AI</p>
          <button
            onClick={() => scrollToSection("upload")}
            style={{
              padding: "1rem 1rem",
              backgroundColor: "#333",
              color: "#fff",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer",
            }}
          >
            Try Now
          </button>
        </div>
      </div>

      <div id="about" className="section" data-aos="fade-down">
        <div style={{ padding: "2rem", textAlign: "center" }}>
          <h1>About</h1>
          <p>
            Arthentic AI is a groundbreaking project created by SynEx, a team of
            forward-thinking innovators who are passionate about using
            artificial intelligence to preserve and authenticate cultural
            heritage. The core goal of Arthentic AI is to combat the growing
            issue of counterfeit artworks, particularly in the world of Thangka
            paintings and other valuable handmade artifacts. Through the use of
            advanced machine learning algorithms and cutting-edge image
            recognition technology, Arthentic AI enables users to verify the
            authenticity of Thangka paintings by analyzing intricate details,
            patterns, and materials that would be impossible for the human eye
            to detect. The AI is capable of distinguishing between authentic,
            hand-painted artwork and mass-produced, machine-made imitations,
            providing a reliable and swift authentication process. This project
            has the potential to revolutionize the art industry by offering an
            accessible and trustworthy tool for collectors, art dealers, and
            museums to safeguard the value of genuine artwork. Additionally, it
            serves as a beacon for the preservation of cultural heritage,
            ensuring that the rich tradition of handmade craftsmanship remains
            protected against the rise of counterfeit products in the market.
            The team behind Arthentic AI—SynEx—is driven by a shared vision of
            integrating technology and culture to protect and honor artistry,
            ensuring that the true essence of handcrafted artifacts remains
            undiluted for future generations.
          </p>
        </div>
      </div>

      <div id="tools" className="section" data-aos="fade-up">
        <div style={{ padding: "2rem", textAlign: "center" }}>
          <h1>Tools</h1>
          <p>
            Explore the tools we use to power our AI model and enhance the
            authentication process.
          </p>
          <div className="card-container">
            <div style={cardStyle}>
              <h3 style={{ marginBottom: "1rem" }}>Image Recognition</h3>
              <p>
                Analyzes high-resolution images of handicrafts to verify
                authenticity.
              </p>
              <p>
                Detects subtle differences in materials, patterns, and designs
                that are invisible to the human eye.
              </p>
            </div>
            <div style={cardStyle}>
              <h3 style={{ marginBottom: "1rem" }}>Pattern Analysis</h3>
              <p>
                Uses machine learning to identify unique artistic styles and
                techniques.
              </p>
              <p>
                Compares uploaded images with a database of genuine and
                counterfeit examples.
              </p>
            </div>
            <div style={cardStyle}>
              <h3 style={{ marginBottom: "1rem" }}>
                Customer Uploads High-Quality Photo of the Art
              </h3>
              <p>
                Users upload a clear, high-resolution image of their artwork.
                Our AI processes the image to extract intricate details and
                ensure accurate analysis.
              </p>
            </div>
            <div style={cardStyle}>
              <h3 style={{ marginBottom: "1rem" }}>
                Real-Time Verification: Handmade or Machine Made
              </h3>
              <p>
                The AI evaluates unique characteristics like brush strokes and
                material patterns in real-time, determining whether the artwork
                is handcrafted or machine-produced with precision.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div id="upload" className="section" data-aos="fade-down">
        <div
          className="upload-container"
          style={{ textAlign: "center", padding: "2rem" }}
        >
          <h1 className="upload-title">Upload Picture</h1>
          <p className="upload-description">
            Select a picture of a Thangka painting to begin the authentication
            process.
          </p>
          <label htmlFor="file-upload" className="upload-label">
            Choose a file
          </label>
          <input
            id="file-upload"
            type="file"
            accept="image/*"
            className="upload-input"
            onChange={handleImageUpload}
          />

          {image && (
            <div className="image-preview-container">
              <img
                src={URL.createObjectURL(image)}
                alt="Uploaded Preview"
                className="uploaded-image"
              />
            </div>
          )}

          {image && (
            <div className="button-container">
              <button
                onClick={handleSubmit}
                className="submit-button"
                disabled={loading}
              >
                {loading ? "Processing..." : "Submit to Verify"}
              </button>
              <button onClick={removeImage} className="remove-button">
                Remove
              </button>
            </div>
          )}

          {prediction && (
            <div className="prediction-result">
              <h3>Prediction Result:</h3>
              {/* <p>Score: {prediction.score}</p> */}
              <p>{prediction.result}</p>
            </div>
          )}

          {error && <div className="error-message">{error}</div>}
        </div>
      </div>
      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <h3>Contact Us</h3>
          <p>Email: info@arthenticai.com</p>
          <p>Phone: +123-456-7890</p>
          <p>Address: 123 Art Street, Kathmandu, Nepal</p>
        </div>
      </footer>
    </div>
  );
};

const cardStyle = {
  padding: "1.5rem",
  backgroundColor: "#fff",
  border: "1px solid #e0e0e0",
  borderRadius: "8px",
  boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
  maxWidth: "300px",
  textAlign: "left",
  lineHeight: "1.6",
};

export default App;

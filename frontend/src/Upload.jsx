import React, { useState } from "react";
import { IconUpload } from "@tabler/icons-react";

const Upload = () => {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
  };

  return (
    <div className="upload-container">
      <h2 className="upload-title">Upload Picture</h2>
      <p className="upload-description">
        Select a picture of a Thangka painting to begin the authentication process.
      </p>
      <div className="upload-input-container">
        <label htmlFor="file-upload" className="upload-label">
          <IconUpload className="upload-icon" />
          <span>{selectedFile ? selectedFile.name : "Choose a file"}</span>
        </label>
        <input
          id="file-upload"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="upload-input"
        />
      </div>
      {selectedFile && (
        <div className="file-details">
          <p>File Name: {selectedFile.name}</p>
          <p>File Size: {(selectedFile.size / 1024).toFixed(2)} KB</p>
          <p>File Type: {selectedFile.type}</p>
        </div>
      )}
    </div>
  );
};

export default Upload;

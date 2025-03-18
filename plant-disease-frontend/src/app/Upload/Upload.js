"use client";
import React, { useState } from 'react';
import ReactMarkdown from "react-markdown";

const Upload = () => {
  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [diseaseInfo, SetdiseaseInfo] = useState(null);

  const fileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile); 
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const predict = async () => {
    if (!file) {
      console.log("No file selected");
      return;
    }

    const formData = new FormData();
    formData.append("file", file); // Ensure the key matches FastAPI

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      setPrediction(data); // Store the disease name and confidence score
      console.log("Prediction:", data);

      searchdisease(data.disease);
    } catch (e) {
      console.log("Error:", e);
    }
  };

  const searchdisease = async (disease) => {
    try {
      console.log("hghghgds",disease)
        const response = await fetch("http://localhost:5000/api/get-disease-info", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ disease: disease })
        });
        const data = await response.json();
         // Extract the text response
         let aiResponse = data.candidates[0].content.parts[0].text;

         // Remove markdown symbols (# for headings, * for bullet points)
        //  aiResponse = aiResponse.replace(/[#*]+/g, "").trim(); 
         SetdiseaseInfo(aiResponse);
        console.log(data.candidates[0].content.parts[0].text);
    } catch (error) {
        console.error("Error:", error);
    }
};

  return (
    <div className="container-fluid">
      <div className="d-flex justify-content-center">
        <form onSubmit={(e) => {
          e.preventDefault();
          predict();
        }}>
          <input type="file" className="p-3" onChange={fileChange} />
          <button className="btn btn-outline-primary" type="submit">Predict</button>
        </form>
      </div>

      {imagePreview && (
        <div className="d-flex justify-content-center">
          <img src={imagePreview} alt="Uploaded Preview" style={{ width: "500px", height: "500px", border: "2px solid black" }} />
        </div>
      )}

      {prediction && (
        <div className="text-center mt-3">
          <h3>Prediction Result:</h3>
          <p><strong>Disease:</strong> {prediction.disease}</p>
          <p><strong>Confidence:</strong> {prediction.confidence.toFixed(2) * 100}%</p>
        </div>
      )}
      {
        diseaseInfo && (
          <>
          <div className='d-flex justify-content-center'> 
            <div className='w-75'>
              <ReactMarkdown>{diseaseInfo}</ReactMarkdown>
          {/* <p className='fs-3'>{diseaseInfo}</p> */}
            </div>
          </div>
          </>
        )
      }
    </div>
  );
};

export default Upload;

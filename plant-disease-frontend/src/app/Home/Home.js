import React from 'react'

const Home = () => {
    return (
        <div className='container-fluid'>
            <div className='d-flex justify-content-center'>
                <div>
                    <h1 >Detect Plant Diseases Instantly with AI!</h1>
                    <p className='d-flex justify-content-center'>Upload a leaf image and get instant disease predictions with solutions.</p>
                </div>
            </div>
            <div>
                <h1>Introduction :</h1>
                <p className='fs-3 ms-4'>Agriculture plays a vital role in feeding the world’s population, but plant diseases pose a significant threat to crop production. These diseases can be caused by bacteria, fungi, viruses, or environmental conditions, leading to reduced yields, poor-quality produce, and economic losses for farmers. Early detection and proper treatment are crucial to minimizing damage and ensuring healthy plant growth.</p>
                <p className='fs-3 ms-4'> &nbsp; &nbsp; &nbsp;To address this challenge, our AI-powered Plant Disease Prediction System provides an efficient and accessible solution. By leveraging advanced deep learning techniques, this system enables farmers, researchers, and agricultural experts to identify plant diseases quickly and accurately. With just a simple image upload, the system can analyze the leaf and provide an instant diagnosis, helping users take prompt action to prevent further crop damage.</p>
            </div>
            <div>
                <h1>About This Project:</h1>
                <p className='fs-3 ms-4'>Our AI-powered Plant Disease Prediction system helps farmers and researchers identify diseases from leaf images in seconds. By simply uploading a leaf image, the system analyzes it using a trained deep-learning model and provides a diagnosis along with suggested treatments.</p>
            </div>
            <div>
                <h1>How This Works:</h1>
                <ul className='fs-3 ms-4'>
                    <li>Upload a clear image of the affected plant leaf.</li>
                    <li>AI processes the image and detects possible diseases.</li>
                    <li> Get instant results with disease information and prevention tips.</li>
                </ul>
            </div>
        </div>

    )
}

export default Home

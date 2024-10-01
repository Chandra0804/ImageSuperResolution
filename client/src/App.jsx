import React, { useState } from 'react';
import { Upload, Image, ArrowRight, Loader, AlertCircle } from 'lucide-react';

const CustomAlert = ({ message }) => (
  <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mt-4" role="alert">
    <div className="flex items-center">
      <AlertCircle className="w-5 h-5 mr-2" />
      <span className="block sm:inline">{message}</span>
    </div>
  </div>
);

const ImageSuperResolution = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [enhancedImage, setEnhancedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    handleFile(droppedFile);
  };

  const handleFile = (file) => {
    if (file && file.type.substr(0, 5) === "image") {
      setFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      setError("Please select a valid image file.");
    }
  };

  const handleSubmit = async () => {
    if (!file) return;
  
    setIsLoading(true);
    setError(null);
  
    const formData = new FormData();
    formData.append('image', file);
  
    try {
      const response = await fetch('http://127.0.0.1:5000/super-resolve', {
        method: 'POST',
        body: formData,
      });
  
      if (!response.ok) {
        throw new Error('Enhancement failed');
      }
  
      // Parse the response as a Blob instead of JSON
      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setEnhancedImage(imageUrl); // Use the blob URL for displaying the enhanced image
  
    } catch (err) {
      console.log(err);
      setError("An error occurred while processing the image.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Landing Page */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">Image Super-Resolution</h1>
          <p className="text-xl text-gray-600">Enhance your images with cutting-edge AI technology</p>
        </div>

        {/* File Upload Component */}
        <div 
          className="max-w-xl mx-auto mb-8 p-6 bg-white rounded-lg shadow-md transition-all duration-300 ease-in-out hover:shadow-lg"
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
        >
          <div className="border-dashed border-2 border-gray-300 rounded-lg p-8 text-center">
            <Upload className="mx-auto h-12 w-12 text-gray-400" />
            <p className="mt-1 text-sm text-gray-600">Drag and drop your image here, or click to select a file</p>
            <input
              type="file"
              className="hidden"
              onChange={(e) => handleFile(e.target.files[0])}
              accept="image/*"
              id="fileInput"
            />
            <label
              htmlFor="fileInput"
              className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 cursor-pointer transition-colors duration-200"
            >
              Select File
            </label>
          </div>
        </div>

        {/* Preview and Enhanced Image Section */}
        {preview && (
          <div className="max-w-4xl mx-auto bg-white p-6 rounded-lg shadow-md">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h2 className="text-lg font-semibold mb-2">Original Image</h2>
                <img src={preview} alt="Preview" className="w-full h-auto rounded-lg" />
              </div>
              <div>
                <h2 className="text-lg font-semibold mb-2">Enhanced Image</h2>
                {enhancedImage ? (
                  <img src={enhancedImage} alt="Enhanced" className="w-full h-auto rounded-lg" />
                ) : (
                  <div className="w-full h-64 bg-gray-200 rounded-lg flex items-center justify-center">
                    <p className="text-gray-500">Enhanced image will appear here</p>
                  </div>
                )}
              </div>
            </div>
            <button
              onClick={handleSubmit}
              disabled={isLoading}
              className="mt-4 w-full inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
            >
              {isLoading ? (
                <Loader className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" />
              ) : (
                <ArrowRight className="-ml-1 mr-2 h-5 w-5" />
              )}
              {isLoading ? 'Processing...' : 'Enhance Image'}
            </button>
          </div>
        )}

        {/* Error Alert */}
        {error && <CustomAlert message={error} />}
      </div>
    </div>
  );
};

export default ImageSuperResolution;
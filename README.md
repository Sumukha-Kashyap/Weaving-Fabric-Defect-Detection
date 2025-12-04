**🧵 Weaving Fabric Defect Detection**
***A Computer Vision-Based Automated Fabric Quality Inspection System***

This project implements automated fabric defect detection using image preprocessing, denoising, thresholding, contour detection, and defect density computation.
It processes input fabric images, identifies defective regions, and visually highlights them using OpenCV.
The system also calculates a quantitative defect density score, useful for quality control in textile manufacturing.

**🚀 Features**

Converts images to HSV color space and extracts brightness channel

Noise reduction using blurring and Non-Local Means Denoising

Automatic OTSU thresholding to generate binary images

Morphological operations (erosion & dilation) to refine defect regions

Contour detection to highlight defects on fabric images

Computes Defect Density = Defect Area / Total Area

Displays:

Original image

Preprocessed binary image

Final image with highlighted defects

**🏗️ Project Workflow**
***1. Preprocessing***

Convert BGR → HSV

Extract Value (V) channel

Apply blur

Apply NLMeans denoising

OTSU thresholding to extract defect regions

***2. Postprocessing & Defect Highlighting***

Erosion → remove small noise

Dilation → strengthen defect structure

Contour detection based on area filtering

Red bounding contours drawn around defects

***3. Defect Density Calculation***
Defect Density = (Area of defect pixels) / (Total pixels)


This helps quantify defect severity.

**📂 Project Structure**
|-- INPUT_images/          # Folder containing input fabric images
|-- defect_detection.py    # Main detection and visualization script
|-- README.md              # Project documentation

**🔧 Installation**
***1. Clone Repository***
git clone https://github.com/Sumukha-Kashyap/Weaving-Fabric-Defect-Detection.git
cd Weaving-Fabric-Defect-Detection

***2. Install Dependencies***

Make sure you have Python 3.8+ installed.

pip install opencv-python numpy matplotlib

**▶️ How to Run**

***Place all fabric images inside the folder:***

INPUT_images/


***Then run:***

python defect_detection.py


***You will see:***

Original fabric image

Binary processed output

Final defect-highlighted visualization

Defect density printed on screen

**🖼️ Output Example**

The program displays images like:

Fabric Image

Processed (Binary) Image

Defects Highlighted

Defect Density: 0.xxxx

**🧠 Algorithms Used**
Step	Technique Used
Noise Reduction	Blurring, NLMeans
Segmentation	OTSU Thresholding
Morphological Ops	Erosion, Dilation
Defect Detection	Contours + Area Threshold
Metrics	Defect Density
**📌 Future Enhancements**

Training a Deep Learning model (CNN/Autoencoder) for defect classification

Adding a GUI dashboard

Exporting defect density reports

Real-time camera integration for production line inspection

**🤝 Contributions**

Contributions are welcome!
Feel free to submit issues or pull requests.

**📜 License**

This project is open-source and available under the MIT License.

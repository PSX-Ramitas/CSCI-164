# Incorporation of Facial Detection Systems in Identifying Cat Breeds
CSU Fresno - CSCI 164 - Fall Semester 2024

Identifying the breed of an animal is crucial for providing personalized care, both at home and at a veterinarian's office, and suitable living conditions. However, in many cases, only partial information, such as facial images, may be available. This project aims to revolutionize identification systems in animal breeds (specifically cats, in this case) by leveraging advanced machine learning techniques to make predictions with reduced information.  By incorporating haar-cascades and the Viola-Jones algorithm for facial feature extraction, the project explores how accurately cat breeds can be identified solely based on facial characteristics, rather than requiring features taken from the animal's full body.  The findings will contribute to the understanding of feature extraction techniques in image classification tasks, with possibilities of enhancing veterinary practices and animal welfare efforts by facilitating the identification of animals taken in for care.

## Objectives
- Develop machine learning models capable of accurately predicting cat breeds based on facial features.
- Investigate the effectiveness of the Viola-Jones algorithm in extracting relevant facial features for breed prediction.
- Assess the potential impact of breed prediction models on veterinary diagnosis, treatment planning, and animal shelter management.
- Demonstrate how reduced information can still be leveraged to identify a cat's breed to provide optimal care and support to cats in need.
    
## Dependencies
 - Python
 - PyTorch
 - OpenCV
 - scikit-learn
 - TorchVision
 - NumPy
 - matplotlib
 - Pillow
 - tqdm

## About Dataset
The dataset utilized is taken from [this Kaggle link here](https://www.kaggle.com/datasets/shawngano/gano-cat-breed-image-collection).

It contains 15 different folders labelled by the respective breed of cats:
- Abyssinian
- American Bobtail
- American Shorthair
- Bengal
- Birman
- Bombay
- British Shorthair
- Egyptian Mau
- Maine Coon
- Persian
- Ragdoll
- Russian Blue
- Siamese
- Sphynx
- Tuxedo

Within those folders, each breed has a total of 375 images, all of which are jpeg files.  Since our neural networks in `cats.ipynb` and `cats-vj.ipynb` are written from scratch and able to work only with images in jpeg format, then it was necessary to ensure that any and all images used to test the model after training are also in the same format.  In `cats.ipynb`, the dataset is preprocessed with a series of transformations from the torchvision library.  In `cats-vj.ipynb`, it also extracts facial features using the Viola-Jones algorithm for the face detection approach and crops the images based on the detected faces, such so that the model could be traiend purely off detected faces.  

The haar cascades utilized to train the Viola-Jones algorithm with cat faces are taken from OpenCV's library, found [here](https://github.com/opencv/opencv/tree/4.x/data/haarcascades).

## Methodology
1. **Data Collection and Preprocessing**: Gather a diverse dataset of cat images and annotate them with breed labels. Preprocess the dataset and extract facial features using the Viola-Jones algorithm.
2. **Model Development**: Train two separate machine learning models using both full-body images and extracted facial features. Experiment with different architectures and algorithms to optimize breed prediction accuracy. Compare the performance of both with a VGG16 model for benchmarking purposes.
3. **Evaluation**: Evaluate the performance of the trained models on a validation dataset, measuring metrics such as accuracy.
4. **Application**: Load the model after training and use it against one or several images, labelled `test.jpg` in our case, to evaluate its prediction.  Discuss how breed prediction models can be integrated into veterinary practices and animal shelters to improve decision-making processes and enhance overall care.

## Usage
To replicate the experiment for future developments and exploring the potential applications:
1. Clone this repository.
2. Install the required dependencies listed above.
3. Ensure you have a folder `CatBreeds` containing the dataset mentioned above, the haar cascades, as well as an input image labelled `test.jpg`, located in the same directory as both `cats.py` and `cats-vj.py`.
4. Run `cats.ipynb` to evaluate the model and its performance when acting on its own, and `cats-vj.ipynb` to evaluate the model when haar-cascades and Viola-Jones are incorporated as well.

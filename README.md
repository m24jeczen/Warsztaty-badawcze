# Research Project Part I: <br> Application of SparseConvMIL Model to Bisque Breast Cancer Data

## Project Description
This research project aims to apply the SparseConvMIL model to analyze breast cancer data from the Bisque Breast Cancer dataset. The model and information about its implementation can be found on the [GitHub page of the SparseConvMIL project](https://github.com/MarvinLer/SparseConvMIL). A detailed description of the model's stucture is provided in the article ["Sparse Convolutional Multiple Instance Learning for Whole Slide Breast Cancer Histology Image Classification"](https://proceedings.mlr.press/v156/lerousseau21a/lerousseau21a.pdf).

### Dataset
The Bisque Breast Cancer dataset contains data related to the segmentation of breast cancer cells. This dataset was published on Kaggle: [Breast Cancer Cell Segmentation](https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation/data). The original data is from the article ["Evaluation and benchmark for biological image segmentation"](https://vision.ece.ucsb.edu/sites/default/files/publications/elisa_ICIP08.pdf).

[Citation: E. Drelie Gelasca, J. Byun, B. Obara and B. S. Manjunath, "Evaluation and benchmark for biological image segmentation," 2008 15th IEEE International Conference on Image Processing, San Diego, CA, 2008, pp. 1816-1819.]

## Schedule and Work Organization
### MS 0:  (March 28)
1. **Repository Creation and Work Organization**: Creating a project repository on GitHub, assigning tasks, and responsibilities to team members.
2. **Data Analysis**: Review of the Bisque Breast Cancer dataset, understanding the data format, and (preliminarily) the structure of the SparseConvMIL algorithm.
3. **Preparation of Presentation**: Preparation of an initial presentation to discuss the obtained information and results of data analysis (5-10min).

### MS 1:
1. **Model Analysis**: Thorough familiarization with the structure and operation of the SparseConvMIL model. (All)
2. **Data Preparation**: Processing the data into an appropriate format for model application. (Zosia & Karolina)
3. **Model Implementation**: Implementation of the SparseConvMIL model according to the documentation and guidelines available on GitHub. (Magda)
4. **Model Training & Evaluation**: Training the model on the Bisque Breast Cancer dataset and assessing the model's performance on the test dataset. (All)
6. **Documentation and Report Preparation**: Analyzing the results, compiling a report with the project results, including technical documentation and interpretation of findings. (All)

The second part of the project remains without description for now and will be completed in a later phase of the project.

[UPDATE]
### MS 2:
In the next part of the project, we focus on developing strategies to improve the effectiveness of the model. We will base our research on 3 hypotheses:  
1. Areas marked with a mask facilitate the model in classifying histopathological images. (Karolina & Zosia)  
2. Areas containing background hinder the model in classifying histopathological images. (Zosia & Karolina)  
3. **Curriculum Learning**: Improves the efficiency/results of the model. (Magda)

   
    Three variants will be tested, dividing into difficulty levels based on **masks** and **background** (depending on the earlier findings) 

## Authors:
[Magdalena Jeczeń](https://github.com/m24jeczen)  
[Aleksandra Kulczycka](https://github.com/akulczycka)  
[Karolina Dunal](https://github.com/xxkaro)  
[Zofia Kamińska](https://github.com/kaminskaz)

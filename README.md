# Neural Network, ML and DL models:

## Text Preprocessing and Data Cleaning Techniques

### Libraries/Frameworks:
- **NLTK**: Used for text preprocessing tasks such as tokenization, stop words removal, and punctuation handling.
- **String operations in Python**: Employed for general text cleaning tasks like removing HTML tags and URLs.
- **Custom Text Preprocessor Class**: Implemented a custom class to encapsulate text preprocessing functions.

## Deep Learning Model Development using PyTorch

### Framework/Library:
- **PyTorch**: Utilized for developing deep learning models, defining neural network architectures, and implementing training loops.

## Utilization of Pre-trained Word Embeddings for NLP Tasks

### Library/Framework/Model:
- **Gensim**: Utilized for loading and working with pre-trained Word2Vec embeddings (word2vec-google-news-300).
- **Word2Vec Model**: Used the pre-trained Word2Vec model for converting text into word vectors.

## Efficient Data Handling, Splitting, and Evaluation using Python Libraries

### Libraries/Frameworks:
- **Pandas**: Used for data manipulation, loading CSV files, and data preprocessing tasks.
- **Scikit-Learn**: Employed for data splitting (train_test_split) and evaluation metrics (accuracy_score) during model evaluation.

## Modular and Reusable Components for Data Processing and Model Training

### Modular Components:
- **Custom Text Dataset Class**: Created a modular Dataset class (TextDataset) for handling text data and labels, ensuring reusability and scalability.
- **DataLoader Usage**: Utilized PyTorch's DataLoader for efficient batch-wise data loading, enhancing code modularity and readability.
- **Custom Transformer Model**: Developed a modular Transformer model with configurable parameters, promoting reusability across different tasks or datasets.

Informative comments are included throughout the notebooks to explain key functionalities, data transformations, and model components.
I have Organized the notebooks into logical sections such as data preprocessing, model development, training loops, and evaluation.


## Models and Hybrid Models Used

### Word2Vec Model for Word Embeddings
#### Task Performed:
- Convert text data into word vectors using pre-trained Word2Vec embeddings (word2vec-google-news-300).
- Employed the Word2Vec model to capture semantic relationships between words in the text data.

### Support Vector Machine (SVM) Classifier
#### Task Performed:
- Trained a SVM classifier on Word2Vec vectors for sentiment analysis.
- Classified text data into positive and negative sentiment categories based on the learned word representations.

### Decision Tree Classifier
#### Task Performed:
- Developed a Decision Tree classifier using Word2Vec vectors for sentiment analysis.
- Analyzed the interpretability of the Decision Tree model in understanding feature importance.

### Logistic Regression Classifier
#### Task Performed:
- Implemented a Logistic Regression classifier on Word2Vec vectors for sentiment analysis.
- Evaluated the model's performance in terms of accuracy and compared it with other classifiers.

### TF-IDF with Classifiers:

####Task Performed:
- Utilized TF-IDF vectorization for text data representation.
- Trained and evaluated multiple classifiers on TF-IDF vectors for sentiment analysis:
  - Support Vector Machine (SVM) Classifier: Classified text data into sentiment categories.
  - Decision Tree Classifier: Analyzed feature importance and model interpretability.
  - Random Forest Classifier: Assessed ensemble learning impact on predictive performance.
  - Logistic Regression Classifier: Examined model's ability for sentiment classification.

### Transformer Model for Sequence Classification
#### Task Performed:
- Developed a custom Transformer model architecture for text sequence classification tasks.
- Used multi-head self-attention mechanisms and position-wise feedforward layers for capturing contextual information.
- Trained the Transformer model on a dataset with text and label pairs for binary classification tasks.
- Evaluated the Transformer model's performance on both training and test datasets using binary cross-entropy loss and accuracy metrics.

Each model or hybrid model was tailored to address specific aspects of the data and task requirements

## Numerical Methods Labwork :
### Polynomial Interpolation and Least Squares Fitting
#### Task Performed:
- Implemented polynomial interpolation and least squares fitting using NumPy and Matplotlib.
- Constructed Vandermonde matrices and solved for coefficients to fit polynomials to data points.
- Visualized the fitted polynomials alongside original data points.

### Simple Linear Regression
#### Task Performed:
- Conducted simple linear regression using Pandas, NumPy, and Matplotlib.
- Calculated regression coefficients and plotted the regression line on the dataset.

### K-Means Clustering
#### Task Performed:
- Applied K-Means clustering using NumPy, Pandas, and Matplotlib.
- Initialized centroids, assigned data points to clusters, and updated centroids iteratively.
- Visualized the clustering results with data points and centroids.

### Gaussian Elimination
#### Task Performed:
- Implemented Gaussian elimination to solve a system of linear equations using NumPy.
- Created an augmented matrix and performed row operations to obtain reduced row-echelon form.






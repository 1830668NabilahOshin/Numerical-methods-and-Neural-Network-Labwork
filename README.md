# Numerical-methods-and-Neural-Network-Labwork

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

### Transformer Model for Sequence Classification
#### Task Performed:
- Developed a custom Transformer model architecture for text sequence classification tasks.
- Used multi-head self-attention mechanisms and position-wise feedforward layers for capturing contextual information.
- Trained the Transformer model on a dataset with text and label pairs for binary classification tasks.
- Evaluated the Transformer model's performance on both training and test datasets using binary cross-entropy loss and accuracy metrics.

### TF-IDF Vectorization with SVM Classifier
#### Task Performed:
- Utilized TF-IDF vectorization for text data representation.
- Trained a SVM classifier on TF-IDF vectors for sentiment analysis.
- Compared the performance of TF-IDF based SVM classifier with Word2Vec based classifiers.

### Evaluation Metrics and Comparisons
#### Task Performed:
- Calculated accuracy scores for each model to assess their performance on test data.
- Conducted comparative analysis between different models and vectorization techniques (Word2Vec, TF-IDF).
- Evaluated the strengths and limitations of each model in terms of accuracy, interpretability, and computational efficiency.





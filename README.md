## Introduction
Built a Website for checking the similarity between two sentences based on the **Quora Question Pairs** dataset. It works by using a **REST API** for accesing the machine learning model trained on the said dataset which outputs a **probability score** between 0 & 1

Dataset : [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)

## Working


### Layouts
![layout_1](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/misc/layout_1.jpg)

![layout_2](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/misc/layout_2.jpg)

![layout_3](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/misc/layout_3.jpg)

## Machine Learning
### Raw Data
Checking **missing** and/or **duplicate** values and removing them\
![similar_vs_unique](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/similar_vs_unique.jpg)

Historgram showing how many question were getting repeated\
![repeat_number](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/repeat_number.jpg)

### Cleaning & Pre-processing
Used **spacy** fr lemmatization and tokenization of words for a given text

Pre-prcessed the text by replacing the numbers with appropriate letters or words, replacing symbols with their suitable words and decontracted contractions as per the given link\
[contractions](https://stackoverflow.com/a/19794953)

Removed **HTML** links using **beautifulsoup**

### Feature Engineering
Here is the list of all the features I added to get more accurate results:

Normal Features
* length of string
* common words
* total common words
* total words
* words shared
* non-stopwords
* stopwords
* common non-stopwords
* common stopwords
* common tokens
* first word equal
* last word equal

Length-based Features
* absolute length difference
* average toekn length of both questions
* longest substring ratio

Fuzzy Features
* fuzz ratio
* fuzz partial ratio
* token sort ratio
* token set ratio

### Vectorization
Used Tfidf weighted Word2Vec approach for vectorization of words for a given text consisting multiple sentences\
vector_size = 100\
max_features = 300

I create word embeddings for every word using Word2Vec and feed them into the Tfidf Vectorizer object to generate a matrix of arrays

### EDA
#### Distplots
Number of characters\
![q1_len](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/q1_len.jpg)

![q2_len](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/q2_len.jpg)


Number of Words\
![q1_num_words](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/q1_num_words.jpg)

![q2_num_words](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/q2_num_words.jpg)


Common Words\
![unique_word_common](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/unique_word_common.jpg)

![similar_word_common](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/similar_word_common.jpg)


Total Words\
![unique_word_total](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/unique_word_total.jpg)

![similar_word_total](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/similar_word_total.jpg)


Total Common\
![unique_common_total](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/unique_common_total.jpg)

![similar_common_total](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/similar_common_total.jpg)


Words Shared\
![unique_word_share](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/unique_word_share.jpg)

![similar_word_share](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/similar_word_share.jpg)

#### Pairplots
![token_min](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/token_min.jpg)

![token_max](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/token_max.jpg)

![word_order](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/word_order.jpg)

![word_stat](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/word_stat.jpg)

![fuzz](https://github.com/subhashishansda4/Sentence-Similarity/blob/main/plots/fuzz.jpg)

### t-SNE
Dimensionality reduction for 15 features to 3 features\
`sample_size = 2000`

https://github.com/subhashishansda4/Sentence-Similarity/assets/84849423/77872622-b17d-41d0-a22e-ccb16109fa67

### Model Evaluation
Used **KFold** for model evaluation from 6 different classification models\
Used **LogLoss**, **Accuracy** and **Confusion Matrix** as scoring parameters\

Selected **Random Forest Classifier** as the most suitable model and pickled it for use in the API

## Predictions
Did all of the above and made functions and classes for them\
Returned output as an array of word embeddings if given a text

## API
Used the pickle file for Random Forest model and made a function as a **POST** request to the **web server** to return 'similar' or 'unique' based on a given probability value using the machine learning model if given `sentence1` and `sentence2`

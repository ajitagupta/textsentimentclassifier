# Text Sentiment Classifier

A simple text sentiment classifier that uses logistic regression on movie reviews from the [IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/) to determine whether a given review is positive or negative.

## Features

- Preprocessing of raw text by converting to lowercase, removing punctuation, and splitting into tokens.
- Extraction of text features using TF-IDF.
- Training a logistic regression model to classify sentiment.
- Evaluation on a test set, reporting accuracy.

## Getting Started

1. Download and extract the [IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/).
2. Update the paths in the script to point to the dataset’s directories.
3. Run the Python script to train the model and view the evaluation results.

## Dependencies

- Python 3.7+
- [scikit-learn](https://scikit-learn.org/)
- [NumPy](https://numpy.org/)
- [re](https://docs.python.org/3/library/re.html)

## License

This project is licensed under the MIT License.

import pandas as pd
import numpy as np
import json
import torch
from numpyencoder import NumpyEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import Dataset, DatasetDict

class Dataloader:
    def __init__(self, file_path, file_type=None, label_column='label', test_size=0.2, val_size=0.25, random_state=25, **kwargs):
        """
        Initialize the Dataloader object for training CustomBertForSequenceClassification
        Args:
            file_path (str): Path to the parquet file.
            label_column (str): The column name representing labels in the dataset.
            test_size (float): Proportion of the dataset to use as the test set.
            val_size (float): Proportion of the train/validation split to use as the validation set.
            random_state (int): Seed for reproducibility.
        """
        self.file_path = file_path
        self.file_type = file_type or file_path.split('.')[-1]
        self.label_column = label_column
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.kwargs = kwargs
        self.df = None
        self.dataset = None
        self.num_labels = None
        self.class_weights = None
        self.encoding, self.reverse_encoding = None, None

    def load_data(self):
        """Loads the file and prepares the dataset."""
        loaders = {
            'csv': pd.read_csv,
            'gzip': pd.read_parquet,
            'excel': pd.read_excel,
            'json': pd.read_json,
            'feather': pd.read_feather,
        }

        if self.file_type not in loaders:
            raise ValueError(f"Unsupported file type: {self.file_type}")
        self.df = loaders[self.file_type](self.file_path, **self.kwargs).reset_index()

        # Labels should start from 0; they will be mapped back
        # when saving the predicted results
        if self.df[self.label_column].min() == 1:
            self.df[self.label_column] -= 1

        # Map labels to dense for CrossEntropyL (the italian BERT doesn't like sparse arrays)
        unique_labels, label_counts = np.unique(self.df[self.label_column], return_counts=True)
        self.num_labels = len(unique_labels)
        self.encoding = {label:idx for idx,label in enumerate(unique_labels)}
        self.reverse_encoding = {idx:label for idx,label in enumerate(unique_labels)}
        self.df[self.label_column] = self.df[self.label_column].map(self.encoding)
        # saving the reverse indexing
        with open("reverse_encoding.json", "w") as f:
            json.dump(self.reverse_encoding, f,
                      indent=4, sort_keys=True,
                      separators=(', ', ': '), ensure_ascii=False,
                      cls=NumpyEncoder)

        # Class weights
        inverse = 1 / label_counts
        normalized_weights = inverse / inverse.sum()
        self.class_weights = torch.FloatTensor(normalized_weights).to('cuda')

    def stratified_split(self):
        """
        Performs stratified train/validation/test split.
        Given that the data has many labels and they are often unequally distributed,
        many classes can be not represented at all in the training session.
        This method forces similar class distributions.
        If some classes are in less than 3 observations, it will raise an error.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # First split: train+val/test
        split = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        train_val_idx, test_idx = next(split.split(self.df, self.df[self.label_column]))
        train_val_data = self.df.iloc[train_val_idx].reset_index(drop=True)
        test_set = self.df.iloc[test_idx].reset_index(drop=True)

        # Second split: train/validation
        split = StratifiedShuffleSplit(n_splits=1, test_size=self.val_size, random_state=self.random_state)
        train_idx, val_idx = next(split.split(train_val_data, train_val_data[self.label_column]))
        train_set = train_val_data.iloc[train_idx].reset_index(drop=True)
        val_set = train_val_data.iloc[val_idx].reset_index(drop=True)

        # Convert to Hugging Face DatasetDict
        self.dataset = DatasetDict({
            'train': Dataset.from_pandas(train_set),
            'validation': Dataset.from_pandas(val_set),
            'test': Dataset.from_pandas(test_set)
        })

    def get_dataset(self):
        """
        Retrieves the dataset dictionary containing train, validation, and test sets.

        Returns:
            DatasetDict: A dictionary containing the splits as Hugging Face datasets.
        """
        if self.dataset is None:
            raise ValueError("Dataset not created. Call stratified_split() first.")
        return self.dataset

    def get_class_weights(self):
        """
        Retrieves the class weights.

        Returns:
            torch.FloatTensor: Class weights for handling imbalanced classes.
        """
        if self.class_weights is None:
            raise ValueError("Class weights not computed. Call load_data() first.")
        return self.class_weights

    def get_num_labels(self):
        """
        Retrieves the number of unique labels.

        Returns:
            int: Number of unique labels in the dataset.
        """
        if self.num_labels is None:
            raise ValueError("Number of labels not available. Call load_data() first.")
        return self.num_labels

    def get_encoding(self):
        """
        Retrieves the mapping of labels to categories.

        Returns:
            dict: Mapping of label IDs to their respective categories.
        """
        if self.encoding is None:
            raise ValueError("Label mapping nonexistent. Call load_data() first.")
        return self.encoding

    def get_r_encoding(self):
        """
        Retrieves the mapping of labels to categories.

        Returns:
            dict: Mapping of label IDs to their respective categories.
        """
        if self.reverse_encoding is None:
            raise ValueError("Label mapping nonexistent. Call load_data() first.")
        return self.reverse_encoding
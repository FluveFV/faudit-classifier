from transformers import AutoTokenizer

class TokenizerFunction:
    def __init__(self, model_name, max_length=512, use_fast=True, use_cache=False):
        """
        Initialize the tokenizer function wrapper 🌯
        Args:
            model_name (str): Name of the pre-trained model.
            max_length (int): Maximum sequence length for tokenization.
            use_fast (bool): Whether to use the fast tokenizer implementation.
            use_cache (bool): Whether to cache the tokenizer results.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                        use_fast=use_fast,
                                                        use_cache=use_cache,
                                                       truncation=True)
        self.max_length = max_length

    def __call__(self, data):
        """
        Tokenizes the input data.

        Args:
            data (dict): A dictionary containing the 'text' field to tokenize.

        Returns:
            dict: Tokenized data including input IDs, attention masks, etc.
        """
        return self.tokenizer(data['text'],
                              padding='max_length',
                              truncation=True,
                              max_length=self.max_length,
                              return_tensors="pt")

    def get_tokenizer(self):
        return self.tokenizer

# namegen
Single-file implementation of a character-level language model using a recurrent neural network (RNN) with the PyTorch library. The code trains the RNN on a dataset of names to generate new sequences of characters that resemble names. The model "learns" sequences based on four characters and as a result, learns to generate pretty good names.

1. Loads a dataset of names from a file named 'names.txt' and processes the data.
2. Creates a character-level RNN model using PyTorch.
3. Defines functions for training the model, generating new sequences of characters, and sampling from the model.
4. Trains the RNN model on the dataset of names.
5. Generates new sequences of characters (names) using the trained model.

In this model, the weights are the frequencies of occurrence of sequences of four symbols, which are stored in the fourgrams tensor. This tensor is a four-dimensional array, where each axis corresponds to a character index in the itos map.

The weights of the model, that is, the frequency of occurrence of four grams, are stored in the cells of this tensor. For example, the value fourgrams[0][1][2][3] would contain the frequency of occurrence of the character sequence 'abcd'.

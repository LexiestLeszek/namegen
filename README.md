# namegen
Single-file implementation of a character-level language model using a recurrent neural network (RNN) with the PyTorch library. The code trains the RNN on a dataset of names to generate new sequences of characters that resemble names. The model "learns" sequences based on four characters and as a result, learns to generate pretty good names.

Much simpler than Karpathy's makemore and minGPT, the idea is to showcase how generative AI can be done on a very small scale - generating names. The task of generating a coherent and normal sounding name is the minimal version of the task of generating a paragraph of text, the difference (although of course not the only one) is that we use combinations of letters, not combinations of words and the model architecture is much simpler.

The idea of the project is to showcase the most minimal implementation of generative language model for education purposes.

# Two main methods:
**train("dataset_name.txt"):**
Loads a dataset of names from a file named 'names.txt', processes the data, tokenizes the data by splitting it using each new line as separator. Creates and trains character-level RNN model using PyTorch, storing it in the self.fourgrams variable.

**generate_names(num_names=1):**
Takes number of names as argument and and generates a number new sequences of characters (names) using the trained model.

# How to run
Open VScode and type in terminal:
1. `git clone https://github.com/LexiestLeszek/namegen.git`
2. `pip install torch`
3. `python namegen.py`

# Weights, Tensors, nGrams
In this model, the fourgrams are the frequencies of occurrence of sequences of four symbols, which are stored in the fourgrams tensor called self.fourgrams. Basically, these are the weights of the model. This tensor is a four-dimensional array, where each axis corresponds to a character index in the itos map. The frequency of occurrence of four grams, are stored in the cells of this tensor. For example, the value self.fourgrams[1][2][3][4] would contain the frequency of occurrence of the character sequence 'abcd', and the dot has the index of [0], because we need some kind of a token to understand where is the beginning and end of the name, in case of this project we use the theree dots in a row as a unique sequence that signals that we reached the end of word. The tensor can be stored separately if you add something like torch.save('tensor.pt') at the end of train() method.

# Usage
Feel free to fork it and use it however you want.

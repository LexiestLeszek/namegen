# namegen

Goal of this educational repository is to provide a self-contained, minimalistic implementation of language model using Pytorch.

Many implementations of diffusion models can be a bit overwhelming. Here, namegen: under 100 lines of code, fully self contained implementation single-file implementation of a character-level language model. The code trains the model based on a Markov Chain on a dataset of names to generate new sequences of characters that resemble names. The model "learns" sequences based on four characters and as a result, learns to generate pretty good names. 

Much simpler than Karpathy's makemore and minGPT, the idea is to showcase how generative AI can be done on a very small scale - generating names. The task of generating a coherent and normal sounding name is the minimal version of the task of generating a paragraph of text, the difference (although of course not the only one) is that we use combinations of letters, not combinations of words and the model architecture is much simpler and it is, technically, not even a neural network. The core though is similar to modern LLMs: give a likely next token given the last n tokens.

Treat this project as "hello world" in the world of language models. The project is made to showcase the most minimal implementation of generative language model for education purposes, to show that LLMs are basically statistics and randomization. It doesn't have any neural networks in its architecture and still Markov Chain can result in a pretty good sounding names. You can also experiment with this project and try to make an Markov Chain n-gram that would generate words, old school chatbots were actually doing exactly that and with big enough dataset they were pretty good for their time.

# Two main methods:
**train("dataset_name.txt"):**
Loads a dataset of names from a file named 'names.txt', processes the data, tokenizes the data by splitting it using each new line as separator. Creates and trains character-level RNN model using PyTorch, storing it in the self.fourgrams variable. During training, we use char_to_ind to convert characters to indeces and save them into self.fourgrams.

**generate_names(num_names=1):**
Takes number of names as argument and generates a number of new sequences of characters (names) using the trained model. During generation, we use ind_to_char to convert indeces back to the characters and generate letters for the name.

# How to run
Open the terminal an type:
1. `git clone https://github.com/LexiestLeszek/namegen.git`
2. `pip install torch`
3. `python namegen.py`
Train the model (takes around 10-20 sec) and generate 10 names.

# Inference, Weights, Tensors, nGrams
In modern LLMs (Mistral, Llama2, Qwen, etc), there are usually two files - inference file (python code that actually runs the model) and the model weights, or, basically, the model itself - usually in the form of *.bin or *.gguf format, that is the file that actually stores model's learned parameters and it is being stored in different format for the sake of optimizing disk space. In our case, the inference is the generate_names() method (the method that allows us to run the model) and the "weights" are stored in the self.fourgrams variable (for our particular model it is not entirely correct to call them weights, but for the learning purposes we will do it nevertheless). You can train the model and add torch.save(self.fourgrams, "tensor.pt") at the end of train() method and that way you will have a "weights" file just like you have it with advanced LLMs.

The fourgrams are the frequencies of occurrence of sequences of four symbols, which are stored in the fourgrams tensor called self.fourgrams. Basically, these are the weights of the model. This tensor is a four-dimensional array, where each axis corresponds to a character index in the itos map. The frequency of occurrence of four grams, are stored in the cells of this tensor. For example, the value self.fourgrams[1][2][3][4] would contain the frequency of occurrence of the character sequence 'abcd', and the dot has the index of [0], because we need some kind of a token to understand where is the beginning and end of the name, in case of this project we use the theree dots in a row as a unique sequence that signals that we reached the end of word. The tensor can be stored separately if you add something like torch.save('tensor.pt') at the end of train() method.

Modern LLMs store parameters in the form of neural network (connections of multiple parameters to each other), but in our case we just use tensor for the sake of simplicity and learning. 

# Usage
Feel free to fork it and use it however you want.

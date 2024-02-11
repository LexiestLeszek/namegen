import torch

class NameGen:
    def __init__(self):
        self.characters = None
        self.char_to_ind = None
        self.ind_to_char = None
        self.fourgrams = None
        # Defining the NameGen class with initialization of attributes

    def train(self, filename):
        # Load dataset and train the model
        
        with open(filename, 'r') as f:
            words = f.read().splitlines()
        # Open and read the file to get a list of words, each word starts from a new line, that's whe we need splitlines()

        self.characters = sorted(list(set(''.join(words))))
        print(f"Vocab Tokens:\n {self.characters}")
        # Create vocabulary - get and sort unique characters in the dataset (basically alphabet)
        
        vocab_len = len(self.characters) + 1
        print(f"Vocab Length: {vocab_len}")
        # Calculate the vocabulary length (characters) + 1 (for the dot) and print it
        
        self.char_to_ind = {}
        for i, s in enumerate(self.characters):
            self.char_to_ind[s] = i +  1
        # Create character-to-index mapping (key-value pairs)
        
        self.char_to_ind['.'] = 0
        # The dot represents marker for the start and end of a name
        
        print(f"Character-to-Index mapping:\n {self.char_to_ind}")
        # Print the mapping to showcase how it works
        
        self.ind_to_char = {}
        for s, i in self.char_to_ind.items():
            self.ind_to_char[i] = s
        # Create index-to-character mapping (key-value pairs), basically the reverse of char_to_ind

        self.fourgrams = torch.zeros((vocab_len, vocab_len, vocab_len, vocab_len), dtype=torch.int32)
        # Initialize a tensor to store frequency of four characters occurring together
        # Size of each axis is dynamic and based on the vocab length
        # dtype=torch.int32 means that the tensor will store integers using 32 bits per element
        
        # Implementation of torch.zeros without pyTorch:
        # self.fourgrams = [[[[0 for _ in range(vocab_len)] for _ in range(vocab_len)] for _ in range(vocab_len)] for _ in range(vocab_len)]
        
        print("Trarining starts ...")
        for word in words:
            chs = ['.', '.', '.'] + list(word) + ['.', '.', '.']
            # Add padding dots to the word
            # Three dots (sentinel characters) act as markers indicating the start and end of a name
            
            for ch1, ch2, ch3, ch4 in zip(chs, chs[1:], chs[2:], chs[3:]):
                ix1 = self.char_to_ind[ch1]
                ix2 = self.char_to_ind[ch2]
                ix3 = self.char_to_ind[ch3]
                ix4 = self.char_to_ind[ch4]
                self.fourgrams[ix1, ix2, ix3, ix4] += 1
            # Populate the fourgrams tensor with frequencies
            # self.fourgrams[1][2][3][4] would contain the frequency of occurrence of the 'abcd' sequence
        
        print("Training finished!")

        #torch.save(self.fourgrams, "namegen_weights.pt")
        # Save the trained model weights

    def generate_names(self, num_names=1):
        # Passing the number of words to generate
        
        for _ in range(num_names):
            name = []
            # Empty list for a name, since every element would be generated
            ix1 = ix2 = ix3 = 0
            # Initialize indices for generating names
            
            while True:
                p = self.fourgrams[ix1, ix2, ix3].float()
                p = p / p.sum()
                # Calculate probabilities of the next character and normalize them

                probs_flat = p.view(-1)
                # Collapse all dimensions into a single dimension
                
                adjusted_ix = torch.multinomial(probs_flat, num_samples=1)
                # Randomly sample the next character index based on the input probabilities provided

                out = self.ind_to_char[adjusted_ix.item()]
                # Use ind_to_char to take sample a character (letter) by its index
                
                name.append(out)
                # Append the sampled character to the name

                ix1 = ix2
                ix2 = ix3
                ix3 = adjusted_ix
                # Update indices for next iteration
                
                if adjusted_ix == 0:
                    break
                # If the sampled character is a dot - stop

            name = ''.join(name[:-1])
            # Remove the last character from the name list (the dot) and concatenate the remaining characters into string
            
            name_capitalized = name[0].upper() + name[1:]
            print(name_capitalized)
            # Capitalize the first letter of the generated name and print it

# Usage
model = NameGen()
model.train('names.txt')
model.generate_names(num_names=10)

import torch
# Import the PyTorch library

class NameGen:
    def __init__(self):
        self.characters = None
        self.stoi = None
        self.itos = None
        self.fourgrams = None
    # Define the NameGen class with initialization of attributes

    def load_and_train(self, filename):
        # Load dataset and prepare character mappings
        with open(filename, 'r') as f:
            words = f.read().splitlines()
        # Open and read the file to get a list of words

        self.characters = sorted(list(set(''.join(words))))
        # Get and sort unique characters in the dataset (basically alphabet)
        vocab_len = len(self.characters) + 1
        # Calculate the vocabulary length (characters + dot)
        self.stoi = {s: i + 1 for i, s in enumerate(self.characters)}
        self.stoi['.'] = 0
        self.itos = {i: s for s, i in self.stoi.items()}
        # Create character-to-index and index-to-character mappings

        self.fourgrams = torch.zeros((vocab_len, vocab_len, vocab_len, vocab_len), dtype=torch.int32)
        # Initialize a tensor to store frequency of four characters occurring together
        for word in words:
            chs = ['.', '.', '.'] + list(word) + ['.', '.', '.']
            # Add padding dots to the word
            # dots act as markers indicating the start and end of words
            for ch1, ch2, ch3, ch4 in zip(chs, chs[1:], chs[2:], chs[3:]):
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]
                ix3 = self.stoi[ch3]
                ix4 = self.stoi[ch4]
                self.fourgrams[ix1, ix2, ix3, ix4] += 1
        # Populate the fourgrams tensor with frequencies
        # self.fourgrams[1][2][3][4] would probably contain the frequency of occurrence of the 'abcd' sequence 

        #torch.save(self.fourgrams, "namegen_weights.pt")
        # Save the trained model weights

    def generate_names(self, num_words=1):
        for _ in range(num_words):
            name = []
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

                out = self.itos[adjusted_ix.item()]
                name.append(out)
                # Append the sampled character to the name

                ix1 = ix2
                ix2 = ix3
                ix3 = adjusted_ix
                # Update indices for next iteration
                
                if adjusted_ix == 0:
                    break
                # Stop if the sampled character is a dot

            name = ''.join(name[:-1])
            name_capitalized = name[0].upper() + name[1:]
            print(name_capitalized)
            # Capitalize the first letter of the generated name and print it

# Usage
model = NameGen()
model.load_and_train('female_names_rus.txt')
model.generate_names(num_words=10)
# Create an instance of NameGen, load data and train on a dataset, then generate names

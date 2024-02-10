import torch

class NameGen:
    def __init__(self):
        self.characters = None
        self.stoi = None
        self.itos = None
        self.fourgrams = None

    def load_and_train(self, filename):
        # Load dataset and prepare character mappings
        with open(filename, 'r') as f:
            words = f.read().splitlines()

        self.characters = sorted(list(set(''.join(words))))
        self.stoi = {s: i +  1 for i, s in enumerate(self.characters)}
        self.stoi['.'] =  0
        self.itos = {i: s for s, i in self.stoi.items()}

        # Calculate frequency of four characters occurring together
        self.fourgrams = torch.zeros((27,  27,  27,  27), dtype=torch.int32)
        for word in words:
            chs = ['.', '.', '.'] + list(word) + ['.', '.', '.']
            for ch1, ch2, ch3, ch4 in zip(chs, chs[1:], chs[2:], chs[3:]):
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]
                ix3 = self.stoi[ch3]
                ix4 = self.stoi[ch4]
                self.fourgrams[ix1, ix2, ix3, ix4] +=  1

    def generate_names(self, num_words=1):
        for _ in range(num_words):
            out = []
            ix1 = ix2 = ix3 =  0
            while True:
                p = self.fourgrams[ix1, ix2, ix3].float()
                p = p / p.sum()

                probs_flat = p.view(-1)
                adjusted_ix = torch.multinomial(probs_flat, num_samples=1)
                
                # Capitalize the first character of the name
                name = self.itos[adjusted_ix.item()]

                out.append(name)
                
                ix1 = ix2
                ix2 = ix3
                ix3 = adjusted_ix
                
                if adjusted_ix ==  0:
                    break
            out = ''.join(out[:-1])
            capitalized_out = out[0].upper() + out[1:]
            print(capitalized_out)

# Example usage:
model = NameGen()
model.load_and_train('names.txt')
model.generate_names(num_words=10)

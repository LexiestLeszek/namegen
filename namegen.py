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
        # 27 is the number of letters in alphabet + dot, you can write any number you want depending on your CPU
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
            name = []
            ix1 = ix2 = ix3 =  0
            while True:
                p = self.fourgrams[ix1, ix2, ix3].float()
                p = p / p.sum()

                probs_flat = p.view(-1)
                adjusted_ix = torch.multinomial(probs_flat, num_samples=1)
                
                out = self.itos[adjusted_ix.item()]

                name.append(out)
                
                ix1 = ix2
                ix2 = ix3
                ix3 = adjusted_ix
                
                if adjusted_ix ==  0:
                    break
            
            name = ''.join(name[:-1])
            # adding capitalization tp the first letter of each name
            name_capitalized = name[0].upper() + name[1:]
            print(name_capitalized)

model = NameGen()
model.load_and_train('names.txt')
model.generate_names(num_words=10)

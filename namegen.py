import torch

# load dataset (32k names) and sort unique chars (meta tokenizer)
words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))

st = {s: i + 1 for i, s in enumerate(chars)}
st['.'] = 0

it = {i: s for s, i in st.items()}

# calculate frequency of four chars occurring together
N = torch.zeros((27, 27, 27, 27), dtype=torch.int32)

for word in words:
    chs = ['.', '.', '.'] + list(word) + ['.', '.', '.']
    
    for ch1, ch2, ch3, ch4 in zip(chs, chs[1:], chs[2:], chs[3:]):
        ix1 = st[ch1]
        ix2 = st[ch2]
        ix3 = st[ch3]
        ix4 = st[ch4]
        
        N[ix1, ix2, ix3, ix4] += 1

# Function to adjust probabilities based on temperature
def adjust_probs(probs, temperature=1.0):
    if temperature == 0.0:
        return torch.argmax(probs).unsqueeze(0)  # Greedy selection at temperature 0
    else:
        probs_flat = probs.view(-1)
        adjusted_ix = torch.multinomial(probs_flat, num_samples=1)
        return adjusted_ix

G = torch.Generator().manual_seed(3822483571)
temperature = 0.7  # Set the temperature value (adjust as needed)

for i in range(9):
    out = []
    ix1 = ix2 = ix3 = 0
    while True:
        p = N[ix1, ix2, ix3].float()
        p = p / p.sum()
        
        adjusted_ix = adjust_probs(p, temperature)
        out.append(it[adjusted_ix.item()])
        ix1 = ix2
        ix2 = ix3
        ix3 = adjusted_ix
        
        if adjusted_ix == 0:
            break
    print(''.join(out[:-1]))
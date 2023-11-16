import torch

# load dataset (32k names) and sort unique chars (meta tokenizer)
words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))

st = {s: i + 1 for i, s in enumerate(chars)}
st['.'] = 0

it = {i: s for s, i in st.items()}

# calculate frequency of three chars occurring together
N = torch.zeros((27, 27, 27), dtype=torch.int32)

for word in words:
    chs = ['.', '.'] + list(word) + ['.', '.']
    
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = st[ch1]
        ix2 = st[ch2]
        ix3 = st[ch3]
        
        N[ix1, ix2, ix3] += 1

# calculate probability distribution, generate names

G = torch.Generator().manual_seed(2147483648)

for i in range(9):
    out = []
    ix1 = ix2 = 0
    while True:
        p = N[ix1, ix2].float()
        
        p = p / p.sum()
        
        ix3 = torch.multinomial(p, num_samples=1,
                                replacement=True,
                                generator=G).item()
        out.append(it[ix3])
        ix1 = ix2
        ix2 = ix3
        
        if ix3 == 0:
            break
    print(''.join(out[:-1]))

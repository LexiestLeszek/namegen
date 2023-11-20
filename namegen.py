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

# calculate probability distribution, generate names

G = torch.Generator().manual_seed(3822483571)

for i in range(9):
    out = []
    ix1 = ix2 = ix3 = 0
    while True:
        p = N[ix1, ix2, ix3].float()
        
        p = p / p.sum()
        
        ix4 = torch.multinomial(p, num_samples=1,
                                replacement=True,
                                generator=G).item()
        out.append(it[ix4])
        ix1 = ix2
        ix2 = ix3
        ix3 = ix4
        
        if ix4 == 0:
            break
    print(''.join(out[:-1]))

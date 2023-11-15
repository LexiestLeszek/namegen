# load dataset (32k names) and sort unique chars (meta tokenizer)

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))

st = {s:i+1 for i,s in enumerate(chars)}
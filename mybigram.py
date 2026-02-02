import torch

#read in the dataset
words = open('indian_names.txt', 'r').read().splitlines()
words = [w.lower() for w in words]

#create a dictionary of the two-character sequences and their frequencies
b = {}
for w in words:
  chs = ['<S>'] + list(w) + ['<E>']
  for ch1, ch2 in zip(chs, chs[1:]):
    bigram = (ch1, ch2)
    b[bigram] = b.get(bigram, 0) + 1
#print(b)


#create empty tensor
N = torch.zeros((27, 27), dtype=torch.int32)

#maps characters to index, adds 0 as "start"
#accounts for edge cases that dont have all 26 characters
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

#counts bigrams and updates the tensor
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] += 1



#Find the percent likelihood of each bigram
P = (N+1).float()
P /= P.sum(1, keepdims=True)


g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  
  out = []
  ix = 0
  while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))

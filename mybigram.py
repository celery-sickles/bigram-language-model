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



#find the percent likelihood of each bigram
P = (N+1).float()
P /= P.sum(1, keepdims=True)


g = torch.Generator().manual_seed(2147483647)

#generate names
for i in range(10):
  
  out = []
  ix = 0
  while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))


#calculate cross entropy loss
x_list, y_list = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        x_list.append(stoi[ch1])
        y_list.append(stoi[ch2])

x_tensor = torch.tensor(x_list, dtype=torch.long)
y_tensor = torch.tensor(y_list, dtype=torch.long)

log_P = torch.log(P + 1e-8)

selected_log_probs = log_P[x_tensor, y_tensor]

#print
cross_entropy = -selected_log_probs.mean().item()
print(f"Cross-entropy loss: {cross_entropy:.4f} nats per character")
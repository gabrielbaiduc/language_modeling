# ngram_names

This project explores n-gram language modeling for generating names, inspired by and extending Andrej Karpathy's makemore series. It includes code and data for training and experimenting with n-gram models on name datasets.

## Findings

Using a 6-gram model on either dataset produced the best results. However, since the memory requirements for n-grams grow exponentially with n, 6 was the practical limit on my machine—training a 7-gram model would have required about 40GB of memory.

**Examples of names generated with the 6-gram model:**

- breton
- bowen
- minh-phuc
- rand
- surendra
- xuong
- vino
- inigo
- pierett
- geriann
- abriela
- doretta
- nanni
- isha
- daire
- chrystal
- izaskum
- rama
- nessie
- gracie
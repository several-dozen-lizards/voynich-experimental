"""Three frozen tokenizations; imported as src.tokenize.

Run pipeline modules from the project root with -m to avoid shadowing Python's
standard-library tokenize module with this specification-requested filename.
"""
from collections import Counter

def train_bpe(words,k):
    pieces=Counter(tuple(w) for w in words)
    merges=[]
    for _ in range(k):
        pairs=Counter()
        for symbols,count in pieces.items():
            for pair in zip(symbols,symbols[1:]):
                pairs[pair]+=count
        if not pairs:
            break
        pair=min(pairs,key=lambda p:(-pairs[p],p))
        merges.append(pair)
        nxt=Counter()
        for symbols,count in pieces.items():
            nxt[tuple(merge_pair(list(symbols),pair))]+=count
        pieces=nxt
    return merges

def merge_pair(pieces,pair):
    out=[];i=0
    while i<len(pieces):
        if i+1<len(pieces) and (pieces[i],pieces[i+1])==tuple(pair):
            out.append(pieces[i]+pieces[i+1]);i+=2
        else:
            out.append(pieces[i]);i+=1
    return out

def tokenize_line(text,scheme,clusters=None,merges=None):
    if scheme=='T1':
        return list(text)
    if scheme=='T3':
        return text.split()
    words=[]
    for word in text.split():
        if merges is not None:
            pieces=list(word)
            for pair in merges:
                pieces=merge_pair(pieces,pair)
        else:
            pieces=[];i=0
            ordered=sorted(clusters,key=lambda x:(-len(x),x))
            while i<len(word):
                chosen=next((c for c in ordered if word.startswith(c,i)),word[i])
                pieces.append(chosen);i+=len(chosen)
        if words:
            words.append(' ')
        words.extend(pieces)
    return words

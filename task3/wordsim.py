from nltk.corpus import wordnet as wn
import csv
import numpy as np
import pandas as pd

f = open('wordsim_similarity.txt', 'r')
lines = f.readlines()


# функция, возвращающая набор различных синонимов для входного слова
def get_synonyms(word):
    synonyms = []
    synsets = wn.synsets(word)
    if len(synsets) == 0:
        return []
    for syn in synsets:
        for l in syn.lemmas():
            if l.name() != word and l.name() not in synonyms:
                synonyms.append(l.name().replace('_', ' '))
    # print(synonyms)
    return synonyms


summa = 0
syn_cnt = 0
print('---Synonyms---')
for i in range(len(lines)):
    words = lines[i].split()
    syn = get_synonyms(words[0])
    cnt = 0
    if words[1] in syn:
        print(words[0], words[1], words[2])
        summa += float(words[2])
        cnt += 1
        syn_cnt += 1
        
if syn_cnt != 0:
    print()
    print('Count:', syn_cnt)
    print('Average:', summa /syn_cnt)
else:
    print('No pairs')
print()


# функция, возвращающая набор различных гипонимов для входного слова
def get_hyponyms(word):
    hypon = []
    synsets = wn.synsets(word)

    for syn in synsets:
        b = wn.synset(syn.name())
        w = b.hyponyms()

        for hyp in w:
            for l in hyp.lemmas():
                if l.name() not in hypon:
                    hypon.append(l.name().replace('_', ' '))
    # print(hypon)
    return hypon


summa = 0
hyp_cnt = 0
print('---Hyponyms---')
for i in range(len(lines)):
    words = lines[i].split()
    hyp = get_hyponyms(words[0])
    cnt = 0
    if words[1] in hyp:
        print(words[0], words[1], words[2])
        summa += float(words[2])
        cnt += 1
        hyp_cnt += 1
s = summa
if hyp_cnt != 0:
    print()
    print('Count:', hyp_cnt)
    print('Average:', summa / hyp_cnt)
else:
    print('No pairs')
print()


# функция, возвращающая набор различных гиперонимов для входного слова
def get_hypernyms(word):
    hyper = []
    synsets = wn.synsets(word)

    for syn in synsets:
        b = wn.synset(syn.name())
        w = b.hypernyms()
        
        for hyp in w:
            for l in hyp.lemmas():
                if l.name() not in hyper:
                    hyper.append(l.name().replace('_', ' '))
    # print(hyper)
    return hyper


summa = 0
hyper_cnt = 0
print('---Hyperonyms---')
for i in range(len(lines)):
    words = lines[i].split()
    hyp = get_hypernyms(words[0])
    cnt = 0
    for hypernym in hyp:
        if words[1] == hypernym:
            print(words[0], words[1], words[2])
            summa += float(words[2])
            cnt += 1
            hyper_cnt += 1

if hyper_cnt != 0:
    print()
    print('Count:', hyper_cnt)
    print('Average:', summa / hyper_cnt)
else:
    print('No pairs')
print()

s += summa
h = hyper_cnt + hyp_cnt
print('Total count of pairs \"Hyponym — Hyperonym\" ignoring words\' order:', h)
print('Total average:', s/h)
print()


def get_meronyms(word):
    meron = []
    synsets = wn.synsets(word)
    
    for syn in synsets:
        b = wn.synset(syn.name())
        p = b.part_meronyms()
        s = b.substance_meronyms()
        m = b.member_meronyms()
        
        for mer in p:
            for l in mer.lemmas():
                if l.name() != word and l.name() not in meron:
                    meron.append(l.name().replace('_', ' '))
        for mer in s:
            for l in mer.lemmas():
                if l.name() != word and l.name() not in meron:
                    meron.append(l.name().replace('_', ' '))
        for mer in m:
            for l in mer.lemmas():
                if l.name() != word and l.name() not in meron:
                    meron.append(l.name().replace('_', ' '))
    # print(meron)
    return meron


summa = 0
mer_cnt = 0
print('---Meronyms---')
for i in range(len(lines)):
    words = lines[i].split()
    mer = get_meronyms(words[0])
    cnt = 0
    if words[1] in mer:
        print(words[0], words[1], words[2])
        summa += float(words[2])
        cnt += 1
        mer_cnt += 1
        
if mer_cnt != 0:
    print()
    print('Count:', mer_cnt)
    print('Average:', summa / mer_cnt)
else:
    print('No pairs')
print()


def get_holonyms(word):
    holon = []
    synsets = wn.synsets(word)
    
    for syn in synsets:
        b = wn.synset(syn.name())
        p = b.part_holonyms()
        s = b.substance_holonyms()
        m = b.member_holonyms()
 
        for hol in p:
            for l in hol.lemmas():
                if l.name() != word and l.name() not in holon:
                    holon.append(l.name().replace('_', ' '))
        for hol in s:
            for l in hol.lemmas():
                if l.name() != word and l.name() not in holon:
                    holon.append(l.name().replace('_', ' '))
        for hol in m:
            for l in hol.lemmas():
                if l.name() != word and l.name() not in holon:
                    holon.append(l.name().replace('_', ' '))
    # print(holon)
    return holon


summa = 0
hol_cnt = 0
print('---Holonyms---')
for i in range(len(lines)):
    words = lines[i].split()
    hol = get_holonyms(words[0])
    cnt = 0
    if words[1] in hol:
        print(words[0], words[1], words[2])
        summa += float(words[2])
        cnt += 1
        hol_cnt += 1

if hol_cnt != 0:
    print()
    print('Count:', hol_cnt)
    print('Average:', summa / hol_cnt)
else:
    print('No pairs')
print()

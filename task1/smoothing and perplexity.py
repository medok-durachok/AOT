#!/usr/bin/env python
# coding: utf-8

import nltk
import collections
import string
import csv
import pandas as pd
import numpy as np
from collections import defaultdict
from collections import Counter

corpus = open('text.txt', encoding="utf-8-sig").read()
test = open('test.txt', encoding='utf-8-sig').read()


# функция очистки текста
def remove_pun(text):
    for pun in string.punctuation:
        if pun == '-':
            text = text.replace(pun, " ")
        else: 
            text = text.replace(pun, "")
    text = text.lower()
    return text


# чистим текст, обе части
test = remove_pun(test)
corpus = remove_pun(corpus)


class Unigram(object):    # модель униграмм
    
    def __init__(self,  data, testing):
        self.text = data
        self.unigrams = []
        self.test = testing.split()
        self.lmbd = 0.25
        c = self.text.split()
        for i in range(len(c)):
            self.unigrams.append(c[i])  
            
    # сглаживание Лапласа
    def uni_lpls(self):
        N = len(self.unigrams)
        V = len(list(set(self.unigrams)))
        C = Counter(self.test)
        
        lpls = []

        for word in C.keys():
            if word in self.unigrams:
                lpls.append((self.unigrams.count(word) + 1) / (N + V))             
            else:
                lpls.append(1 / (N + V))
        return lpls, C.keys()
    
    # сглаживание с лямбдой
    def uni_lind(self):
        N = len(self.unigrams)
        C = Counter(self.test)
        B = len(list(set(self.unigrams)))
        
        lind = []
        for word in C.keys():
            if word in self.unigrams:
                lind.append((self.unigrams.count(word) + self.lmbd) / (N + B * self.lmbd))
            else:
                lind.append(self.lmbd / (N + B * self.lmbd))
        return lind, C.keys()
     
    # исходные вероятности униграмм
    def uni_prob(self):
        lst = []
        d = [] 

        for i in range(len(self.unigrams)):
            if d.count(self.unigrams[i]) < 1:
                d.append(self.unigrams[i])
                lst.append(self.unigrams.count(self.unigrams[i]))  

        for i in range(len(lst)):
            lst[i] = lst[i] / len(self.unigrams)

        return lst, d

    # расчет перплексий для методов сглаживания
    def lpls_perplexity(self):
        lp, d = self.uni_lpls()
        perp = (1 / np.prod(lp)) ** (1 / len(self.test))
        return perp
    
    def lind_perplexity(self):
        ld, d = self.uni_lind()
        perp = (1 / np.prod(ld)) ** (1 / len(self.test))
        return perp


uni_model = Unigram(corpus, test)  # строим модель униграмм

pc, p = uni_model.uni_prob()
d = {'Words': p, 'Probabilities': pc}
df = pd.DataFrame(data=d, index=None)
print("Simple unigram probability:")
print(df, "\n")
# df.to_csv("uni.csv", encoding='utf-8-sig')

pc, p = uni_model.uni_lpls()
d = {'Words': p, 'Probabilities': pc}
df = pd.DataFrame(data=d, index=None)
print("Laplace unigram smoothing:")
print(df)
# df.to_csv("unilpls.csv", encoding='utf-8-sig')
print(uni_model.lpls_perplexity(), "\n")

uni_model.lmbd = 0.25  # меняем значение параметра
pc, p = uni_model.uni_lind()
d = {'Words': p, 'Probabilities': pc}
df = pd.DataFrame(data=d, index=None)
print("Lindstone unigram smoothing (lambda = 0.25):")
print(df, "\n")
# df.to_csv("unilind025.csv", encoding='utf-8-sig')
print("Perplexity: ", uni_model.lind_perplexity(), "\n")

uni_model.lmbd = 0.5  # меняем значение параметра
pc, p = uni_model.uni_lind()
d = {'Words': p, 'Probabilities': pc}
df = pd.DataFrame(data=d, index=None)
print("Lindstone unigram smoothing (lambda = 0.5):")
print(df, "\n")
# df.to_csv("unilind05.csv", encoding='utf-8-sig')
print("Perplexity: ", uni_model.lind_perplexity(), "\n")

uni_model.lmbd = 0.75
pc, p = uni_model.uni_lind()
d = {'Words': p, 'Probabilities': pc}
df = pd.DataFrame(data=d, index=None)
print(df)
# df.to_csv("unilind075.csv", encoding='utf-8-sig')
print(uni_model.lind_perplexity())

uni_model.lmbd = 1 # при лямбда=1 перплексия совпадает с Лапласом
print("Perplexity (lambda = 1): ", uni_model.lind_perplexity(), "\n")


class Bigram(object):  # класс для биграмм
    
    def __init__(self,  data, testing):
        self.text = data
        self.bigrams = []
        c = self.text.split()
        self.bigrams = [(s1, s2) for s1, s2 in zip(c, c[1:])]
        self.test = testing.split()
        c = self.test 
        self.test_bi = [(s1, s2) for s1, s2 in zip(c, c[1:])]
        self.lmbd = 1
    
    # исходные вероятности биграмм
    def bi_prob(self):
        lst = []
        bi = Counter(self.bigrams)
        uni = Counter(self.text.split())
        for bigram in bi:
            for unigram in uni:
                if bigram[0] == unigram: 
                    lst.append(bi[bigram] / uni[unigram])
        return bi.keys(), lst

    # сглаживание Лапласа
    def lpls_bi(self):
        V = len(list(set(self.bigrams))) 
        N = len(self.bigrams)

        bigrams_test = Counter(self.test_bi)
        unigrams_test = Counter(self.test)

        lpls = []
        d = []
        for bigram in bigrams_test.keys():
            for unigram in unigrams_test.keys():
                if bigram[0] == unigram: 
                    d.append(bigram)
                    if bigram in self.bigrams:
                        lpls.append((self.bigrams.count(bigram) + 1) / (self.text.split().count(unigram) + V ** 2))
                    else:
                        lpls.append(1 / (V ** 2))
        return d, lpls
    
    # сглаживание по формуле из презентации (в знаменателе N)
    def lind2_bi(self):
        B = len(list(set(self.bigrams)))    
        N = len(self.bigrams) 
        bigrams_test = Counter(self.test_bi)
        
        C = list(set(self.test_bi))
        unigrams_test = Counter(self.test)

        d = []
        lind = []
        for bigram in bigrams_test.keys():
            for unigram in unigrams_test.keys():
                if bigram[0] == unigram: 
                    d.append(bigram)
                    if bigram in self.bigrams:
                        lind.append((self.bigrams.count(bigram) + self.lmbd) / (N + B * self.lmbd))
                    else:
                        lind.append(self.lmbd / (N + B * self.lmbd))
        return d, lind
        
    # cглаживание через вторую вариацию закона Линдстоуна (с знаменателе считаем частотность первого слова в биграмме)
    def lind_bi(self):
        B = len(list(set(self.bigrams))) ** 2
        bigrams_test = Counter(self.test_bi)
        
        C = list(set(self.test_bi))
        unigrams_test = Counter(self.test)

        d = []
        lind = []
        for bigram in bigrams_test.keys():
            for unigram in unigrams_test.keys():
                if bigram[0] == unigram: 
                    d.append(bigram)
                    if bigram in self.bigrams:
                        lind.append((self.bigrams.count(bigram) + self.lmbd) / (self.text.split().count(unigram) + B * self.lmbd))
                    else:
                        lind.append(self.lmbd / (B * self.lmbd))
        return d, lind
        
    # расчет перплексий
    def lpls_perplexity(self):
        d, lp = self.lpls_bi()
        perp = (1 / np.prod(lp)) ** (1 / len(self.test_bi))
        return perp
    
    def lind_perplexity(self):
        d, ld = self.lind_bi()
        perp = (1 / np.prod(ld)) ** (1 / len(self.test_bi))
        return perp


bi_model = Bigram(corpus, test)

bi, bc = bi_model.bi_prob()
d = {'Words': bi, 'Probabilities': bc}
df = pd.DataFrame(data=d, index=None)
print("Simple bigram probability:")
print(df, "\n")
# df.to_csv("uni.csv", encoding='utf-8-sig')

bi, bc = bi_model.lpls_bi()
d = {'Words': bi, 'Probabilities': bc}
df = pd.DataFrame(data=d)
print("Laplace unigram smoothing:")
print(df, "\n")
# df.to_csv("unilpls.csv", encoding='utf-8-sig')
print("Perplexity: ", bi_model.lpls_perplexity(), "\n")

# первый вариант Линдстоуна
print("Lindstone Law with N")
bi_model.lmbd = 0.25
bi, bc = bi_model.lind2_bi()
d = {'Words': bi, 'Probabilities': bc}
df = pd.DataFrame(data=d)
print("Lindstone bigram smoothing (lambda = 0.25):")
print(df, "\n")
# df.to_csv("bi2025.csv", encoding='utf-8-sig')
print("Perplexity: ", bi_model.lind_perplexity(), "\n")

bi_model.lmbd = 0.5
bi, bc = bi_model.lind2_bi()
d = {'Words': bi, 'Probabilities': bc}
df = pd.DataFrame(data=d)
print("Lindstone bigram smoothing (lambda = 0.5):")
print(df, "\n")
# df.to_csv("bi205.csv", encoding='utf-8-sig')
print("Perplexity: ", bi_model.lind_perplexity(), "\n")

bi_model.lmbd = 0.75
bi, bc = bi_model.lind2_bi()
d = {'Words': bi, 'Probabilities': bc}
df = pd.DataFrame(data=d)
print("Lindstone bigram smoothing (lambda = 0.75):")
print(df, "\n")
# df.to_csv("bi2075s.csv", encoding='utf-8-sig')
print("Perplexity: ", bi_model.lind_perplexity(), "\n")

# второй вариант Линдстоуна
print("Lindstone Law with c(w_i)")
bi_model.lmbd = 0.25
bi, bc = bi_model.lind_bi()
d = {'Words': bi, 'Probabilities': bc}
df = pd.DataFrame(data=d)
print("Lindstone bigram smoothing (lambda = 0.25):")
print(df, "\n")
# df.to_csv("bi025.csv", encoding='utf-8-sig')
print("Perplexity: ", bi_model.lind_perplexity(), "\n")

bi_model.lmbd = 0.5
bi, bc = bi_model.lind_bi()
d = {'Words': bi, 'Probabilities': bc}
df = pd.DataFrame(data=d)
print("Lindstone bigram smoothing (lambda = 0.5):")
print(df, "\n")
# df.to_csv("bi05.csv", encoding='utf-8-sig')
print("Perplexity: ", bi_model.lind_perplexity(), "\n")

bi_model.lmbd = 0.75
bi, bc = bi_model.lind_bi()
d = {'Words': bi, 'Probabilities': bc}
df = pd.DataFrame(data=d)
print("Lindstone bigram smoothing (lambda = 0.75):")
print(df, "\n")
# df.to_csv("bi075.csv", encoding='utf-8-sig')
print("Perplexity: ", bi_model.lind_perplexity(), "\n")

bi_model.lmbd = 1
print("Perplexity (lambda = 1): ", bi_model.lind_perplexity(), "\n")

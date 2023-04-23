import nltk
import fasttext
import pandas as pd
import fasttext.util
import numpy as np
from spicy import stats
from numpy.linalg import norm
from prettytable import PrettyTable

# загружаем модель
ft = fasttext.load_model('cc.en.300.bin')

ws = open('wordsim_similarity.txt', 'r')
ws_lines = ws.readlines()

wr = open('wordsim_relatedness.txt', 'r')
wr_lines = wr.readlines()


# функция расчета косинуса между векторами
def get_cos(A, B):
    cosine = np.dot(A,B) / (norm(A) * norm(B))
    return cosine


# функция получения словаря вида "пара слов : близость"
def get_dicts(lines):
    d = { }
    d_ft = { }
    for i in range(len(lines)):
        words = lines[i].split()
        d[words[0] + ' ' + words[1]] = float(words[2])
        d_ft[words[0] + ' ' + words[1]] = get_cos(ft.get_word_vector(words[0]), ft.get_word_vector(words[1]))
    return d, d_ft


# вывод полученных значений таблицей
def out(d, d_ft):
    t = PrettyTable(['Word pair', 'Human', 'Cosine'])
    for key, value in d.items():
        t.add_row([key, value, d_ft[key]])
    print(t)


dict_ws, dict_ft_ws = get_dicts(ws_lines)
dict_wr, dict_ft_wr = get_dicts(wr_lines)
print("Similarity")
out(dict_ws, dict_ft_ws)
print()
print("Relatedness")
out(dict_wr, dict_ft_wr)

# из словаря получаем все значения в виде списка
l_ws = list(dict_ws.values())
l_ft_ws = list(dict_ft_ws.values())

l_wr = list(dict_wr.values())
l_ft_wr = list(dict_ft_wr.values())

# расчет корреляции с использованием библиотечной функции
s_res = stats.spearmanr(l_ws, l_ft_ws)
r_res = stats.spearmanr(l_wr, l_ft_wr)

print("Similarity correlation:", s_res.correlation)
print("Relatedness correlation:", r_res.correlation)


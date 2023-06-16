import pandas as pd
import os
import shutil
from os import listdir
from os.path import isfile, join
import math
import regex

def get_script_list():
    script_codes_df = pd.read_csv('../../miscellaneous/unicode-iso-15924-script_codes.csv', dtype=str)
    script_list = script_codes_df['script_code'].values

    return script_list

def script_detector(script_code, token):
    keywords = "\\p{script=" + script_code + "}"
    return bool(regex.match(keywords, token))

def script_count(selected_chars, char_scripts, script_list):
    counts = [0 for i in range(len(script_list))]
    for each_char in selected_chars:
        if each_char in char_scripts:
            counts[char_scripts[each_char]] += 1
        else:
            for i in range(len(script_list)):
                if script_detector(script_list[i], each_char):
                    counts[i] += 1
                    char_scripts[each_char] = i
                    break
    return counts

def detect_script_by_text(text, char_scripts, script_list):
    chars = [c for c in text]
    # random.shuffle(chars)
    char_count = len(chars)
    thresh = int(100 + 10 * math.log(char_count, 2))
    # thresh = 100
    if thresh < char_count:
        chars = chars[:thresh]
    counts = script_count(chars, char_scripts, script_list)
    sorted_counts = sorted(counts, reverse=True)
    return script_list[counts.index(sorted_counts[0])]

def get_str(fname):
    s = ''
    with open(path + fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            s += line.strip()
    return s

task_name = 'sentence_retrieval'
path = '../download_data/download/tatoeba-tmp/'
fnames = [f for f in listdir(path) if isfile(join(path, f)) if 'tatoeba' in f]

df = pd.read_csv('../../miscellaneous/lang.tab', sep='\t')
langs = df['Id']
langs2 = df['Part1']
d1 = {lang: str(lang2) for lang, lang2 in zip(langs, langs2)}
d1['eml'] = 0
d1['nah'] = 0
d2 = {lang2: str(lang) for lang, lang2 in zip(langs, langs2)}
d2['bat-smg'] = 'sgs'
d2['bh'] = 'bih'
d2['cbk-zam'] = 'cbk'
d2['fiu-vro'] = 'vro'
d2['roa-rup'] = 'rup'
d2['zh-classical'] = 'lzh'
d2['zh-min-nan'] = 'nan'
d2['zh-yue'] = 'yue'

lang_set = []
iso_set = []

langs = []
lang_list = 'sentence_retrieval_tatoeba.txt'
with open(lang_list, 'r') as f:
    lines = f.readlines()
    for line in lines:
        langs.append(line.strip())

for fname in fnames:
    code0 = fname.split('.')[-1]
    if code0 == 'eng':
        continue

    char_scripts = {}
    script_list = get_script_list()
    s = get_str(fname)
    result = detect_script_by_text(s, char_scripts, script_list)
    if 'ja' in fname or 'jpn' in fname:
        result = 'Jpan'
    elif 'zh-classical' in fname:
        result = 'Hani'
    if code0 in d1:
        code = code0
    elif code0 in d2:
        code = d2[code0]
    else:
        print(code0)
        continue
    if code in iso_set and code + '_' + result not in lang_set:
        print(code, result)
    iso_set.append(code)
    code = code + '_' + result
    lang_set.append(code)
    if code in langs:
        continue
    shutil.copy(path + fname, "/PATH/TO/DATA/" + task_name + "/" + fname.replace('.' + code0, '.' + code).replace('-eng', '-eng_Latn'))
    shutil.copy(path + fname.replace('eng.' + code0, 'eng.eng'), "/PATH/TO/DATA/" + task_name + "/" + fname.replace('eng.' + code0, 'eng.eng').replace('.' + code0, '.' + code).replace('-eng', '-eng_Latn').replace('.eng', '.eng_Latn'))

lang_set = sorted(list(set(lang_set)))
with open(task_name + '.txt', 'w') as f:
    for lang in lang_set:
        f.write('%s\n' % lang)
    

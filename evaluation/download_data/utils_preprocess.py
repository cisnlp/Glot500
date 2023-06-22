# coding=utf-8
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function

import argparse
import codecs
import csv
import json
import math
import os
import random
import shutil
from collections import defaultdict
from os import listdir
from os.path import isfile, join

import pandas as pd
import regex
from transformers import BertTokenizer, XLMRobertaTokenizer, XLMTokenizer

PANX_LANGUAGES = []
with open('ner_lang_list.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        PANX_LANGUAGES.append(line.strip())
UDPOS_LANGUAGES = []
with open('pos_lang_list.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        UDPOS_LANGUAGES.append(line.strip())

df = pd.read_csv('../../miscellaneous/lang.tab', sep='\t')
isos3 = df['Id']
isos1 = df['Part1']
iso3toiso1 = {iso3: str(iso1) for iso3, iso1 in zip(isos3, isos1)}
iso3toiso1['eml'] = 0
iso3toiso1['nah'] = 0
iso1toiso3 = {iso1: str(iso3) for iso3, iso1 in zip(isos3, isos1)}
iso1toiso3['bat-smg'] = 'sgs'
iso1toiso3['bh'] = 'bih'
iso1toiso3['cbk-zam'] = 'cbk'
iso1toiso3['fiu-vro'] = 'vro'
iso1toiso3['roa-rup'] = 'rup'
iso1toiso3['zh-classical'] = 'lzh'
iso1toiso3['zh-min-nan'] = 'nan'
iso1toiso3['zh-yue'] = 'yue'

def panx_preprocess(args):
    def _process_one_file(infile, outfile):
        lines = open(infile, 'r').readlines()
        if lines[-1].strip() == '':
            lines = lines[:-1]
        with open(outfile, 'w') as fout:
            for l in lines:
                items = l.strip().split('\t')
                if len(items) == 2:
                    label = items[1].strip()
                    idx = items[0].find(':')
                    if idx != -1:
                        token = items[0][idx+1:].strip()
                        fout.write(f'{token}\t{label}\n')
                else:
                    fout.write('\n')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for lg in PANX_LANGUAGES:
        for split in ['train', 'test', 'dev']:
            infile = os.path.join(args.data_dir, f'{lg}-{split}')
            outfile = os.path.join(args.output_dir, f'{split}-{lg}.tsv')
            _process_one_file(infile, outfile)

def udpos_preprocess(args):
    def _read_one_file(file):
        data = []
        sent, tag, lines = [], [], []
        for line in open(file, 'r'):
            items = line.strip().split('\t')
            if len(items) != 10:
                empty = all(w == '_' for w in sent)
                num_empty = sum([int(w == '_') for w in sent])
                if num_empty == 0 or num_empty < len(sent) - 1:
                    data.append((sent, tag, lines))
                sent, tag, lines = [], [], []
            else:
                sent.append(items[1].strip())
                tag.append(items[3].strip())
                lines.append(line.strip())
                assert len(sent) == int(items[0]), 'line={}, sent={}, tag={}'.format(line, sent, tag)
        return data

    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def remove_empty_space(data):
        new_data = {}
        for split in data:
            new_data[split] = []
            for sent, tag, lines in data[split]:
                new_sent = [''.join(w.replace('\u200c', '').split(' ')) for w in sent]
                lines = [line.replace('\u200c', '') for line in lines]
                assert len(" ".join(new_sent).split(' ')) == len(tag)
                new_data[split].append((new_sent, tag, lines))
        return new_data

    def check_file(file):
        for i, l in enumerate(open(file)):
            items = l.strip().split('\t')
            assert len(items[0].split(' ')) == len(items[1].split(' ')), 'idx={}, line={}'.format(i, l)

    def _write_files(data, output_dir, lang, suffix):
        for split in data:
            if len(data[split]) > 0:
                prefix = os.path.join(output_dir, f'{split}-{lang}')
                if suffix == 'mt':
                    with open(prefix + '.mt.tsv', 'w') as fout:
                        for idx, (sent, tag, _) in enumerate(data[split]):
                            newline = '\n' if idx != len(data[split]) - 1 else ''
                            # if split == 'test':
                            #     fout.write('{}{}'.format(' '.join(sent, newline)))
                            # else:
                            fout.write('{}\t{}{}'.format(' '.join(sent), ' '.join(tag), newline))
                    check_file(prefix + '.mt.tsv')
                    print('    - finish checking ' + prefix + '.mt.tsv')
                elif suffix == 'tsv':
                    with open(prefix + '.tsv', 'w') as fout:
                        for sidx, (sent, tag, _) in enumerate(data[split]):
                            for widx, (w, t) in enumerate(zip(sent, tag)):
                                newline = '' if (sidx == len(data[split]) - 1) and (widx == len(sent) - 1) else '\n'
                                fout.write('{}\t{}{}'.format(w, t, newline))
                            fout.write('\n')
                elif suffix == 'conll':
                    with open(prefix + '.conll', 'w') as fout:
                        for _, _, lines in data[split]:
                            for l in lines:
                                fout.write(l.strip() + '\n')
                            fout.write('\n')
                print(f'finish writing file to {prefix}.{suffix}')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for root, dirs, files in os.walk(args.data_dir):
        lg = root.strip().split('/')[-1]
        if root == args.data_dir or lg not in UDPOS_LANGUAGES:
            continue

        data = {k: [] for k in ['train', 'dev', 'test']}
        for f in sorted(files):
            if f.endswith('conll'):
                file = os.path.join(root, f)
                examples = _read_one_file(file)
                if 'train' in f:
                    data['train'].extend(examples)
                elif 'dev' in f:
                    data['dev'].extend(examples)
                elif 'test' in f:
                    data['test'].extend(examples)
                else:
                    print('split not found: ', file)
                print(' - finish reading {}, {}'.format(file, [(k, len(v)) for k,v in data.items()]))

        data = remove_empty_space(data)
        for sub in ['tsv']:
            _write_files(data, args.output_dir, lg, sub)


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

def get_tagging_content(fname):
    content = ''
    with codecs.open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split('\t')
            if len(items) == 2:
                content += items[0]
    return content

def get_retrieval_content(fname):
    content = ''
    with codecs.open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            content += line
    return content

def iso_convert_tagging(args):
    path = args.data_dir
    save_path = args.output_dir
    fnames = [f for f in listdir(path) if isfile(join(path, f))]

    lang_set = []
    iso_set = []
    for fname in fnames:
        char_scripts = {}
        script_list = get_script_list()
        content = get_tagging_content(path + fname)
        result = detect_script_by_text(content, char_scripts, script_list)
        if 'ja' in fname or 'jpn' in fname:
            result = 'Jpan'
        elif 'zh-classical' in fname:
            result = 'Hani'
        old_code = fname.split('.')[0].replace(fname.split('-')[0] + '-', '')
        if old_code in iso3toiso1:
            new_code = old_code
        elif old_code in iso1toiso3:
            new_code = iso1toiso3[old_code]
        else:
            continue
        iso_set.append(new_code)
        new_code = new_code + '_' + result
        lang_set.append(new_code)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        shutil.copy(path + fname, save_path + fname.replace('-' + old_code + '.', '-' + new_code + '.'))

def iso_convert_retrieval(args):
    path = args.data_dir
    save_path = args.output_dir
    fnames = [f for f in listdir(path) if isfile(join(path, f)) if 'tatoeba' in f]

    lang_set = []
    iso_set = []
    for fname in fnames:
        old_code = fname.split('.')[-1]
        if old_code == 'eng':
            continue

        char_scripts = {}
        script_list = get_script_list()
        content = get_retrieval_content(path + fname)
        result = detect_script_by_text(content, char_scripts, script_list)
        if 'ja' in fname or 'jpn' in fname:
            result = 'Jpan'
        elif 'zh-classical' in fname:
            result = 'Hani'
        if old_code in iso3toiso1:
            new_code = old_code
        elif old_code in iso1toiso3:
            new_code = iso1toiso3[old_code]
        else:
            continue
        iso_set.append(new_code)
        new_code = new_code + '_' + result
        lang_set.append(new_code)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        shutil.copy(path + fname, save_path + fname.replace('.' + old_code, '.' + new_code).replace('-eng', '-eng_Latn'))
        shutil.copy(path + fname.replace('eng.' + old_code, 'eng.eng'), save_path + fname.replace('eng.' + old_code, 'eng.eng').replace('.' + old_code, '.' + new_code).replace('-eng', '-eng_Latn').replace('.eng', '.eng_Latn'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                                            help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                                            help="The output data dir where any processed files will be written to.")
    parser.add_argument("--task", default="panx", type=str, required=True,
                                            help="The task name")
    parser.add_argument("--model_name_or_path", default="bert-base-multilingual-cased", type=str,
                                            help="The pre-trained model")
    parser.add_argument("--model_type", default="bert", type=str,
                                            help="model type")
    parser.add_argument("--max_len", default=512, type=int,
                                            help="the maximum length of sentences")
    parser.add_argument("--do_lower_case", action='store_true',
                                            help="whether to do lower case")
    parser.add_argument("--cache_dir", default=None, type=str,
                                            help="cache directory")
    parser.add_argument("--languages", default="en", type=str,
                                            help="process language")
    parser.add_argument("--remove_last_token", action='store_true',
                                            help="whether to remove the last token")
    parser.add_argument("--remove_test_label", action='store_true',
                                            help="whether to remove test set label")
    args = parser.parse_args()

    if args.task == 'panx_tokenize':
        panx_tokenize_preprocess(args)
    if args.task == 'panx':
        panx_preprocess(args)
    if args.task == 'udpos_tokenize':
        udpos_tokenize_preprocess(args)
    if args.task == 'udpos':
        udpos_preprocess(args)
    if args.task == 'udpos':
        udpos_preprocess(args)
    if args.task == 'tagging_iso_convert':
        iso_convert_tagging(args)
    if args.task == 'retrieval_iso_convert':
        iso_convert_retrieval(args)


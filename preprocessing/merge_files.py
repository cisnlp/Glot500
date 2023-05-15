import argparse
import codecs
import copy
import json
import logging
import os
import random
import sys
from datetime import datetime
from os import listdir

import pandas as pd
from datasets import load_from_disk


def write_file(input_fname, output_f, num):
    dataset = load_from_disk(input_fname)
    indexs = random.choices(list(range(len(dataset['train']))), k=num)
    for index in indexs:
        output_f.write('%s\n' % (dataset['train'][index]['text']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', type=str)
    parser.add_argument('--save_directory', type=str)
    parser.add_argument('--experiment_name',type=str)
    parser.add_argument("--lg_sampling_factor", type=float, default=-1, help="Language sampling factor")
    parser.add_argument("--scale", type=float, default=1e4, help="controls the minimum number of sentences of each language")

    args = parser.parse_args()

    lg2count = {}
    df = pd.read_csv('../miscellaneous/languages_stats.csv')
    lg2count = {str(lg) + '_' + script.replace("['", "").replace("']", ""): count for lg, script, count, is_seen in zip(df['language'], df['script'], df['new_length'], df['XLM-R']) if count >= 30000 and is_seen is not True}
    seen_lg2count = {str(lg) + '_' + script.replace("['", "").replace("']", ""): count for lg, script, count, is_seen in zip(df['language'], df['script'], df['new_length'], df['XLM-R']) if is_seen is True}
    print('%s unseen languages and %s seen languages' % (len(lg2count), len(seen_lg2count)))

    # downsample (S < 1) or upsample (S>1) the importance of high-resource languages
    S = args.lg_sampling_factor
    tot = sum([lg2count[lg] for lg in lg2count])
    tot_S = sum([lg2count[lg]**S for lg in lg2count])

    # minimum count and minimum probability
    min_c = min([lg2count[lg] for lg in lg2count])
    min_p = min([lg2count[lg]**S for lg in lg2count]) / tot_S
    tot_before_seen = sum([seen_lg2count[lg] for lg in seen_lg2count])
    tot_before_unseen = tot
    tot_after_seen = int(args.scale * min_c) * len(seen_lg2count)
    tot_after_unseen = sum([int(args.scale * min_c * (lg2count[lg]**S / tot_S / min_p)) for lg in lg2count])
    tot_before = tot_before_seen + tot_before_unseen
    tot_after = tot_after_seen + tot_after_unseen
    print('before resampling: %d sentences (%d for seen, %d for unseen) - after resampling: %d (%d for seen, %d for unseen)' % (tot_before, tot_before_seen, tot_before_unseen, tot_after, tot_after_seen, tot_after_unseen))
    print()

    for lg, count in lg2count.items():
        p = lg2count[lg]**S / tot_S
        print('%s - before resampling: %.2f, %d - after resampling: %.2f, %d' % (lg, 100.0 * lg2count[lg] / tot, lg2count[lg], 100.0 * p, int(args.scale * min_c * (p / min_p))))
    for lg, count in seen_lg2count.items():
        print('%s - before resampling: %d - after resampling: %d' % (lg, seen_lg2count[lg], int(args.scale * min_c)))

    # scale:
    output_fname = args.save_directory + args.experiment_name + '.txt'
    output_f = codecs.open(output_fname, 'w', encoding='utf-8')
    # Load files of seen langs for training tokenizer
    for lg in seen_lg2count:
        input_fname = args.data_directory + lg
        write_file(input_fname, output_f, int(args.scale * min_c))
    # Load files of unseen langs for training tokenizer
    for lg in lg2count:
        input_fname = args.data_directory + lg
        p = lg2count[lg]**S / tot_S
        write_file(input_fname, output_f, int(args.scale * min_c * (p / min_p)))


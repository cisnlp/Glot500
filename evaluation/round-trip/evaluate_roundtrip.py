import math
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import torch
from transformers import (
	XLMRobertaTokenizer,
	XLMRobertaForMaskedLM,
)

import simalign.simalign

class RoundTripEvaluator():
	def __init__(self, model, device="cpu", batch_size=25) -> None:
		self.model_name = model
		self.aligner = simalign.SentenceAligner(model=model, matching_methods="a", device=device)
		self.batch_size = batch_size
		self.max_branch_thresh = 5

	def align_all(self, sentence_group):
		tokenized_sent_set = [[] for s in range(len(sentence_group[0]))]
		for sent_set in sentence_group:
			for i, sent in enumerate(sent_set):
				sent_ids = self.aligner.embed_loader.tokenizer(sent)['input_ids'][1:-1]
				sent_tokens = self.aligner.embed_loader.tokenizer.convert_ids_to_tokens(sent_ids)
				tokenized_sent_set[i].append(" ".join(sent_tokens))

		all_aligns = []
		for i in tqdm(range(len(tokenized_sent_set))):
			if i + 1 < len(tokenized_sent_set):
				trg_i = i + 1
			else:
				trg_i = 0

			raw_pair_al = self.aligner.get_batch_word_aligns(
				tokenized_sent_set[i], 
				tokenized_sent_set[trg_i], 
				batch_size=self.batch_size
				)
			lang_pair_al = []
			for s_raw_al in raw_pair_al:
				s_al_dict = defaultdict(list)
				for p in s_raw_al['inter']:
					s_al_dict[p[0]].append((trg_i, p[1]))
				lang_pair_al.append(s_al_dict)
			all_aligns.append(lang_pair_al)

		return tokenized_sent_set, all_aligns

	def eval_all_roundtrip_sentences(self, sentence_set):
		aligns = []
		labels = []
		total_words = 0
		for i in range(len(sentence_set)):
			labels.append([0 for w in sentence_set[i]])
			total_words += len(sentence_set[i])

			if i + 1 < len(sentence_set):
				trg_i = i + 1
			else:
				trg_i = 0

			s1 = " ".join(sentence_set[i])
			s2 = " ".join(sentence_set[trg_i])
			raw_al = self.aligner.get_word_aligns(s1, s2)['inter']
			al_dict = defaultdict(list)
			for p in raw_al:
				al_dict[p[0]].append((trg_i, p[1]))
			aligns.append(al_dict)

		# iterate all words
		for si, s in enumerate(sentence_set):
			for wi in range(len(s)):
				if labels[si][wi] > 0: continue

				# DFS over words
				stack = [[(si, wi), (-1, -1)]]
				cur_parent = {}
				while len(stack) > 0:
					topw = stack.pop()
					w = topw[0]
					pw = topw[1]

					if w in cur_parent and cur_parent[w][0][0] > 0:
						continue
					else:
						cur_parent[w] = [pw, 0]

					for ch_w in aligns[w[0]][w[1]]:
						if ch_w not in cur_parent:
							new_w = [ch_w, w]
							stack.append(new_w)
							cur_parent[w][1] += 1
						elif cur_parent[ch_w][0][0] >= 0:
							# go up until the first word of the loop
							jmp_w = w
							labels[jmp_w[0]][jmp_w[1]] = 1

							while jmp_w != ch_w:
								jmp_w = cur_parent[jmp_w][0]
								labels[jmp_w[0]][jmp_w[1]] = 1

					while cur_parent[w][1] == 0 and cur_parent[w][0][0] >= 0:
						# print("del", w)
						cur_parent[w][0] = (-2, -2)
						cur_parent[pw][1] -= 1
						w = pw
						pw = cur_parent[w][0]
						# cur_parent.pop(w, None)

		num_in_loop = sum([sum(s) for s in labels])
		return num_in_loop, total_words

	def eval_first_roundtrip_sentences(self, sentence_set, sentence_aligns=None):
		aligns = []
		total_words = len(sentence_set[0])
		if sentence_aligns == None:
			for i in range(len(sentence_set)):
				if i + 1 < len(sentence_set):
					trg_i = i + 1
				else:
					trg_i = 0

				s1 = " ".join(sentence_set[i])
				s2 = " ".join(sentence_set[trg_i])
				raw_al = self.aligner.get_word_aligns(s1, s2)['inter']
				al_dict = defaultdict(list)
				for p in raw_al:
					al_dict[p[0]].append((trg_i, p[1]))
				aligns.append(al_dict)
		else:
			aligns = sentence_aligns

		labels = [0 for w in sentence_set[0]]
		# iterate all words
		for wi in range(len(sentence_set[0])):
			stack = set([wi])
			for lvl in range(len(sentence_set)):
				new_stack = [trg_w[1] for w in stack for trg_w in aligns[lvl][w]]
				stack = set(new_stack)
			if len(stack) > self.max_branch_thresh:
				continue
			if wi in stack:
				labels[wi] = 1

		num_in_loop = sum(labels)
		return num_in_loop, total_words

	def eval_test_set(self, sentence_group):
		total_in_loop = 0.
		total_words = 0.

		for sent_set in tqdm(sentence_group):
			tokenized_sent_set = []
			for sent in sent_set:
				sent_ids = self.aligner.embed_loader.tokenizer(sent)['input_ids'][1:-1]
				tokenized_sent_set.append(self.aligner.embed_loader.tokenizer.convert_ids_to_tokens(sent_ids))

			# correct, total = self.eval_all_roundtrip_sentences(tokenized_sent_set)
			correct, total = self.eval_first_roundtrip_sentences(tokenized_sent_set)
			total_in_loop += correct
			total_words += total

		if total_words == 0:
			return 0.0
		return round(total_in_loop / total_words, 4)

	def eval_batch_test_set(self, sentence_group):
		total_in_loop = 0.
		total_words = 0.

		tokenized_sent_group, all_aligns = self.align_all(sentence_group)

		for idx in range(len(sentence_group)):
			tokenized_sent_set = [l[idx].split() for l in tokenized_sent_group]
			sets_align = [al[idx] for al in all_aligns]

			correct, total = self.eval_first_roundtrip_sentences(tokenized_sent_set, sets_align)
			total_in_loop += correct
			total_words += total

		if total_words == 0:
			return 0.0
		return round(total_in_loop / total_words, 4)


if __name__ == "__main__":
	rte = RoundTripEvaluator("cis-lmu/glot500-base")

	sents = ["I have a book . ugly dog", "Ich habe ein buch . dog", "I have a book .", "J'ai un livre ."]

	scollection = [sents[:-1], sents[1:]]
	res = rte.eval_test_set(scollection)
	print("\n\nTest set Evaluation:", res)

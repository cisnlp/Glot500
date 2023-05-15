import argparse, sys, logging, copy, os, json, codecs
from datetime import datetime
import sentencepiece as spm
import sentencepiece_model_pb2 as sp_model
from os import listdir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fname', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--save_directory', type=str)
    parser.add_argument('--vocab_size', type=int)

    args = parser.parse_args()
    
    #Get new tokens
    input_fname = args.input_fname
    output_fname = args.save_directory + 'Glot500'
    cmd = '--input={} --model_prefix={} --vocab_size={} --model_type=unigram --character_coverage=1.0 --train_extremely_large_corpus=true'.format(input_fname, output_fname, args.vocab_size)
    spm.SentencePieceTrainer.train(cmd)

    #Load pretrained XLM-R SPM
    original_m = sp_model.ModelProto()
    original_m.ParseFromString(open(args.save_directory + 'sentencepiece.bpe.model', 'rb').read())
    new_m = sp_model.ModelProto()
    new_m.ParseFromString(open(output_fname + '.model', 'rb').read())

    add_cnt = 0 
    piece_d = {piece.piece: 0 for piece in original_m.pieces}
    for new_piece in new_m.pieces:
        if new_piece.piece not in piece_d:
            piece_to_add = sp_model.ModelProto().SentencePiece()
            # Add token
            piece_to_add.piece = new_piece.piece
            # Add token log-prob
            piece_to_add.score = new_piece.score
            original_m.pieces.append(piece_to_add)
            add_cnt += 1

    print('Add {} tokens'.format(add_cnt))
    logging.info('Add {} tokens'.format(add_cnt))
    
    new_spm_save_dir = args.save_directory + 'Glot500_extended_spm.model'
    with open(new_spm_save_dir, 'wb') as f:
        f.write(original_m.SerializeToString())
    
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)
    tokenizer.vocab_file = new_spm_save_dir
    tokenizer.sp_model.load(tokenizer.vocab_file)
    tokenizer.save_pretrained(args.save_directory + 'Glot500_extended_spm')


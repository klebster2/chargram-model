""" Authorship: Created at Edinburgh University:
 Kleber Noel (klebnoel@gmail.com) and Oliver Goldstein (oliverjgoldstein@gmail.com)
 This code was originally written for ANLP coursework to evaluate n-gram patterns in data. (Sept.-Nov. 2018)
 Kleber Noel changed the functionality of the code to a general case across European languages.
"""

import re
import sys
import random
from math import log
from collections import defaultdict
from collections import Counter
from collections import OrderedDict
from string import ascii_lowercase
import numpy as np
import math
import argparse
import pdb
import itertools

PAD_CHAR='#'

TRAIN_METHODS={
    'PLUS_ALPHA',
    'PLUS_ONE',
}

#class NGramModel():
#    def __init__(self):
#        pass
#
#
#

class NGramTrainer():
    
    def __init__(self, n):
        self.n = n
    
    def load_str_set(self, filename: str) -> str:
        """
        loads a set of characters to be used in a language model
        """
        entries=[]
        with open(filename, 'r') as f:
            for line in f.readlines():
                if line.strip() not in entries:
                    entries.append(re.sub('-', '\-', line.strip()))
        str_set = ''.join(entries)
        return str_set

    def preprocess_line(self, line: str) -> str:
        """ 
        This function currently removes all characters that are not in the set of:
        1. Some specified alphabet
        2. Spaces
        3. Digits
        4. Some specified punctuation set e.g. '.,:;@'
        
        This function also lowercases all characters and all digit
        are converted to zero.

        Treat each line as a separate sequence for 
        modeling purposes (rather than treating the entire 
        input file as a single sequence
        """

        line = re.sub(f'([^{self.char_set_str} ]|\n)', '', line)
        line = '##'+line+'#'
        return line

    def init_dictionary_with_regexp(self, n: int, dict_type: type) -> dict:
        """
        This function inits a dict to include 3 or 2 letter sequences.

        Can't have : _#_ or ###
        """
        char_set_re  = f'[{self.char_set_str}]'
        self.init_count_dict = defaultdict(dict_type)

        # All strings permissible for ngram:
        char_product = itertools.product(self.char_set_str + self.PAD_CHAR, repeat=n) 
        self.init_count_dict.update({''.join(i):dict_type(0) for i in char_product})
        # TODO: define throw-out strings (for now we are only using perplexity)

        init_count_dict = self.init_count_dict
        del self.init_count_dict
        return init_count_dict

    def update_state_count(self):
        """
        Initalise the using the possible key values found in trigram train dict

        Creates a dictionary of all emission counts
        transition-to-state counts accumulates counts of all ngrams to n-1gram 'state' class counts
        in a HMM this is the probability of being within a state (emission probability)
        """
        for k, v in self.ngram_trans_count.items():
            self.ngram_state_count[k[:self.n-1]]+=1

    def train(self, alpha:float, trainfile: str, alphabet:str, symbols:str, digits:str, pad_char:str) -> None:
        """
        returns None, but reads and updates dictionary counts based on a file
        """
        self.alphabet = self.load_str_set(alphabet)
        self.symbols = self.load_str_set(symbols)
        self.digits = self.load_str_set(digits)
        self.char_set_str = f'{self.alphabet}{self.digits}{self.symbols} '

        self.PAD_CHAR = pad_char
        # ngram trans: transition gram counts
        self.ngram_trans_count = self.init_dictionary_with_regexp(self.n, dict_type=int)
        # ngram state: are counts of seeing n-1 characters
        self.ngram_state_count = self.init_dictionary_with_regexp(self.n-1, dict_type=int)

        self.ngram_trans_prob = self.init_dictionary_with_regexp(self.n, dict_type=float) # probability dictionary mapping trigram keys to smoothed probabilities
        self.ngram_state_prob = self.init_dictionary_with_regexp(self.n-1, dict_type=float) # probability dictionary mapping bigram keys to smoothed probabilities
        self.ngram_trans_count = self.count_ngrams(trainfile, self.ngram_trans_count)
        self.update_state_count()

        # Generate probabilities from the trigrams and bigrams using present counts
        trigram_probabilities = self.add_alpha_smoothing(alpha)

    def count_ngrams(self, infile: str, ngram_count: dict) -> OrderedDict:
        """
        count ngrams from input file containing text for ngram counting
        for example, if we are counting ngrams, and the file text.txt
        input containing only:
            hello world
        we will have the following counts:
            ##h: 1
            #he: 1
            hel: 1
            ell: 1
            llo: 1
            lo : 1
            o w: 1
             wo: 1
            wor: 1
            orl: 1
            rld: 1
            ld#: 1
        """
        with open(infile) as f:
            for line in f:
                line = self.preprocess_line(line)
                if len(line) <= self.n:
                    # count caveat: ignore empty lines
                    continue

                for j in range(0, len(line)-(self.n-1)):
                    ngram = line[j:j+self.n]
                    ngram_count[ngram] += 1

        return OrderedDict(sorted(ngram_count.items(), key=lambda t: t[0]))
       
    def add_alpha_smoothing(self, alpha):
        """
        updates ngram_trans_probabilities
        """
        for ngram, count in self.ngram_trans_count.items():
            numerator                    =  count + alpha
            denominator                  =  self.ngram_state_count[ngram[:self.n-1]] + self.ngram_state_count[ngram[:self.n-1]] * alpha
            self.ngram_trans_prob[ngram] = numerator / denominator


    def test(self, test_file:str, test_level:str) -> None:
        """
        Prints perplexity according to level (either file or line by line)
        line-by-line level should print each line with each perplexity
        """
        if test_level=='file':
            self.ngram_test_dict = self.init_dictionary_with_regexp(self.n, int)
            self.ngram_test_dict = self.count_ngrams(test_file, self.ngram_test_dict)
            perplexity = self.perplexity_given_test_counts()
        elif test_level=='line':
            pass
        
        print("Final perplexity of " + str(test_file) + " with the training trigram probabilities is: " + str(perplexity))

    def perplexity_given_test_counts(self):
        
        log_probabilities = 0.0
        N = 0
        for k, v in self.ngram_test_dict.items():
            for p in range(0, int(v)):
                N += 1
                log_probabilities += log(self.ngram_trans_prob[k], 2)
            
        perplexity = math.pow(2, (float(-1)/float(N))*log_probabilities)
        return perplexity

    def write_model_to_file(self, file_out:str):
        f = open(file_out, "w+")
        for ngram, prob in self.ngram_trans_prob.items():
            f.write(str(ngram) + '\t'+ str(prob) + '\n')
        f.close()
        print(f"Model written to {file_out}")
    

    def model_generate_random_seq(self, seq_length:int, sequence='##', recursive_step=300):
        """
        generates a random sequence of specified length based on a model
        """
        if recursive_step == 0:
            return sequence
        else:
            # find partial key
            prev_two = {k:v for (k,v) in self.ngram_trans_prob.items() 
                        if k[:self.n-1]==(sequence[-2]+sequence[-1])}
            prev_two_vals = np.array([i for i in prev_two.values()])
            prev_two_vals /= prev_two_vals.sum()
            next_char = np.random.choice(list(prev_two.keys()), replace=True, p = list(prev_two_vals))[2]
            sequence += next_char
            
            if(next_char == '#'):
                sequence += '#'
                recursive_step -=1
            
            sequence = self.model_generate_random_seq(recursive_step - 1, sequence, seq_length)
            return sequence

        def read_char_probabilities(self, filename):
            self.ngram_trans_prob = default(float)
            with open(filename) as f:
                for line in f:
                    ngram, log_prob = line.strip().split("\t")
                    self.ngram_trans_prob.update({ngram: float(log_prob)})


def test_alphas(bigram_bin_size, bigram, trigram_train_dict, trigram_probabilities):
    """
    useful for testing models
    """
    alpha_vals = np.arange(0.001,1,0.001)

    for x in range(0, len(alpha_vals)):
        trigram_probabilities = add_alpha_smoothing(bigram_bin_size, bigram, trigram_train_dict, trigram_probabilities, alpha_vals[x])
        perplexity            = perplexity_given_tri_counts(trigram_probabilities, trigram_test_dict)
        print(str(alpha_vals[x]) + ", " + str(perplexity))

def test_bigram_bin_size(bigram, bigram_bin_size):
    for k, v in bigram.items():
        if "#" not in k:
            print("Bigram: " + k + " has bin size " + str(bigram_bin_size[k]))

def test_training_probabilities(trigram_probabilities):
    print(trigram_probabilities.items())

def test_sum_to_one(trigram_probabilities):
    sum_to_one = init_dictionary_with_regexp(0)
    for k,v in trigram_probabilities.items():
        sum_to_one[k[:2]] += trigram_probabilities[k]
    for k, v in sum_to_one.items():
        print(sum_to_one[k])


def get_args():
    parser = argparse.ArgumentParser(description="A program that can generate a ngram language model, with associated perplexities.")
    # add UNK token for LM
    parser.add_argument('--level', '-l', choices=['word','char'], help="Defines the level the trigram lm will work at. Level can either be word or character.")
    parser.add_argument('--train-file', '-t', action='store', help="train file")
    parser.add_argument('--number', '-n', action='store', type=int, default=3, help="ngram level")

    parser.add_argument('--test-file', '-e', action='store', help="test file")
    parser.add_argument('--test-level', '-x', action='store', help="test level")

    parser.add_argument('--alpha', action='store', type=float, default=0.7, help="alpha value for plus alpha smoothing")

    parser.add_argument('--file-out', '-o', action='store', help="path to model file out")

    parser.add_argument('--alphabet', '-a', action='store', help="alphabet dictionary")
    parser.add_argument('--digits', '-d', action='store', help="digit dictionary")
    parser.add_argument('--symbols', '-s', action='store', help="symbol dictionary")

    parser.add_argument('--random-sequence-length', '-r', action='store', type=int, help="length of random sequence based on trained/input model")

    args = parser.parse_args()
    return args


def main(args):
    
    tri_trainer = \
        NGramTrainer(
            n = args.number,
        )

    if args.train_file: 

        tri_trainer.train(
            alpha = args.alpha,
            trainfile = args.train_file,
            alphabet = args.alphabet,
            symbols = args.symbols,
            digits = args.digits,
            pad_char = PAD_CHAR,
        )

        tri_trainer.write_model_to_file(
            args.file_out,
        )

    elif args.model_file:

        tri_trainer.read_char_probabilities(
            args.model_file
        )
        
        # Tests 
        # test_bigram_bin_size(bigram, bigram_bin_size)
        # test_training_probabilities(trigram_probabilities)
        # test_sum_to_one(new_trigram_probabilities)

    if args.test_file and args.test_level:
        tri_trainer.test(
            test_file  = args.test_file,
            test_level = args.test_level,
        )

    if args.random_sequence_length:
        random_seq_new = \
            tri_trainer.model_generate_random_seq(
                seq_length = args.random_sequence_length,
        )
        if args.model_file:
            print(f"Random sequence generated by trigram generated from {args.model_file}")
        elif args.train_file:
            print(f"Random sequence generated by trigram generated from {args.train_file}")
        print(random_seq_new)

    return 1

if __name__ == "__main__":
    args = get_args()
    main(args)

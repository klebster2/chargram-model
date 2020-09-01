""" Authorship: Created at Edinburgh University:
 Kleber Noel (klebnoel@gmail.com) and Oliver Goldstein (oliverjgoldstein@gmail.com)
 This code was originally written for ANLP coursework to evaluate n-gram patterns in data. (Sept.-Nov. 2018)
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
import itertools
import os
import ast

class bcolors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'
    UNDERLINE = '\033[4m'

class NGramModel():

    def __init__(self, n, lower, pad, wildcard, diacritics_str, punctuations_str, str_set, verbose, ngram_trans_prob=OrderedDict()):
        """
        n: define n of (ngrams) to build a model 
        lower: lowercase input and model
        pad: considered meaningless (removed from raw string to pad HMM)
        diacritics: filepath to set of diacritics
        punctuations: filepath to punctuations
        verbose: boolean denoting verbose printing
        """
        self.n = n

        self.lower = lower
        self.pad = pad
        self.wildcard = wildcard

        self.verbose = verbose
        
        self.diacritics_str = diacritics_str
        self.punctuations_str = punctuations_str
        self.str_set = str_set
        self.ngram_trans_prob = ngram_trans_prob
         
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
        if self.lower == True:
            line = line.lower()
        line = re.sub(f'[0-9]', '0', line)
        line = re.sub(f'([^a-za-z0-9{self.diacritics_str}{self.punctuations_str} ]|[{self.pad}{self.wildcard}\n])', '', line)

        # wrap line in padding
        line = str(''.join([self.pad] * (self.n - 1))) + line + self.pad
        return line

    def count_ngrams(self, infile: str, ngram_count: dict, test_level: str) -> OrderedDict:
        """
        count ngrams from input file containing text for ngram counting
        for example, if we are counting ngrams, and the file text.txt
        input containing only:
            hello world
        we will have the following counts for a trigram char model:
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
                    # window string
                    ngram = line[j:j+self.n]
                    ngram_count[ngram] += 1

                if test_level == 'line':
                    self.ngram_test_dict.update(ngram_count)
                    pp = self.perplexity_given_test_counts()
                    print(line, '\t', pp)
                    ngram_count = defaultdict(int)
                    self.ngram_test_dict = defaultdict(int)
                elif test_level == 'file':
                    pass
        
        if test_level == 'file':
            self.ngram_test_dict.update(ngram_count)

        return OrderedDict(sorted(ngram_count.items(), key=lambda t: t[0]))
       

    def generate_random_seq(self, pretty_print:bool, steps_left:int, sequence='', step=0) -> str:
        """
        generates a random sequence of specified length based on a model trained, or input
        """
        full_sequence = ''
        rows, columns = os.popen('stty size', 'r').read().split()
        while steps_left > 0:

            step+=1
            if step==1:
                sequence = self.pad * (self.n - 1)
            
            possible_trans = {}
            # find partial key for transition
            for (trans, prob) in self.ngram_trans_prob.items():
                partial_key = trans[:(self.n-1)-len(self.wildcard)]
                partial_sequence = sequence[-(self.n-1):(len(sequence)-len(self.wildcard))]

                if partial_key == partial_sequence:
                    possible_trans.update({trans: prob})
            
            trans_log_probs = np.array([p for p in possible_trans.values()])
            trans_log_probs /= trans_log_probs.sum()
            next_char = np.random.choice(list(possible_trans.keys()), replace=True, p = list(trans_log_probs))

            if self.wildcard in next_char:
                # check the span that could contain wildcard(s)
                for _ in re.finditer(next_char[((self.n-1)-len(self.wildcard)):], self.wildcard):
                    sequence = sequence[:len(sequence)-len(self.wildcard)] + np.random.choice(list(self.str_set), replace=True)
            else:
                sequence = sequence[:len(sequence)-len(self.wildcard)] + next_char[((self.n-1)-len(self.wildcard)):]
            
            if(next_char[-1] == self.pad):
                # for trigram two pad chars come after one another (denoting line beginning)
                sequence += self.pad*(self.n-1)
                steps_left -= 1

            # when presenting output
            if pretty_print:
                if len(sequence)%(int(columns)-3)!=0: 
                    print(sequence, end='\r')
                else:
                    full_sequence+=sequence[:self.n-1]
                    sequence=sequence[:self.n-1]
                    print(re.sub(f'[{self.pad}]+', '\n', sequence), end='\n')
            steps_left-=1   
        print('\n\n')         
        full_sequence+=sequence
        
        return full_sequence

    def train(self, alpha:float, trainfile:str) -> None:
        """
        returns None, but reads and updates dictionary counts based on a file
        """
        # initialise count dicts
        # ngram trans: transition gram counts
        self.ngram_trans_count = defaultdict(int)
        # ngram state: state gram counts 
        self.ngram_state_count = defaultdict(int)
        # raw counts
        self.ngram_trans_count = self.count_ngrams(trainfile, self.ngram_trans_count, '')
        self.ngram_trans_count = self.create_wildcard_keys(self.ngram_trans_count, 1)
        # state counts
        self.update_state_count()

        # initialise probability dicts
        self.ngram_trans_prob = defaultdict(float)
        self.ngram_state_prob = defaultdict(float)
        # probability dictionary mapping trigram keys to smoothed probabilities
        self.ngram_trans_prob.update({k:float(0) for k in self.ngram_trans_count.keys()})
        # probability dictionary mapping bigram keys to smoothed probabilities
        self.ngram_state_prob.update({k:float(0) for k in self.ngram_state_count.keys()})

        # Generate probabilities from the trigrams and bigrams using present counts
        self.add_alpha_smoothing(alpha)


    def add_alpha_smoothing(self, alpha:float) -> None:
        """
        updates self.ngram_trans_prob dict
        setting alpha to zero will amount to MLE
        setting alpha to one will amount to add one smoothing
        """
        for trans, trans_count in self.ngram_trans_count.items():

            state = trans[:self.n-1]
            state_count = self.ngram_state_count[state]

            numerator =  trans_count + alpha
            denominator = state_count + (state_count * alpha)

            self.ngram_trans_prob.update({trans: (numerator / denominator)})
    
    def perplexity_given_test_counts(self) -> float:
        
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        ENDC = '\033[0m'
        UNDERLINE = '\033]4m'
        log_probabilities = 0.0
        N = 0
        def try_get_key(minus, key):
            """
            get key by using wildcard
            """
            if key not in self.ngram_trans_prob.keys():
                minus+=1
                key = str( key[:len(key)-minus] + self.wildcard*minus )
                if self.verbose:
                    print(bcolors.OKBLUE + key + bcolors.ENDC)
                return try_get_key(minus, key)
            else:
                if self.verbose:
                    print(bcolors.OKGREEN + key + bcolors.ENDC, end='\r')
                return key
            
        for k, v in self.ngram_test_dict.items():
            prob = self.ngram_trans_prob.get(try_get_key(0, k))
            if self.verbose:
                print(' ', prob, end='\n')
            for p in range(0, int(v)):
                N += 1
                log_probabilities += log(prob, 2)
            
        perplexity = math.pow(2, (float(-1)/float(N))*log_probabilities)
        return perplexity


class WordGramTrainer():
    def __init__(self):
        pass


class CharGramTrainer(NGramModel):
    
    def __init__(self, n:int, lower:bool, pad:str, wildcard:str, diacritics_file:str, punctuations_file:str, verbose:bool):
        self.n = n
        diacritics_str = self.load_str_set(diacritics_file)
        # punctuations can be special characters
        punctuations_str = self.load_str_set(punctuations_file)
        # all possible characters
        str_set = f'{ascii_lowercase}0{diacritics_str}{punctuations_str} {pad}'

        NGramModel.__init__(self, n, lower, pad, wildcard, diacritics_str, punctuations_str, str_set, verbose)
    
    def load_str_set(self, filename: str) -> str:
        """
        loads a set of characters to be used in a language model
        """
        entries=[]
        with open(filename, 'r') as f:
            for line in f.readlines():
                entry = line.strip()
                if entry not in entries:
                    entries.append(entry)
        str_set = ''.join(entries)
        return str_set


    def update_state_count(self) -> None:
        """
        Initalise the using the possible key values found in trigram train dict

        Creates a dictionary of all emission counts
        transition-to-state counts accumulates counts of all ngrams to n-1gram 'state' class counts
        in a HMM this is the probability of being within a state (emission probability)
        """
        for k, v in self.ngram_trans_count.items():
            self.ngram_state_count[k[:self.n-1]] += 1

    def create_wildcard_keys(self, ngram_count, n):
        """
        creates wildcard keys to compress model size (best for n>=4)
        wildcard keys hold counts/probabilities true for the an 
        uncommon character sequence if it didn't exist in the test set
        """
        keys_sets = defaultdict(set)
        
        for key in ngram_count.keys():
            keys_sets[key[:n]].add(key)
        
        # first step special case (some character doesn't exist at all!):
        if n==1 and len(keys_sets.keys())!=len(self.str_set):
            # wildcard only
            ngram_count[self.wildcard*self.n] = 0
        for key, key_set in keys_sets.items():
            if len(key_set)==len(self.str_set):
                continue
            else:
                # key set for each key substring
                ngram_count[key[:n]+self.wildcard*(self.n-n)] = 0

        # recursive logic
        if n==self.n-1:
            return ngram_count
        else:
            ngram_count.update(self.create_wildcard_keys(ngram_count, n+1))
            return ngram_count

    def test(self, test_file:str, test_level:str) -> None:
        """
        Prints perplexity according to level (either file or line by line)
        line-by-line level should print each line with each line perplexity
        """

        self.ngram_test_dict = defaultdict(int)
        if test_level=='file':
            self.ngram_test_dict = self.count_ngrams(test_file, self.ngram_test_dict, test_level)
            perplexity = self.perplexity_given_test_counts()
            print("Final perplexity of " + str(test_file) + " with the training trigram probabilities is: " + str(perplexity))

        elif test_level=='line':
            self.count_ngrams(test_file, self.ngram_test_dict, test_level)

    def write_model_to_file(self, file_out:str):
        with open(file_out, "w+") as f:
            f.write(f'N=({self.n}), d=({self.diacritics_str}), pct=({self.punctuations_str}), l={self.lower}, w={self.wildcard}, p={self.pad}\n')        
            for ngram, prob in self.ngram_trans_prob.items():
                f.write(str(ngram) + '\t'+ str(prob) + '\n')
        f.close()
        print(f"Model written to {file_out}")


def read_char_probability_file(file_in: str) -> NGramModel:
    """
    read probabilities from an input file
    """
    header_flag=True
    
    with open(file_in) as f:
        for line in f:
           if header_flag==True:
               matches = re.match('^N=\(([1-9][0-9]*?)\), d=\((.*?)\), pct=\((.*?)\), l=(.*?), w=(.*?), p=(.*?)$', line.rstrip())
               n, diacritics_str, punctuations_str, lower, wildcard, pad = matches.groups()
               n = ast.literal_eval(n)
               lower = ast.literal_eval(lower)
               header_flag=False

               model = NGramModel(
                   n = n,
                   lower = lower,
                   diacritics_str = diacritics_str,
                   punctuations_str = punctuations_str,
                   pad = pad,
                   wildcard = wildcard,
                   str_set = f'{ascii_lowercase}0{diacritics_str}{punctuations_str} {pad}',
                   verbose = False,
                   )

               model.ngram_trans_prob = defaultdict(float)

           else:
               ngram, log_prob = line.rstrip().split("\t")
               model.ngram_trans_prob.update({ngram: float(log_prob)})
    print(f"Model read from {file_in}")
    return model

    
def test_alphas(bigram_bin_size, bigram, trigram_train_dict, trigram_probabilities):
    """
    Test alphas by using an eval set.
    Find alpha that produces a minima in perplexity
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
    parser.add_argument('--test-level', '-x', default='', action='store', help="test level")

    parser.add_argument('--alpha', action='store', type=float, default=0.7, help="α vαalue for +α smoothing")

    parser.add_argument('--model-out', '-o', action='store', help="path to model file out")
    parser.add_argument('--model-in', '-i', action='store', help="path to model file in")

    parser.add_argument('--diacritics', '-d', action='store', help="other latin based character with diacritics for language. e.g. ç, é, etc.")
    parser.add_argument('--punctuations', '-p', action='store', help="punctuation to consider in input")
    parser.add_argument('--lower', action='store_true', help="lowercase input for training and/or testing")

    parser.add_argument('--pad', action='store', default='#', help="pad string for model")
    parser.add_argument('--wildcard', action='store', default='~', help="wildcard string for model compression")
    parser.add_argument('--random-sequence-length', '-r', action='store', type=int, help="length of random sequence based on trained/input model")

    parser.add_argument('--verbose', action='store_true', default=False, help="print messages")

    args = parser.parse_args()
    return args


def main(args):
    
    # Either we train, or load a pretrained model file
    if args.train_file: 
        # setup
        trainer = \
            CharGramTrainer(
                n = args.number,
                lower = args.lower,
                diacritics_file = args.diacritics,
                punctuations_file = args.punctuations,
                pad = args.pad,
                wildcard = args.wildcard,
                verbose = args.verbose
            )
        # train
        trainer.train(
            alpha = args.alpha,
            trainfile = args.train_file,
        )
        # output
        if args.model_out:
            trainer.write_model_to_file(
                args.model_out,
            )

        del trainer.ngram_trans_count
        del trainer.ngram_state_count
        del trainer.ngram_state_prob

        # get model directly from trainer for subsequent steps
        model = NGramModel(**trainer.__dict__)

        del trainer

    elif args.model_in:
        # load
        model = read_char_probability_file(
            args.model_in,
        )
        
    
    if args.test_file and args.test_level:
        # test a file
        model.test(
            test_file  = args.test_file,
            test_level = args.test_level,
        )

    if args.random_sequence_length:
        # generate a sequence
        random_seq_new = \
            model.generate_random_seq(
                steps_left = args.random_sequence_length,
                pretty_print = True,
        )
        
        if args.model_in:
            print(f"Random sequence generated by trigram generated from {args.model_in}")
        elif args.train_file:
            print(f"Random sequence generated by trigram generated from {args.train_file}")
        print(random_seq_new)

    # Unit tests 
    # test_bigram_bin_size(bigram, bigram_bin_size)
    # test_training_probabilities(trigram_probabilities)
    # test_sum_to_one(new_trigram_probabilities)

    return

if __name__ == "__main__":
    args = get_args()
    main(args)

# perplexity-toolkit

Commands can be run to create character level language models, evaluate test files, and generate a random sequence using a model. There is currently no functionality for word level models.

## Training:
Command to train a model:
```bash
python ngram_lm.py --level char --train-file my_train_file --number 10 --alpha 0.3 -o my_model --diacritics ./diacritics/en-GB --wildcard "~" --verbose --punctuations punctuations/set1 --lower
```

## Testing:
Command to test a model on a test file directly after training:
```bash
python ngram_lm.py --model-in gutenberg_model -r 100 --verbose
```

Command to test a model on a test file:
```bash
python ngram_lm.py --model-in gutenberg_model -r 100 --verbose
```

Command to test a model on a test file linewise:
```bash
python ngram_lm.py --model-in gutenberg_model -r 100 --verbose
```

## Generation:
Command to generate random sequence of 100 characters:
```bash
python ngram_lm.py --model-in my_model -r 100 --verbose
```

![alt text](https://github.com/klebster2/perplexity-toolkit/blob/master/gutenberg_example.gif "Logo Title Text 1")

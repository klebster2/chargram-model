# chargram model

Commands can be run to create character level language models, evaluate test files, and generate a random sequence using a model. There is currently no functionality for word level models.

## Why pad?

We pad in order to generate perplexity faster. Transition and emission probabilities over the HMM lattice our model will generate are precomputed and stored in the model file.

## How does the wildcard character work?

Models can become very large even for character LMs. The wildcard character reduces model size for add alpha smoothing. When preprocessing an input corpus, the wildcard character first is removed from input, then the wildcard is added to the end of any character sequence for which there were no observations in the training set. During testing, the wildcard character corresponds to character sequences that share the same smoothed perplexity.

## Training:

Use the command below to train a N-gram character model with add alpha smoothing and a corresponding alpha value of `alpha`:

The language model will be generated with reading information and probabilities.

```bash
python ngram_lm.py --level char --train-file <train_file> --number <N> --alpha <alpha> -o <model> --diacritics <diacritics_list> --wildcard "~" --verbose --punctuations <punctuation_list> --lower
```

If you don't have a corpus to use you can download an open source corpus such as the brown or gutenberg corpus from python's [nltk](https://www.nltk.org/).

For example running the following from a python interpreter will output the file `nltk_gutenberg` in your local directory.

```python
from nltk.corpus import gutenberg
import os

with open(os.path.join(os.getcwd(),'nltk_gutenberg'), 'w') as f:
    f.write(gutenberg.raw())

```

Then, training a 10-gram language model the file is just a case of running:

```bash
python ngram_lm.py --level char --train-file nltk_gutenberg --number 3 --alpha 0.3 -o gutenberg_model --diacritics ./diacritics/en-GB --wildcard "~" --verbose --punctuations punctuations/set1 --lower
```


## Testing:

Command to test a model on a test file:
```bash
python ngram_lm.py --test-file nltk_brown --test-level file --model-in gutenberg_model --verbose
```

Command to test a model on a test file linewise:
```bash
python ngram_lm.py --test-file nltk_brown --test-level line --model-in gutenberg_model --verbose
```

## Generation:

Command to generate random sequence of 100 characters:

```bash
python ngram_lm.py --model-in my_model -r 100 --verbose
```
The following example is sped up by x7. Generation using large models can be slow.

![gutenberg_example](https://github.com/klebster2/perplexity-toolkit/blob/master/gutenberg_example.gif "An example of generating characters using a 10-gram gutenberg model")

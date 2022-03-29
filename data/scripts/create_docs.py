import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import string
import json

def process(source, dst):
    doc_numbr = 0
    init = 0
    abstract = []
    count = 0
    punctuation = '“—’”'+string.punctuation
    translator = str.maketrans(punctuation, ' '*len(punctuation))

    with open(source, 'r') as dataset:
        for line in tqdm(dataset, desc='Reading dataset..'):
            orig_line = json.loads(line)

            line = orig_line['intent']
            line = line.encode("ascii", errors="ignore").decode().rstrip("\n").lower()

            line = line.translate(translator)
            line = line.rstrip()
            line = line.lstrip()

            if line != "":
                if doc_numbr < 50000:
                    with open('../docs/%d.txt'%(doc_numbr),'w') as doc_file:
                        doc_file.write(line)

                    doc_numbr = doc_numbr + 1
                else:
                    print("Docs created")
                    quit()

process("../conala-mined.jsonl", "./")

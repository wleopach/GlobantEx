from utils import *
import pandas as pd
dic = []
for f in filenames:
    x = getabstract(f)
    x = x.split()
    print(f)
    for word in x:
        if word not in dic:
            dic.append(word)

df = pd.DataFrame(dic)
df.to_csv('dictionary.csv', index = False)

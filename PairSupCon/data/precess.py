import pandas as pd
import numpy as np

df = pd.read_csv("./PairSupCon/data/nli_for_simcse.csv")
sent0 = df.sent0.values
sent1 = np.concatenate((sent0, sent0), axis=0)
sent2 = np.concatenate((df.sent1.values, df.hard_neg.values), axis=0)

dfnew = pd.DataFrame({'sentence1':sent1, 'sentence2':sent2, 'pairsimi': np.array([1]*len(sent0) + [0]*len(sent0))})
dfnew = dfnew.sample(frac=1, replace=False)
dfnew.to_csv("./PairSupCon/data/nli_train_posneg.csv", index=False)
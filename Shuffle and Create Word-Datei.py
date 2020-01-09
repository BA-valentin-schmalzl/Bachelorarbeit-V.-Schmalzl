import pandas as pd
import os
from sklearn.utils import shuffle
df = pd.read_excel('Ordner der Excel-Datei',
                   dtype={'Filename': str, 'Schadensbeschreibung': str, 'Schadensklasse': str})

df = shuffle(df)
# Gemischte Excel-Datei wird im selben Ordner wie diese Python-Datei gespeichert
path = os.path.join(os.getcwd(), 'shuffled_file.xlsx')
with pd.ExcelWriter(path) as writer:
    df.to_excel(writer, index=False)

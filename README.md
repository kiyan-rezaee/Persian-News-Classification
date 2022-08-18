# Persian-News-Classification

## Understanding Dataset : 

1. Dataset contains 10999 news in persian.
2. features of news are **NewsID**,  **Title**,     **Body**, **Date**, **Time**, **Category**, **Category2**

## Catergory :
 
|   Persian  |        English        |
|:----------:|:---------------------:|
|   آموزشي   |      educational      |
|   اجتماعي  |         social        |
|   اقتصادي  |        economic       |
|   بهداشتي  |        hygienic       |
|   تاريخي   |       historical      |
|    سياسي   |       political       |
|    علمي    |       scientific      |
|   فرهنگي   |        cultural       |
| فقه و حقوق | Law and Jurisprudence |
|    مذهبي   |       religious       |
|    ورزشي   |        athletic       |

## preprocessing : 
1. stopwords.txt contains a list of stop words that we use them in preprocessing to eliminate Nonsignificant words. We can also use **nltk** to download stopwords in English and 14 languages more.

2. I use Stemmer and Lemmatizer and word_tokenize implemented in [**hazm**](https://github.com/sobhe/hazm) Library.

3. I use **bag of words** technique for that I use this two packages from sklearn : 
>from sklearn.feature_extraction.text import TfidfVectorizer

>from sklearn.preprocessing import LabelEncoder 

Due to fact, I use **SVC** model so values of **category** columns should also be numerical.

## Evaluation of model

```
              precision    recall  f1-score   support

           0       0.85      0.91      0.88       246
           1       0.55      0.69      0.61       261
           2       0.82      0.80      0.81       246
           3       0.84      0.90      0.87       243
           4       0.91      0.85      0.88       244
           5       0.76      0.74      0.75       274
           6       0.82      0.74      0.78       242
           7       0.85      0.89      0.87       242
           8       0.91      0.85      0.88       258
           9       0.98      0.86      0.92       274
          10       0.98      0.95      0.97       220

    accuracy                           0.83      2750
   macro avg       0.84      0.83      0.84      2750
weighted avg       0.84      0.83      0.83      2750

```

### confusion Matrix : 
```
[[234  10   0   1   0   2   9   1   1   0   1]
 [ 14 179  21   6   8   9   7  16   5   4   1]
 [  3  26 200   0   4   5  12   3   1   2   0]
 [  1   7   4 233   0   2   8   0   0   0   0]
 [  0   4   1   0 219   9   1   2   3   1   0]
 [  2   9  17   1   7 172   5   4  14   2   1]
 [ 14   8   9  25   0   3 201   3   0   0   2]
 [  2  12   5   1   4   5   5 199   1   4   4]
 [  1   8   3   1   3  10   1   2 224   1   0]
 [  0   6   2   1   1   2   0   2   1 237   0]
 [  1   8   0   1   0   3   2   2   0   0 206]]
```

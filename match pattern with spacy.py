'''
envs:
    conda install -c conda-forge spacy
    python -m spacy download en
'''
import spacy
nlp = spacy.load('en')  # load text object
# doc = nlp("Tea is healthy and calming, don't you think?")   # doc is a document object, containing tokens
# 遍历输出doc
# for token in doc:
#     print(token)

# # 检查lemma, stopword
# print("Token\t\tLemma\t\tStopword")
# print('-'*40)
# for token in doc:
#     print(f"{str(token)}\t\t{token.lemma_}\t\t{token.is_stop}")
from spacy.matcher import PhraseMatcher

matcher = PhraseMatcher(nlp.vocab, attr='LOWER')    # 构建模型
terms = ['Galaxy Note', 'iPhone 11', 'iPhone XS', 'Google pixel']
patterns = [nlp(text) for text in terms]    # 为要匹配的词构建模型
matcher.add("TerminologyList", patterns)    # 给matcher增加一个属性，该属性的ID为TermologyList

text_doc = nlp("Glowing review overall, and some really interesting side-by-side "
               "photography tests pitting the iPhone 11 Pro against the "
               "Galaxy Note 10 Plus and last year’s iPhone XS and Google Pixel 3.")   # 待匹配的文本

matches = matcher(text_doc)     #  用matcher规则来找text_doc中的内容

print(matches)
match_id, start, end = matches[0]
print(nlp.vocab.strings[match_id], text_doc[start:end])

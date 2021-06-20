""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]



#==========加上自己字典的特殊pinyin符号.

with open('madarin_lexicon.txt') as f:
    tmp=f.readlines()
    tmp=[i.strip().split(' ')[1:] for i in tmp]
tmp2=[]
for i in tmp:
    tmp2+=i
print(tmp2)
tmp2=list(set(tmp2))
print(len(tmp2))



_pinyin = ["@" + s for s in pinyin.valid_symbols]#===========这个地方要自己添加.
print('old',len(_pinyin))
print(_pinyin)
_pinyin += ["@" + s for s in tmp2]#===========这个地方要自己添加.

print(_pinyin)
print('new',len(_pinyin))
pass
print(1)




# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    + _silences
)
# print("打印全的不symbols",symbols)
with open("当前使用的symbols是",'w')as f :
    f.write(str(symbols))
#=============symbols要自己手动加入自己需要的汉语拼音才行!!!!!!!!
""" from https://github.com/keithito/tacotron """
print(1)


from text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]#===========这个地方要自己添加.


#==========加上自己字典的特殊pinyin符号.

with open('madarin_lexicon.txt') as f:
    tmp=f.readlines()
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
print("打印全的不symbols",symbols)
with open("当前使用的symbols是",'w')as f :
    f.write(str(symbols))
#=============symbols要自己手动加入自己需要的汉语拼音才行!!!!!!!!
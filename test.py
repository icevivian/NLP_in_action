# encoding=utf-8
a=[{'a':1,'b':4,'c':2}, {'a':1,'b':1,'c':5}]
print(a)
b = sorted(a, key=itemgetter('c'),reverse=True)
print(b)
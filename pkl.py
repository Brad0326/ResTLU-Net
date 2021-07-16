import pickle

# 重点是rb和r的区别，rb是打开2进制文件，文本文件用r
f = open('/media/user/TOSHIBA EXT/2021/SecondYG/6_25/feihong_6_21out/attention/only_test_out/Dice.pkl','rb')
data = pickle.load(f)
print(data)
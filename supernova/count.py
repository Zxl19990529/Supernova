import os

count =0 
flag = 1
for line in open('submit_bak.csv'):
    print(line)
    if flag:
        flag =0
        continue
    spt = line.strip().split(',')
    print(spt)
    t = int(spt[-1])
    if t == 1:
        count +=1

print(count)
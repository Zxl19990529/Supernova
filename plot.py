import matplotlib.pyplot as plt
import numpy as np

score_01 = []
score_001 = []
iter_s = []
loss = []
epoch_ = []
mAP = []
threhod_01 = 0.1
threhod_001 = 0.01
tmp =0
for line in open('./20190405_195004.log'):
    spt = line.strip().split(' ')
    loss_ = float(spt[-1])
    if 'mAP:' in spt:
        continue
    epoch = spt[6]
    epoch = epoch.strip().split('2259')# ['[119][1000/', ']\tlr:']
    # iter_s.append(int())
    it = epoch[0].split('[')[-1].split('/')[0]
    it = int(it)
    # it = it.
    epoch = epoch[0].split(']')
    epoch = epoch[0].split('[')[-1]
    epoch = int(epoch)
    if tmp != epoch:
            epoch_.append(epoch)
            loss.append(loss_)
    tmp = epoch
    iteration = (epoch-1)*2259 + it
    iter_s.append(iteration)
#     loss.append(loss_)
#     # print('epoch:',epoch,'it:',it,'loss: ',loss_)

# count = 1
# for line in open('./today.log'):
#     spt = line.strip().split(' ')
#     if 'mAP:' in spt:
#         epoch.append(count)
#         count+=1
#         mAP.append(float(spt[-1]))
#         # print(spt)

# for line in open('./0406/log_map_next.txt'):
#     spt = line.strip().split(' ')
#     # print(spt)
#     if float(spt[1]) == threhod_01:

#         epoch.append(int(spt[0]))
#         mAP.append(float(spt[2]))
#         # score_01.append(float(spt[-1])
#         score_01.append(float(spt[-1]))
#     if float(spt[1]) == threhod_001:
#             score_001.append(float(spt[-1]))
plt.xlabel('it')
plt.ylabel('loss')
# plt.plot(epoch,mAP,color='blue',label='mAP')
# plt.plot(epoch,score_01,color='cyan',label='score_01')
# plt.plot(epoch,score_001,color='green',label='score_001')
plt.plot(epoch_,loss)
plt.legend()
plt.show()

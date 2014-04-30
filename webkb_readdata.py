import numpy as np
import pickle

basepath = '/Users/changyale/dataset/webkb/'

link = []
web_id = []
web_word = []

univ = ['cornell','texas','washington','wisconsin']
label_univ = []

topic = ['course','faculty','project','staff','student']
label_topic = []

for i in univ:
    for j in ['.cites','.content']:
        file_name = basepath+i+j
        inp = open(file_name).readlines()
        
        # link information
        if j == '.cites':
            for line in inp:
                item = str.split(line)
                assert len(item) == 2
                link.append(item)              
        
        # content information
        if j == '.content':
            for line in inp:
                item = str.split(line)
                assert len(item) == 1705
                web_id.append(item[0])
                web_word.append(item[1:-1])
                label_topic.append(topic.index(item[-1]))
                label_univ.append(univ.index(i))

# web_id: list, len(877)
# label_topic: list, len(877)
# label_univ: list, len(877)

# web_word: array, shape(877,1703)
web_word = np.double(np.array(web_word))

counter_univ = np.zeros((4,4))
# construct link matrix from link information
aff_link = np.identity(len(web_id))
for i in range(len(link)):
    j = web_id.index(link[i][0])
    k = web_id.index(link[i][1])
    aff_link[j,k] += 1
    aff_link[k,j] += 1
    counter_univ[label_univ[j],label_univ[k]] += 1

# Save results to pickle file
res = [web_id,web_word,aff_link,label_univ,label_topic,univ,topic]

file_pkl = open("webkb.pkl","wb")
pickle.dump(res,file_pkl)
file_pkl.close()

counter = np.zeros((4,5))
# Count number of webpages belonging to each class
for i in range(len(label_univ)):
    counter[label_univ[i],label_topic[i]] += 1

            


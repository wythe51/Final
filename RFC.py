
# coding: utf-8

# In[1]:

import numpy as np
import sys
from sklearn import preprocessing, ensemble


# In[2]:

output_name = 'test.out'  #sys.argv[1]
train_file = open('train', 'r')
test_file = open('test.in','r')
attack_type_file = open('training_attack_types.txt', 'r')
label_index_file = open('label_index.txt', 'r')

label_index_dict = {}
for line in label_index_file.readlines():
    label_index_dict[line.split()[0]] = int(line.split()[1])
label_index_file.close()

attack_index_dict = {'normal':0}
for line in attack_type_file.readlines():
    attack_index_dict[line.split()[0]] = label_index_dict[line.split()[1]]
attack_index_list = attack_index_dict.keys()
attack_type_file.close()

train_line_list = train_file.readlines()
train_file.close()

test_line_list = test_file.readlines()
test_file.close()


# In[3]:

trainN = len(train_line_list)
testN = len(test_line_list)
print('# of train data = ' + str(trainN))
print('# of test data = ' + str(testN))


# In[4]:

disc_idx = [1,2,3]
disc_list = [['udp', 'icmp', 'tcp'],
             ['aol', 'urp_i', 'netbios_ssn', 'Z39_50', 'smtp', 'domain', 'private', 'echo', 'printer', 'red_i', 'eco_i', 'ftp_data', 'sunrpc', 'urh_i', 'uucp', 'pop_3', 'pop_2', 'systat', 'ftp', 'sql_net', 'whois', 'tftp_u', 'netbios_dgm', 'efs', 'remote_job', 'daytime', 'pm_dump', 'other', 'finger', 'ldap', 'netbios_ns', 'kshell', 'iso_tsap', 'ecr_i', 'nntp', 'http_2784', 'shell', 'domain_u', 'uucp_path', 'courier', 'exec', 'tim_i', 'netstat', 'telnet', 'gopher', 'rje', 'hostnames', 'link', 'ssh', 'http_443', 'csnet_ns', 'X11', 'IRC', 'harvest', 'login', 'supdup', 'name', 'nnsp', 'mtp', 'http', 'ntp_u', 'bgp', 'ctf', 'klogin', 'vmnet', 'time', 'discard', 'imap4', 'auth', 'http_8001'],
             ['OTH', 'RSTR', 'S3', 'S2', 'S1', 'S0', 'RSTOS0', 'REJ', 'SH', 'RSTO', 'SF']]


# In[5]:

trainX = np.zeros((trainN, 41))
trainY = np.zeros((trainN,1))
for i in range(trainN):
    items = train_line_list[i].replace('\n','')[0:-1].split(',')
    for j in disc_idx:
        disc_array = disc_list[disc_idx.index(j)]
        if items[j] in disc_array:
            items[j] = disc_array.index(items[j])
        else:
            items[j] = len(disc_array)
    for k in range(41):
        trainX[i][k] = float(items[k])
    trainY[i] = attack_index_list.index(items[41])
print('Training data matrix is prepared.')


# In[6]:

one_hot_enc = preprocessing.OneHotEncoder(categorical_features=np.array(disc_idx),handle_unknown='ignore')
enc_trainX = one_hot_enc.fit_transform(trainX)
print('One-hot encoding of training data is done.')


# In[7]:

num_class = len(attack_index_list)+1
rfc = ensemble.RandomForestClassifier(n_estimators=num_class)
rfc.fit(enc_trainX, trainY.ravel())
print('Random forest classifier is trained.')


# In[8]:

testX = np.zeros((testN, 41))
for i in range(testN):
    items = test_line_list[i].replace('\n','')[0:-1].split(',')
    for j in disc_idx:
        disc_array = disc_list[disc_idx.index(j)]
        if items[j] in disc_array:
            items[j] = disc_array.index(items[j])
        else:
            items[j] = len(disc_array)
    for k in range(41):
        testX[i][k] = float(items[k])
print('Testing data matrix is prepared.')


# In[9]:

enc_testX = one_hot_enc.transform(testX)
print('One-hot encoding of testing data is done.')


# In[10]:

testY = rfc.predict(enc_testX)
print('Random forest classifier prediction is finished.')


# In[11]:

output_file = open(output_name, 'w')
output_file.write('id,label\n')
for i in range(testN):
    one_y = attack_index_dict[attack_index_list[int(testY[i])]]
    output_file.write(str(i+1) + ',' + str(one_y) + '\n')
output_file.close()
print('Output file completed.')


# In[ ]:




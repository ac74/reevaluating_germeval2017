import urllib.request
import os

os.mkdir("../data")

print('Beginning file download with urllib2...')

train = 'http://ltdata1.informatik.uni-hamburg.de/germeval2017/train-2017-09-15.xml'
dev = 'http://ltdata1.informatik.uni-hamburg.de/germeval2017/dev-2017-09-15.xml'
test_syn = 'http://ltdata1.informatik.uni-hamburg.de/germeval2017/test_syn-2017-09-15.xml'
test_dia = 'http://ltdata1.informatik.uni-hamburg.de/germeval2017/test_dia-2017-09-15.xml'

urllib.request.urlretrieve(train, '../data/train-2017-09-15.xml')
urllib.request.urlretrieve(dev, '../data/dev-2017-09-15.xml')
urllib.request.urlretrieve(test_syn, '../data/test_syn-2017-09-15.xml')
urllib.request.urlretrieve(test_dia, '../data/test_dia-2017-09-15.xml')

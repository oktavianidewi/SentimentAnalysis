try:
    # for python 3.5
    from urllib.request import Request, urlopen, URLError
    # belum di-test, package matematika adanya di 2.7
except ImportError:
    # for python 2.7
    import urllib
    import urllib2
    import requests

test_data_url = 'https://dl.dropboxusercontent.com/u/8082731/datasets/UMICH-SI650/testdata.txt'
train_data_url = 'https://dl.dropboxusercontent.com/u/8082731/datasets/UMICH-SI650/training.txt'

test_data_file = 'test_data.csv'
train_data_file = 'train_data.csv'

test_data = urllib.urlretrieve(test_data_url, test_data_file)
train_data = urllib.urlretrieve(train_data_url, train_data_file)
print("succesfully downloaded the dataset")
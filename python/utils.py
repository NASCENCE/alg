import cPickle as pickle
import gzip

def save_pickle_gzip(data, filename):
    #print "saving to", filename

    f = gzip.GzipFile(filename, 'wb')
    f.write(pickle.dumps(data, 1))
    f.close()

    #print "done"

def load_pickle_gzip(filename):
    f = gzip.GzipFile(filename, 'rb')
    buffer = ""
    while 1:
        data = f.read()
        if data == "":
            break
        buffer += data
    obj = pickle.loads(buffer)
    return obj

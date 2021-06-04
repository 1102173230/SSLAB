import sys
sys.path.append("..")
from RecQ import RecQ
from tool.config import Config
from visual.display import Display



if __name__ == '__main__':
    import time
    s = time.time()
    try:
        conf = Config('../config/'+ 'self_FBPR_LN' +'.conf')
    except KeyError:
        print('Error num!')
        exit(-1)
    recSys = RecQ(conf)
    recSys.execute()
    e = time.time()
    print("Run time: %f s" % (e - s))

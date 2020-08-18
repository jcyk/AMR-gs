import sys

fname = sys.argv[1]
ofname = fname + '.raw'

idx = 0
with open(ofname, 'w') as fo:
    for line in open(fname).readlines():
        fo.write("# ::id %d\n"%(idx, ))
        fo.write("# ::snt %s\n"%(line.rstrip()))
        fo.write("(d / dummy)\n\n")
        idx += 1
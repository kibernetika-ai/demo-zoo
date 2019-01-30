
import PIL.Image as Image
import os
import glob
import math

names = {}
seq = {}
def short(name):
    p = name.split('_')
    if len(name)>7:
        if len(p)>1:
            c = p[0]
        else:
            c = name.split('-')[0]
    else:
        c = name
    s = seq.get(c,0)
    if s==0:
        names[c] = name
        seq[c] = 1
    else:
        c = '{}{}'.format(c,s)
        names[c] = name
        seq[c] = s+1
    return c

def main():
    line = []
    for f in glob.glob('./data/styles/*.jpg'):
        i = Image.open(f)
        w,h = i.size
        if w<h:
            ratio = 256.0/w
            h = int(math.ceil(ratio*h))
            w = 256
        else:
            ratio = 256.0/h
            w = int(math.ceil(ratio*w))
            h = 256
        i = i.resize((w,h),resample=Image.BILINEAR)
        wc = (w-256)//2
        hc = (h-256)//2
        w,h = i.size
        i = i.crop((wc, hc, min(256+wc,w), min(256+hc,h)))
        i = i.resize((256,256),resample=Image.BILINEAR)
        name = os.path.basename(f)
        c = name.split('.')[0]
        c = short(c)
        i.save('./data/styles-256/'+name)
        line.append('{}[https://storage.googleapis.com/edwindemo/styles-256/{}]'.format(c,name))
    print(','.join(line))
    print('\n\n')
    print(names)


if __name__ == '__main__':
    main()
import sys
import glob
import os

jobfile = sys.argv[1]
logfi = sys.argv[2]
keyword = sys.argv[3]


if os.path.isdir(logfi):
    files = glob.glob(logfi+'*')
else:
    files = [logfi]



done = set()
for fi in files:
    with open(fi) as fin:
        for line in fin:
            if keyword in line:
                filename = line.strip().split()[-1]
                if '/' in filename:
                    done.add(filename[filename.rfind('/'):-1])
                else:
                    done.add(filename[1:-1])

with open(jobfile) as fin, open(jobfile+'.partial','w') as fout:
    for line in fin:
        line = line.strip()

        if line[line.rfind('/'):] in done or line[line.rfind('/')+1:] in done:
            continue
        else:
            fout.write(line+'\n')

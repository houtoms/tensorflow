from __future__ import print_function
import sys
import re
import collections
from collections import defaultdict

###############################################
def filenameToTestname(filename):
    sobj_net = re.search(r'output_(.*)_b', filename)
    sobj_batchsize = re.search(r'_b(\d*)_', filename)
    sobj_nGPU = re.search(r'_(\d*)gpu', filename)

    if(sobj_net):
        net = sobj_net.group(1)
    else:
        net = 'NOT FOUND'
    if(sobj_batchsize):
        batchsize = sobj_batchsize.group(1)
    else:
        batchsize = 'NOT FOUND'
    if(sobj_nGPU):
        nGPU = sobj_nGPU.group(1)
    else:
        nGPU = 'NOT FOUND'

    if((isinstance(nGPU, int)) and (isinstance(batchsize, int))):
        totalBatchsize = 'NOT FOUND'
    else:
        totalBatchsize = int(batchsize) * int(nGPU)
    testname = '{}, Total Batchsize {}, nGPUs {}'.format(net, totalBatchsize, nGPU)
    return testname

###############################################
def generate_scaling_numbers(data):
  for key in list(data):
    for i in range(0,4):

      if data[key][0] != []:
        if (data[key][i] != []) and (data[key][0][0] != 0):
          strongScale = round(float(data[key][i][0]) / data[key][0][0], 2)
          data[key][i][1] = strongScale

        newKey = (key[0], key[1] * pow(2, i), key[2])
        if (data[newKey] != []) and (data[key][0][0] != 0):
          weakScale = round(float(data[newKey][i][0]) / data[key][0][0], 2)
          data[key][i][2] = weakScale
###############################################
def print_csv(filename, data):
  openFile=open(filename,'w')
  
  openFile.write("Network,GPU,Batch,#GPU1,#GPU2,#GPU4,#GPU8,Strong Scaling 2 GPUs, Strong Scaling 4 GPUs, Strong Scaling 8 GPUs, Weak Scaling 2 GPUs, Weak Scaling 4 GPUs, Weak Scaling 8 GPUs\n")
  for net_batch in data:
    if (data[net_batch] == []):
      continue
    openFile.write(net_batch[0])
    openFile.write(',')
    openFile.write(net_batch[2])
    openFile.write(',')
    openFile.write(str(net_batch[1]))
    openFile.write(',')
    #Print ImgPerSec
    if (data[net_batch][0] != []):
      openFile.write(str(data[net_batch][0][0]))
    openFile.write(',')
    if (data[net_batch][1] != []):
      openFile.write(str(data[net_batch][1][0]))
    openFile.write(',')
    if (data[net_batch][2] != []):
      openFile.write(str(data[net_batch][2][0]))
    openFile.write(',')
    if (data[net_batch][3] != []):
      openFile.write(str(data[net_batch][3][0]))
    openFile.write(',')
    #Print strong scaling
    if (data[net_batch][1] != []):
      openFile.write(str(data[net_batch][1][1]))
    openFile.write(',')
    if (data[net_batch][2] != []):
      openFile.write(str(data[net_batch][2][1]))
    openFile.write(',')
    if (data[net_batch][3] != []):
      openFile.write(str(data[net_batch][3][1]))
    openFile.write(',')
    #Print weak scaling
    if (data[net_batch][1] != []):
      openFile.write(str(data[net_batch][1][2]))
    openFile.write(',')
    if (data[net_batch][2] != []):
      openFile.write(str(data[net_batch][2][2]))
    openFile.write(',')
    if (data[net_batch][3] != []):
      openFile.write(str(data[net_batch][3][2]))

    openFile.write("\n")

###############################################
def calcNetBatchGPU(filenames, net, batch, gpu):
    speed = 0
    for filename in filenames:
        sobj = re.search(r'output_'+net+r'_b'+str(batch)+r'_'+str(gpu)+r'gpu', filename)
        if(sobj):
            f=open(filename,'r')
            sFile = f.read()
            sobj = re.search(r'Images/sec: (\d+)', sFile)
            if(sobj):
                speed = int(sobj.group(1))
            else:
                print('Can\'t extract speed from ', filename)
    return speed

###############################################
def calcNetBatch(filenames, net, batch):
    speed = []
    for gpu in [1, 2, 4, 8]:
        speed.append(calcNetBatchGPU(filenames, net, batch, gpu))
    return speed

###############################################
def getAllBatches(filenames, net):
    batches = {}
    for filename in filenames:
        sobj = re.search(r'output_'+net+r'_b(\d*)_', filename)
        if(sobj):
            batch = int(sobj.group(1))
            batches[batch] = 1
    lBatches = list(batches.keys())
    lBatches.sort()
    return lBatches

###############################################
def calcNet(filenames, net):
    batches = []
    batches = getAllBatches(filenames, net)

    data = defaultdict(list)
    for b in batches:
        key = (net, b, 'GPU')
        speeds = calcNetBatch(filenames, net, b)
        for i in range(4):
            data[key].append([speeds[i], 0, 0])
    return data

###############################################
def getAllNets(filenames):
    nets = {}
    for filename in filenames:
        sobj = re.search(r'output_(.*)_b\d', filename)
        net = sobj.group(1)
        nets[net] = 1
    return list(nets.keys())

###############################################
def main():
    args = sys.argv[1:]
    if not args:
      print('usage: [file ...]')
      sys.exit(1)

    filenames = []
    for name in args:
      if name.endswith('.log'):
        filenames.append(name)

    nets = getAllNets(filenames)
    
    data = defaultdict(list)
    for net in nets:
        data.update(calcNet(filenames, net))

    generate_scaling_numbers(data)

    #Sort data by keys
    sorted_data = collections.OrderedDict(sorted(data.items()))

    print("Writing csv file...")
    for filename in filenames:
      print_csv("bench.csv", sorted_data);
    print("Done")

###############################################
main()



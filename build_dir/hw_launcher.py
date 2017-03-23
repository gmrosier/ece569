##############################################################################
#
# HW Launcher
#
# Aaron Blood
#
import argparse
import os
import sys
import json
import time

def printValueByMsg(file, msg, value):
  with open(file) as fp:
    for line in fp:
      d = json.loads(line)
      if 'message' in d['data']:
        if d['data']['message'] == msg:
          if 'elapsed_time' in d['data']:
            print value + ':', d['data'][value], 'of', msg
            return int(d['data'][value])

def printValueInMsg(file, msg):
  with open(file) as fp:
    for line in fp:
      d = json.loads(line)
      if 'message' in d['data']:
        if msg in d['data']['message']:
          print d['data']['message']
          return float(d['data']['message'].replace(msg,'').replace(' ',''))
            
def checkSolution(file):
  with open(file) as fp:
    for line in fp:
      d = json.loads(line)
      if 'correctq' in d['data']:
        print d['data']['message']
        if 'The solution is correct' in d['data']['message']:
          return True    
        else:
          return False
            
parser = argparse.ArgumentParser(description='HW Launcher')
parser.add_argument("setTimes",type=int, help="Set number and number of times. This can include any number of arguments but must be a multiple of 2 total."
                    + "For example, to launch set #1 10 times and set #2 6 times: hw_launcher 1 10 2 6", nargs="+")
args = parser.parse_args()
set=0
times=0
if  len(args.setTimes) % 2:
  print "Error: Odd number of command line arguments, must supply a number of times value for each set"
  sys.exit()
else:
  os.system('rm histogram_output_*');
  for i in range(0,len(args.setTimes)/2):
    set=args.setTimes[i*2]
    times=args.setTimes[i*2+1]
    command = 'cp Histogram/Dataset/' + str(set) + '/* ./'
    print 'running ' + command
    os.system(command)
    for j in range(1,times+1):
      command = 'sed \"s/histogram_output/'+ 'histogram_output_' + str(set) + '_' + str(j) + '/g\" < lsf_histogram.sh | bsub'
      print 'running ' + command
      os.system(command)
    #Wait for all job(s) to be finished
    print "Waiting for",str(times),"job(s) to finish..."
    bjobs = 'JOBID'
    while 'JOBID' in bjobs:
      bjobs = os.popen('bjobs').read()
      time.sleep(3)

  #Parse each mode each time
  for i in range(0,len(args.setTimes)/2):
    set=args.setTimes[i*2]
    times=args.setTimes[i*2+1]
    verOneSum = 0
    verTwoSum = 0
    for j in range(1,times+1):
      file = 'histogram_output_' + str(set) + '_' + str(j) + '.txt'
      verOneSum += printValueInMsg(file,'Elapsed kernel time (Version 1):')
      verTwoSum += printValueInMsg(file,'Elapsed kernel time (Version 2):')
    
    print "-----Set #" + str(set) + " results: ------"
    checkSolution(file)
    print "Version 1 Average:", str(verOneSum/(times * 1.0))
    print "Version 2 Average:", str(verTwoSum/(times * 1.0))
    print "---------------------------"
"""
Local Hardware Launcher
"""

import argparse
import os
import sys
import json
import time
import shutil
import subprocess

def printValueByMsg(output, msg, value):
    for line in output.split('\n'):
        d = json.loads(line)
        if 'message' in d['data']:
            if d['data']['message'] == msg:
                if 'elapsed_time' in d['data']:
                    print(value + ':', d['data'][value], 'of', msg)
                    return int(d['data'][value])

def printValueInMsg(output, msg):
    for line in output.split('\n'):
        d = json.loads(line)
        if 'message' in d['data']:
            if msg in d['data']['message']:
                print(d['data']['message'])
                return float(d['data']['message'].replace(msg,'').replace(' ',''))

def checkSolution(output):
    for line in output.split('\n'):
        d = json.loads(line)
        if 'correctq' in d['data']:
            print(d['data']['message'])
            if 'The solution is correct' in d['data']['message']:
                return True
            else:
                return False

parser = argparse.ArgumentParser(description='HW Launcher')
parser.add_argument("setTimes",type=int, help="Set number and number of times. This can include any number of arguments but must be a multiple of 2 total." + "For example, to launch set #1 10 times and set #2 6 times: hw_launcher 1 10 2 6", nargs="+")
args = parser.parse_args()
set = 0
times = 0
if  len(args.setTimes) % 2:
    print("Error: Odd number of command line arguments, must supply a number of times value for each set")
    sys.exit()
else:
    for i in range(0,len(args.setTimes) // 2):
        set = args.setTimes[i * 2]
        times = args.setTimes[i * 2 + 1]
        shutil.copy('Histogram/Dataset/' + str(set) + '/input.raw', './input.raw')
        shutil.copy('Histogram/Dataset/' + str(set) + '/output.raw', './output.raw')     

        verOneSum = 0
        verTwoSum = 0
        validSum = 0
        for j in range(times):
            result = subprocess.run(['Debug/Histogram.exe', '-e', 'output.raw', '-i', 'input.raw', '-t', 'integral_vector'], stdout=subprocess.PIPE)
            output = result.stdout.decode('utf-8')
            if (checkSolution(output)):
                validSum += 1
                verOneSum += printValueInMsg(output,'Elapsed kernel time (Version 1):')
                verTwoSum += printValueInMsg(output,'Elapsed kernel time (Version 2):')

        print("-----Set #" + str(set) + " results: ------")
        print("Valid Solutions:",   str(validSum))
        print("Version 1 Average:", str(verOneSum / times))
        print("Version 2 Average:", str(verTwoSum / times))
        print("---------------------------")

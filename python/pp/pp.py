#!/usr/bin/python
#The above-mentioned Thrift interface files...
import sys
import socket
import time

if socket.gethostname() == 'leny':
  sys.path.append('/home/marijnfs/dev/mecobo/Thrift interface/gen-py/NascenseAPI_v01e')
else:
  sys.path.append('/home/stollenga/dev/mecobo/Thrift interface/gen-py/NascenseAPI_v01e')

import emEvolvableMotherboard
import numpy as np
import datetime

import matplotlib.pyplot as plt

#thrift imports that should come into
from ttypes import *
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

import mixer

now = datetime.datetime.now()
timestr = "%s-%s+%s:%s" % (now.month, now.day, now.hour, now.minute)
log = open("pp-" + timestr + ".log", "w+")

#this is the transport used for thrift, it can be a socket or memory or ... other things
transport = TSocket.TSocket('129.241.103.191', 9090)
transport = TTransport.TBufferedTransport(transport)

#this is the proocol used in the transport, we just use binary
prot = TBinaryProtocol.TBinaryProtocol(transport)

#interface instance
cli = emEvolvableMotherboard.Client(prot);

#open the used transport method
transport.open();

#The rest are Thrift API calls
#reset everything
cli.reset()

#The host keeps a queue of items to be applied to the board, this clears it
cli.clearSequences()

#this makes a sequence item and sets up static voltages on the selected pin.
#note the list-form of the pins
#time is in milliseconds, amplitude is -5 to 5V scaled to between 0 and 255.

random_pins = np.arange(16)
#np.random.shuffle(random_pins)

random_pins = [5, 3, 6, 13, 10, 8, 11, 1, 14]

print >>log, "PINS: output, input..", random_pins

def evaluate(setups):
  cli.reset()
  cli.clearSequences()

  it = [emSequenceItem() for i in range(8)]

#  record_pin = 11

#  it[0].pin = [13]
#  it[1].pin = [9]
#  it[2].pin = [5]
#  it[3].pin = [8]
#  it[4].pin = [7]
#  it[5].pin = [14]
#  it[6].pin = [6]
#  it[7].pin = [12]

  record_pin = random_pins[0]

  it[0].pin = [random_pins[1]]
  it[1].pin = [random_pins[2]]
  it[2].pin = [random_pins[3]]
  it[3].pin = [random_pins[4]]
  it[4].pin = [random_pins[5]]
  it[5].pin = [random_pins[6]]
  it[6].pin = [random_pins[7]]
  it[7].pin = [random_pins[8]]

  out = emSequenceItem()
  out.pin = [record_pin]
  out.frequency = 10000
  out.operationType = emSequenceOperationType().RECORD   #implies analogue recording

  t = 0
  T = 10 #time should be milli seconds
  Ts = out.frequency / 1000 * T
  for setup in setups: #Each setup is one configuration to test
    for i, x in enumerate(setup): #Each config contains n values which are set as input
      it[i].startTime = t
      it[i].endTime = t + T
      # it[i].amplitude = np.clip(x, 0., 1.) * 255 #convert to the potentials
      it[i].amplitude = np.clip(x, 0., 1.) * 64 #convert to the potentials
      it[i].operationType = emSequenceOperationType().CONSTANT   #analogue output
      cli.appendSequenceAction(it[i])
    t += T

  #Now we know when to stop recording

  out.startTime = 0
  out.endTime = t

  cli.appendSequenceAction(out)

  cli.runSequences()
  cli.joinSequences()

  #the data that comes back are raw ADC samples scaled to +-5 V with a sign bit,
  #so we just rescale it to 'real' volts here.
  samples = np.array(cli.getRecording(record_pin).Samples)
  res = samples * (1.0/4096.0)
  print (len(samples))
  if len(samples) < len(setups): #failed, try again until heap fills...
    return evaluate(setups)
  #plt.plot(res)
  #plt.show()

  #  print(res[Ts-1::Ts])
  return res[Ts-1::Ts]

def display(ind, prob, target, save=None):
  plt.clf()
  samples = []
  RES = 25

  for y in np.linspace(1, 0., RES):
    for x in np.linspace(0, 1., RES):
      samples.append(np.append(ind.copy(), [x, y]))
  data = evaluate(samples)
  data = data.reshape(RES, RES)
  #data = np.random.normal(size=(RES, RES))
  plt.imshow(data, cmap="hot", interpolation='bilinear')
  plt.gca().set_autoscale_on(False)
  plt.plot([p[0] * (RES-1) for p, t in zip(prob, target) if t >= MEAN_VOLT], [p[1] * (RES-1) for p, t in zip(prob, target) if t >= MEAN_VOLT], "bx", [p[0] * (RES-1) for p, t in zip(prob, target) if t < MEAN_VOLT], [p[1] * (RES-1) for p, t in zip(prob, target) if t < MEAN_VOLT], "ro")
  #plt.plot([p[0] for p, t in zip(prob, target) if t >= 0.0], [p[1] for p, t in zip(prob, target) if t >= 0.0], "kx")
  #plt.axis([0, RES, 0, RES])
  if save:
    plt.savefig(save)
  else:
    plt.show()

# tests = []
# for i in range(20):
#   tests.append([np.random.randint(255), np.random.randint(255), np.random.randint(255), np.random.randint(255)])
# evaluate(tests)
# evaluate(tests)

TARGET_VOLT = .01 #Determines what voltage is the target
MEAN_VOLT = -0.485
problem = [np.array([.4, .4])]
target = [TARGET_VOLT + MEAN_VOLT]

POP, GEN = 20, 6

pop = np.random.uniform(size=(POP, GEN))
display(pop[0], problem, target, "test1.pdf")
display(pop[0], problem, target, "test2.pdf")
display(pop[0], problem, target, "test3.pdf")
exit(1)

mix = mixer.Mixer(noise = .03, shuffle_prob = .1)

solved = []
add_times = []
global_time_start = time.time()

epoch = 0
counter = 0

n_solved = 0

start_budget = 10
budget = start_budget

while True:
  print >>log, "===================="
  print >>log, "epoch:", epoch

  #Evaluating population
  sample_set = []
  for ind in pop:
    for p in problem:
      x = np.append(ind.copy(), p)
      sample_set.append(x)

  results = evaluate(sample_set)
  print "res ", results[:5]
  n = 0
  errs = np.zeros(POP)
  if len(results) != len(sample_set): #board failed, try again
    continue

  for i, ind in enumerate(pop):
    for t in target:
      #errs[i] += (results[n] - t) ** 2
      print(len(sample_set), len(results), n, i, len(errs))
      errs[i] += 1.0 if ((results[n] > MEAN_VOLT) ^ (t > MEAN_VOLT)) else 0.0
      errs[i] += (results[n] - t) ** 2
      n += 1

  # print(results, target)
  print >>log, "errs:", errs

  #Evolve
  mix.mix(pop, errs, cost=True)
  pop = np.clip(pop, 0., 1.) #volages need to be supplied in positive range?

  min_err = np.min(errs) / len(problem)
  best_ind = pop[np.argmin(errs)]

  if min_err < .1:
    n_solved += 1
    solved.append((global_time_start, n_solved, epoch, problem, target, add_times))
    print("solved", solved)
    print >>log, "solved", solved
    display(best_ind, problem, target, "best_%s.pdf" % epoch)
    counter = 0
    new_prob = np.random.uniform(size=2)
    x = np.append(best_ind.copy(), new_prob)
    guess = evaluate([x])[0]

    problem.append(new_prob)
    if guess > MEAN_VOLT:
      target.append(MEAN_VOLT - TARGET_VOLT)
    else:
      target.append(MEAN_VOLT + TARGET_VOLT)
    add_times.append(epoch)
    print >>log, "added problem", problem, target
    budget = start_budget

  elif counter > budget:
    counter = 0
    new_prob = np.random.uniform(size=2)
    x = np.append(best_ind.copy(), new_prob)
    guess = evaluate([x])[0]

    problem[-1] = new_prob
    if guess > MEAN_VOLT:
      target[-1] = MEAN_VOLT - TARGET_VOLT
    else:
      target[-1] = MEAN_VOLT + TARGET_VOLT
    print >>log, "changed problem", problem, target
    budget *= 1.1

  print("errs [min] [mean] [ind]:", np.min(errs), np.mean(errs), np.argmin(errs))
  print("best ind: ", pop[np.argmin(errs)])
  print >>log, "errs:", np.min(errs), np.mean(errs)
  print >>log, pop
  epoch += 1
  counter += 1

transport.close()

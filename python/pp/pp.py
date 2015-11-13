#!/usr/bin/python
#The above-mentioned Thrift interface files...
import sys
sys.path.append('/home/stollenga/dev/mecobo/Thrift interface/gen-py/NascenseAPI_v01e')
import emEvolvableMotherboard
import numpy as np

import matplotlib.pyplot as plt

#thrift imports that should come into
from ttypes import *
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

import mixer

log = open("log", "w+")

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

def evaluate(setups):
  cli.reset()
  cli.clearSequences()

  it = [emSequenceItem() for i in range(8)]

  it[0].pin = [0]
  it[1].pin = [7]
  it[2].pin = [3]
  it[3].pin = [5]
  it[4].pin = [6]
  it[5].pin = [4]
  it[6].pin = [1]
  it[7].pin = [2]

  out = emSequenceItem()
  out.pin = [0]
  out.frequency = 10000
  out.operationType = emSequenceOperationType().RECORD   #implies analogue recording

  t = 0
  T = 10 #time should be milli seconds
  Ts = out.frequency / 1000 * T
  for setup in setups: #Each setup is one configuration to test
    for i, x in enumerate(setup): #Each config contains n values which are set as input
      it[i].startTime = t
      it[i].endTime = t + T
      it[i].amplitude = np.clip(x, 0., 1.) * 255. #convert to the potentials
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
  samples = np.array(cli.getRecording(0).Samples)
  res = samples * (1.0/4096.0)

  #plt.plot(res)
  #plt.show()

  #  print(res[Ts-1::Ts])
  return res[Ts-1::Ts]

def display(ind, prob, target, save=None):
  plt.clf()
  samples = []
  RES = 5

  for y in np.linspace(0, 1., RES):
    for x in np.linspace(0, 1., RES):
      samples.append(np.append(ind.copy(), [x, y]))
  data = evaluate(samples)
  data = data.reshape(RES, RES)
  print("showing")
  plt.imshow(data, cmap="gray")
  plt.plot([p[0] for p, t in zip(prob, target) if t >= 0.0], [p[1] for p, t in zip(prob, target) if t >= 0.0], "bx", [p[0] for p, t in zip(prob, target) if t < 0.0], [p[1] for p, t in zip(prob, target) if t < 0.0], "ro")
  #plt.plot([p[0] for p, t in zip(prob, target) if t >= 0.0], [p[1] for p, t in zip(prob, target) if t >= 0.0], "kx")
  if save:
    plt.savefig(save)
  else:
    plt.show()


# tests = []
# for i in range(20):
#   tests.append([np.random.randint(255), np.random.randint(255), np.random.randint(255), np.random.randint(255)])
# evaluate(tests)
# evaluate(tests)

TARGET_VOLT = .1 #Determines what voltage is the target
problem = [np.array([.4, .4])]
target = [TARGET_VOLT]


POP, GEN = 10, 6

pop = np.random.uniform(size=(POP, GEN))
#display(pop[0], problem, target, "test.pdf")
#display(pop[1], problem, target)

mix = mixer.Mixer(noise = .03, shuffle_prob = .1)

epoch = 0
counter = 0
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

  n = 0
  errs = np.zeros(POP)

  for i, ind in enumerate(pop):
    for t in target:
      print(results[n], t)

      #errs[i] += (results[n] - t) ** 2
      errs[i] += 1.0 if ((results[n] > 0.0) ^ (t > 0.0)) else 0.0
      errs[i] += (results[n] - t) ** 2
      n += 1

  print(results, t)
  print >>log, "errs:", errs

  #Evolve
  mix.mix(pop, errs, reverse=True)
  pop = np.clip(pop, 0., 1.) #volages need to be supplied in positive range?

  min_err = np.min(errs) / len(problem)
  best_ind = pop[np.argmin(errs)]

  if min_err < .1 and counter > 10:
    display(best_ind, problem, target, "best_%s.pdf" % epoch)
    counter = 0
    new_prob = np.random.uniform(size=2)
    x = np.append(best_ind.copy(), new_prob)
    guess = evaluate([x])[0]

    problem.append(new_prob)
    if guess > 0.0:
      target.append(-TARGET_VOLT)
    else:
      target.append(TARGET_VOLT)
    print >>log, "added problem", problem, target

  elif counter > 30:
    counter = 0
    problem[-1] = np.random.uniform(size=2)
    new_prob = np.random.uniform(size=2)
    x = np.append(best_ind.copy(), new_prob)
    guess = evaluate([x])[0]

    problem[-1] = new_prob
    if guess > 0.0:
      target[-1] = -TARGET_VOLT
    else:
      target[-1] = TARGET_VOLT
    print >>log, "changed problem", problem, target

  print("errs [min] [mean] [ind]:", np.min(errs), np.mean(errs), np.argmin(errs))
  print("best ind: ", pop[np.argmin(errs)])
  print >>log, "errs:", np.min(errs), np.mean(errs)
  print >>log, pop
  epoch += 1
  counter += 1

transport.close()
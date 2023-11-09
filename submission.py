import argparse

from HMM import HMM, Observation, forward, viterbi
from alarm import alarm_infer
from carnet import car_infer

# add to the argument (partofspeech.browntags.trained --generate 20 --viterbi ambiguous_sents.obs --forward ambiguous_sents.obs)
# Vinay Bojja
if __name__ == '__main__':
    model = HMM()

    parser = argparse.ArgumentParser(description='Getting arguments from the terminal')
    parser.add_argument('--generate', type=int, help='generate method')
    parser.add_argument('filename', type=str, help='file')
    parser.add_argument('--forward', type=str, help = 'obs file')
    parser.add_argument('--viterbi', type=str, help='obs file')

    args = parser.parse_args()


    print(":::::::::::::Question 2::::::::::::")
    # +++++++++++++++ Load Data
    model.load(args.filename)
    print("Data Loaded.")
    # ++++++++++++++ Generate
    observations = model.generate(args.generate)
    print("Genrated Data: \n", observations)
    # +++++++++++++++++ forward matrix
    forward(args.forward, model)
    print("Please check file named forward.output.obs for the output.")
    # +++++++++++++++++ viterbi
    viterbi(args.viterbi, model)
    print("Please check file named viterbi.output.obs for the output.")

    print("::::::::::::::Question3:::::::::::::")

    print("Question 3.1:")
    # Question 1.a
    p = alarm_infer.query(variables=["MaryCalls"],evidence={"JohnCalls":"yes"})
    print(p)

    #Question 1.b
    t = alarm_infer.query(variables=["MaryCalls","JohnCalls"],evidence={"Alarm":"yes"})
    print(t)

    # Question 1.c
    y = alarm_infer.query(variables=["Alarm"],evidence={"MaryCalls":"yes"})
    print(y)

    print("Question 3.2:")

    # Question 2.a Given that the car will not move, what is the probability that the battery is not working?
    a = car_infer.query(variables=["Battery"], evidence={"Moves": "no"})
    print(a)
    # Answer: 0.3590

    # Question 2.b Given that the radio is not working, what is the probability that the car will not start?
    b = car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"})
    print(b)
    # Answer: 0.8687

    # Question 2.c Given that the battery is working, does the probability of the radio working change if we discover that the car has gas in it?
    c1 = car_infer.query(variables=["Radio"], evidence={"Battery": "Works"})
    print(c1)
    c = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"})
    print(c)
    # Answer: No the probability does not change

    # Question 2.d Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car dies not have gas in it?
    d1 = car_infer.query(variables=["Ignition"], evidence={"Moves": "no"})
    print(d1)
    d = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"})
    print(d)
    # Answer: The probability of ignition failing reduces from 0.5666 to 0.4822

    # Question 2.3 What is the probability that the car starts if the radio works and it has gas in it?
    e = car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"})
    print(e)
    # Answer: 0.7212

    e = car_infer.query(variables=["Starts"], evidence={"Gas": "Full", "Ignition": "Works", "KeyPresent": "yes"})
    print(e)
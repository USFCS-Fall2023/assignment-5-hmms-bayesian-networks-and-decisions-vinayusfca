import argparse

from HMM import HMM, Observation, forward, viterbi

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

import pickle
import numpy
from midi_to_statematrix import *

import multi_training
import model

import os


def gen_adaptive(m, pcs, times, keep_thoughts=False, name="final"):
    xIpt, xOpt = map(lambda x: numpy.array(x, dtype='int8'),
                     multi_training.getPieceSegment(pcs))
    all_outputs = [xOpt[0]]
    if keep_thoughts:
        all_thoughts = []
    m.start_slow_walk(xIpt[0])
    cons = 1
    for time in range(multi_training.batch_len * times):
        resdata = m.slow_walk_fun(cons)
        nnotes = numpy.sum(resdata[-1][:, 0])
        if nnotes < 2:
            if cons > 1:
                cons = 1
            cons -= 0.02
        else:
            cons += (1 - cons) * 0.3
        all_outputs.append(resdata[-1])
        if keep_thoughts:
            all_thoughts.append(resdata)
    noteStateMatrixToMidi(numpy.array(all_outputs), 'output/' + name)
    if keep_thoughts:
        pickle.dump(all_thoughts, open('output/' + name + '.p', 'wb'))


def fetch_train_thoughts(m, pcs, batches, name="trainthoughts"):
    all_thoughts = []
    for i in range(batches):
        ipt, opt = multi_training.getPieceBatch(pcs)
        thoughts = m.update_thought_fun(ipt, opt)
        all_thoughts.append((ipt, opt, thoughts))
    pickle.dump(all_thoughts, open('output/' + name + '.p', 'wb'))


def get_last_epoch(model_directory):
    # Get all file names in model_directory
    files = [file for file in os.listdir(model_directory) if '.p' in file]
    # Function that go over a string and create in a list all individual
    #   numbers and then return all of them joined as a string.
    get_number = (lambda string: "".join(list(filter(lambda c: c.isdigit(),
                                                     string))))
    # Map the get_number over all the names and return them as integers
    epochs = list(map(lambda string: int(get_number(string)), files))
    epochs.append(0)  # Append 0 in case of void list
    return max(epochs)


if __name__ == '__main__':
    # Directory in which the parameters that have been calculated for the model
    #  are saved.
    music_type_dir = "Scale2"
    save_output_dir = music_type_dir + "/output"
    os.makedirs(save_output_dir, exist_ok=True)

    # Create and evaluate model
    pcs = multi_training.loadPieces("Scale2")
    start = get_last_epoch(save_output_dir)

    m = model.BiaxialRNNModel([300, 300], [100, 50])

    multi_training.trainPiece(m, pcs, 10000, "Scale", start)

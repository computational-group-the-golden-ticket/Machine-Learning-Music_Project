import itertools
from midi_to_statematrix import upperBound, lowerBound


def startSentinel():
    def noteSentinel(note):
        position = note
        part_position = [position]

        pitchclass = (note + lowerBound) % 12
        part_pitchclass = [int(i == pitchclass) for i in range(12)]

        return part_position + part_pitchclass + [0] * 66 + [1]
    return [noteSentinel(note) for note in range(upperBound - lowerBound)]


def getOrDefault(l, i, d):
    try:
        return l[i]
    except IndexError:
        return d


def buildContext(state):
    context = [0] * 12
    for note, notestate in enumerate(state):
        if notestate[0] == 1:
            pitchclass = (note + lowerBound) % 12
            context[pitchclass] += 1
    return context


def buildBeat(time):
    """
    The module class 64 of time (numbers time%64) are represented in binary
      representation, changint the 0's by ones, this in order to be more
      suitable to affec the result in the weight matrixes to be determined.
    This in fact is a binary representation of the time withing the time for
      the note played or hold at a given time.
    """
    return [2 * x - 1 for x in [time % 2, (time // 2) % 2, (time // 4) % 2,
                                (time // 8) % 2]]


def noteInputForm(note, state, context, beat):
    position = note
    part_position = [position]

    pitchclass = (note + lowerBound) % 12
    part_pitchclass = [int(i == pitchclass) for i in range(12)]
    # Concatenate the note states for the previous vicinity
    part_prev_vicinity = list(itertools.chain.from_iterable(
        (getOrDefault(state, note + i, [0, 0]) for i in range(-12, 13))))

    part_context = context[pitchclass:] + context[:pitchclass]

    return (part_position + part_pitchclass + part_prev_vicinity +
            part_context + beat + [0])


def noteStateSingleToInputForm(state, time):
    # Construct binary representation of time in the bar.
    beat = buildBeat(time)
    # This will crate a list of the 25 nearby notes to the actual one in time, 
    #   this ones say if the note was played or holded.
    context = buildContext(state)
    return [noteInputForm(note, state, context, beat)
            for note in range(len(state))]


def noteStateMatrixToInputForm(statematrix):
    # Each list like the ones in statematrix[:,:,:,0] is transform to a more
    #  meaningful set of features that gives information about the note
    #  and its surroundings.
    inputform = [noteStateSingleToInputForm(state, time)
                 for time, state in enumerate(statematrix)]
    return inputform


###########################################################################
# Input output functions
###########################################################################

def append_data_to(file_to_save, data):
    with open(file_to_save, "a") as file:
        file.write("%d %d\n" % (data[0], data[1]))

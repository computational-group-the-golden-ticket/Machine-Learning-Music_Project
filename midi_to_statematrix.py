import mido
import torch

# In this range a map is made from number to note. Therefore, there are 78
#  notes; this values are defined as global variables.
lowerBound = 24
upperBound = 102


def midiToNoteStateMatrix(midifile):
    """
    input:
      midifile(str): standart midi file; this format is desired in order to
        have a music metric into account for the training, also it contains
        what are called values in music.
    ouput:
      statematrix(list): list of dimensions (n_1/16, n_notes, bool[2]).
        n_1/16: number of times covered sampling in 1/16 of a round, e. g.:
          if a midi file has only a black it n_1/16 = 4.
        n_notes: The number of notes to be taken into account in a given scale,
          this one is given by upperBound - lowerBound. In this when Sol key
          40(in idx) is designated for MI. Bug: the program does not count
          notes out of the  range and could put subsequent notes that are not;
          please verify range approipately.
        bool[]: its first value in statematrix[i_1/16, i_note] says if the note
          was played in that time, the second one if it was articilated at that
          time. Then an eight La would have values statematrix[i, La_number] =
          [1, 1] and then statematrix[i + 1, La_number] = [0, 1]
    """
    # Init principal midi object
    pattern = mido.MidiFile(midifile)

    # It can sample with a presicion of 1/resolution each beat, by default each
    #   beat is equivalent to a quarter of a note.
    resolution = pattern.ticks_per_beat

    # Make a list of 0's for each track in pattern, each value save time to
    #   to arrive to next note
    time_to_next = [track[0].time for track in pattern.tracks]

    # Make a list of 0's for each track in pattern, each value saves the
    #   position of the corrent message
    posns = [0 for track in pattern.tracks]

    statematrix = []  # Init output variable
    span = upperBound - lowerBound  # Number or notes that will be represented
    time = 0

    # Init the variable state, that is not more that the representation that
    #  has the notes to say if they were actually played or articulated; this
    #  will be updated correctelly
    state = [[0, 0] for x in range(span)]
    statematrix.append(state)

    # This the minimum that can have a note in order to be read
    note_resolution_in_ticks = resolution / 4

    while True:
        # This if in order to aggregate each state when time has values given
        #  by: note_resolution_in_ticks*i + note_resolution_in_ticks/2. For
        #  example if note_resolution_in_ticks = resolution / 4 it has notes in
        #  in the follwing values: (1/8, 3/8, 5/8, 7/8) * resolution.
        if time % note_resolution_in_ticks == (note_resolution_in_ticks / 2):
            # Crossed a note boundary. Create a new state and save it, default
            #   to hold notes if the note before has been articulated; see
            #   state[evt.note - lowerBound].
            oldstate = state
            state = [[oldstate[x][0], 0] for x in range(span)]
            statematrix.append(state)

        # Make a for over number of tracks
        for i in range(len(time_to_next)):
            while time_to_next[i] == 0:
                track = pattern.tracks[i]  # Choose the track
                pos = posns[i]  # Message number in track
                evt = track[pos]  # Message in pos

                # Verify if a note was played
                if evt.type == 'note_on':
                    # If not is not in the range of interest then pass
                    if evt.note < lowerBound or evt.note >= upperBound:
                        print("Note {} at time {} out of bounds \
                            (ignoring)".format(evt.note, time))
                    else:
                        if evt.type == 'note_off' or evt.velocity == 0:
                            # Substract lower bound for indexing in the list
                            state[evt.note - lowerBound] = [0, 0]
                        else:
                            state[evt.note - lowerBound] = [1, 1]
                elif evt.type == 'time_signature':
                    # We don't want to worry about non-4 time signatures.
                    if evt.numerator not in (2, 4):
                        print("Found incorrect time signature event \
                              {}!".format(evt))
                        return statematrix

                try:
                    # Time to arrive to the following note, in the case of
                    #   being different from zero a note begins to be played.
                    time_to_next[i] = track[pos + 1].time
                    # Go to the next position in track i
                    posns[i] += 1
                except IndexError:
                    # State time_to_next[i] to None to indicate that the track
                    #   i has been read.
                    time_to_next[i] = None

            # Stop when all the tracks have been read
            if time_to_next[i] is not None:
                time_to_next[i] -= 1

        if all(t is None for t in time_to_next):
            break

        # The same note is hold until time_to_next[i] turns to 0 again, note
        #  that the state map is correctly made thanks to the fact that the
        #  time the note is hold is not the theoretical one.
        time += 1

    return statematrix


def noteStateMatrixTomidi(statematrix, name="example"):
    """
    TODO: This is the inverse process that midiToNoteStateMatrix makes;
      comments to do.
    """
    pattern = mido.MidiFile()
    track = mido.MidiTrack()
    pattern.tracks.append(track)

    span = upperBound - lowerBound
    tickscale = 55

    lastcmdtime = 0
    prevstate = [[0, 0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(mido.Message('note_off', note=note + lowerBound,
                                      time=(time - lastcmdtime) * tickscale))
            lastcmdtime = time
        for note in onNotes:
            track.append(mido.Message('note_on', note=note + lowerBound,
                                      time=(time - lastcmdtime) * tickscale,
                                      velocity=40))
            lastcmdtime = time

        prevstate = state

    track.append(mido.MetaMessage('end_of_track'))
    pattern.save("{}.mid".format(name))


def main():
    song = midiToNoteStateMatrix('data/beethoven_hammerklavier_1.mid')

    noteStateMatrixTomidi(torch.Tensor(song), '123prueba')


if __name__ == '__main__':
    main()

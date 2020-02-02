import mido
import torch


lowerBound = 55
upperBound = 65


def midiToNoteStateMatrix(midifile):
    pattern = mido.MidiFile(midifile)
    resolution = pattern.ticks_per_beat

    timeleft = [track[0].time for track in pattern.tracks]

    posns = [0 for track in pattern.tracks]

    statematrix = []
    span = upperBound - lowerBound
    time = 0

    state = [[0, 0] for x in range(span)]
    statematrix.append(state)
    while True:
        if time % (resolution / 4) == (resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to
            # holding notes
            oldstate = state
            state = [[oldstate[x][0], 0] for x in range(span)]
            statematrix.append(state)

        for i in range(len(timeleft)):
            while timeleft[i] == 0:
                track = pattern.tracks[i]
                pos = posns[i]

                evt = track[pos]
                if evt.type == 'note_on':
                    if (evt.note < lowerBound) or (evt.note >= upperBound):
                        pass
                        # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                    else:
                        if evt.type == 'note_off' or evt.velocity == 0:
                            state[evt.note - lowerBound] = [0, 0]
                        else:
                            state[evt.note - lowerBound] = [1, 1]
                elif evt.type == 'time_signature':
                    if evt.numerator not in (2, 4):
                        # We don't want to worry about non-4 time signatures. Bail early!
                        # print "Found time signature event {}. Bailing!".format(evt)
                        return statematrix

                try:

                    timeleft[i] = track[pos + 1].time
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1

    return statematrix


def noteStateMatrixTomidi(statematrix, name="example"):
    """FIX"""
    statematrix = statematrix.numpy()
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


def test():
    song = midiToNoteStateMatrix('../MusicGenerator/1negra.mid')

    for notes in song:
        print(notes)


def main():
    song = midiToNoteStateMatrix('music/beethoven_opus10_2.mid')

    noteStateMatrixTomidi(torch.Tensor(song), '123prueba')


if __name__ == '__main__':
    # main()
    test()

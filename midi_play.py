import mido as mido
from mido import MidiFile
import time as time

mid = MidiFile('Twinkle_Twinkle_Little_Star.mid')
name = mido.get_output_names() 


port = mido.open_output(name[0])

for msg in mid:
    time.sleep(msg.time)
    if not msg.is_meta:
        port.send(msg)






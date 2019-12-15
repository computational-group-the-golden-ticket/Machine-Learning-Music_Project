import mido
from mido import MidiFile
import pandas as pd
import os 

d_of_songs = {}
song_counter = 0 

def add_song_to_dict(filename):
	""" This is to add a song to the dictionary that contain 
		all the song data """

	global song_counter

	mid = MidiFile(filename)

	times = []
	notes = []
	velocities = []
	channels = []

	for msg in mid.tracks[0]:
		if ( msg.type == 'note_on'):
			times.append( msg.time )
			notes.append( msg.note ) 
			velocities.append( msg.velocity )
			channels.append( msg.channel )
			
			

	d_of_songs["s%dtime" %song_counter ] = times
	d_of_songs["s%dnote" %song_counter ] = notes
	d_of_songs["s%dvelocity" %song_counter ] = velocities
	d_of_songs["s%dchannels" %song_counter ] = channels

	#print(d_of_songs)

	song_counter +=1

def add_songs_in_directory_to_filename(directory_path = "songs", filename = "songs_data.hdf" ,
	current_counter = 0):
	"""" The actual counter says how many sounds have been saved in file name, the function 
	would pass all the files in the directory to the songs_data.hdf """
	
	global song_counter 
	song_counter = current_counter

	list_names = os.listdir(directory_path)

	for name in list_names:
		if '.mid' in name:
			add_song_to_dict(os.path.join(directory_path, name))
			break

	data = pd.DataFrame(data = d_of_songs, dtype=int)

	data.to_hdf(filename,key="GT")




## This codes so far works if the single track is represented with pulcrity with note_on's

add_songs_in_directory_to_filename("twinkle", "songs_data.hdf")
data = pd.read_hdf("songs_data.hdf", key="GT") 
print(data)







#print( dir(mid) )
#print (mid.debug)

#print( type( mid.tracks[0] ) )
#print( dir( mid.tracks[0] ) )
#print( help(mid.tracks[0] ) )
#print( mid.tracks[0].index('time') )


"""
print( dir(mid) )
print( mid.ticks_per_beat)
#print( mid.tempo2bpm() )
print( mid.type )
print( mid.charset )
print( mid.print_tracks() )
"""



#for msg in mid.play():
#    port.send(msg)


"""
c = 0 
for i, track in enumerate(mid.tracks):
	print('Track {}: {}'.format(i, track.name))
	print(track[i])
	#for msg in track[i]:
	#	print(msg)
"""






"""
msg.type
'note_on'
msg.note
60
msg.bytes()
[144, 60, 64]
msg.copy(channel=2)
Message('note_on', channel=2, note=60, velocity=64, time=0)
"""
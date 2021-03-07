# insert: <song name>.mp3 (must be saved in same folder)
Song_Name = "Garcon.mp3"

from mutagen.mp3 import MP3
audio_info = MP3(Song_Name).info

# Extract Sample Rate

Sample_Rate = audio_info.sample_rate
Time = audio_info.length

print("sampling frequency =", Sample_Rate, "Hz")

Total_Samples = int(Sample_Rate*Time)

dt = Time / Total_Samples

print("Time = ", Time)
print("dt = ", dt)
print("Samples = ", Total_Samples)
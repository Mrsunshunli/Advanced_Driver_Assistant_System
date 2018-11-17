import sys
import os
import speech_recognition as sr
import vlc
import sys

i=0
ctr=0
r=sr.Recognizer()
r.adjust_for_ambient_noise
r.energy_threshold = 10000

with sr.Microphone() as source:
    print('Say Something')
    audio=r.listen(source)

    try:
        detected=r.recognize_google(audio)
        #print('Yay\n'+detected)
    except:
        pass

playlist=[]

for x in os.listdir('C:\\Hackathon\\playlist'):
    playlist.append (x)

#print(playlist)

a="next"
b="pause"
z=detected
print(z)
#print(b)
#print(a)
if z.lower().find(a.lower()) != -1:
    complete_add="C:\\Hackathon\\playlist\\"+playlist[i]
    if(ctr>0):
        player.pause()
        player = vlc.MediaPlayer(complete_add)
        player.play()
    else:
        player = vlc.MediaPlayer(complete_add)
        player.play()
        ctr=1
        #ctr+=1
    i+=1
elif (z.lower().find("stop".lower()) != -1) or (z.lower().find("ap".lower()) != -1):
    player.stop()

elif(z.lower().find("exit".lower())!=-1):
    player.stop()
    sys.exit()

python = sys.executable
os.execl(python, python, * sys.argv)

import os
import re
import speech_recognition as sr
import vlc
import sys
import time
#from vlc import player

def restart_program():
    """Restarts the current program.
    Note: this function does not return. Any cleanup action (like
    saving data) must be done before calling this function."""
    python = sys.executable
    os.execl(python, python, * sys.argv)

i=0
ctr=0
r=sr.Recognizer()
r.adjust_for_ambient_noise
r.energy_threshold = 5000

while (True):
    detected=""
    with sr.Microphone() as source:
        print('Say Something')
        audio=r.listen(source)

    try:
        detected=r.recognize_google(audio)
        #print('Yay\n'+detected)
    #detected=r.recognize_google(audio)
    except:
        pass

    playlist=[]

    for x in os.listdir('C:\\Hackathon\\playlist'):
        playlist.append (x)
    
    #print(playlist)

    print(detected)

    #print(z)
    if(detected.lower().find("play".lower())!=-1):
        complete_add="C:\\Hackathon\\playlist\\"+playlist[0]
        player = vlc.MediaPlayer(complete_add)
        player.play()
        time.sleep(5)
        ctr=1
        
    #print(b)
    #print(a)
    elif detected.lower().find("next".lower()) != -1:
        #print("The value of counter"+str(ctr))
        #print("Next has been detected")
        complete_add="C:\\Hackathon\\playlist\\"+playlist[i]
        if(ctr==0):
            #print("The value of counter after coming in loop first time"+str(ctr))
            player = vlc.MediaPlayer(complete_add)
            player.play()
            ctr=1
            time.sleep(5)

        elif(ctr==1):
            #print("The value of counter after coming in loop next time"+str(ctr))
            player.stop()
            player = vlc.MediaPlayer(complete_add)
            player.play()
            time.sleep(5)
        #ctr+=1
        i+=1
        print("The value of i is "+str(i))
    elif (detected.lower().find("stop".lower()) != -1) or (detected.lower().find("ap".lower()) != -1):
        #print("Yes pause")
#        complete_add="C:\\Hackathon\\playlist\\"+playlist[i]
#       player = vlc.MediaPlayer(complete_add)
        player.stop()
        time.sleep(5)

    elif(detected.lower().find("exit".lower())!=-1):
        player.stop()
        break

sys.exit()
    

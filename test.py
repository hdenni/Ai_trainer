import os

os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:A aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input="result.avi", output="result.mp4"))

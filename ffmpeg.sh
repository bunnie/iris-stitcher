ffmpeg -framerate 10 -i %04d-ffmpeg.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p anim.mp4

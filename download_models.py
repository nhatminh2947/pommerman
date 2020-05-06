import subprocess

print("Downloading...")
result = subprocess.run(["scp", "-r", "nhatminh2947@NV03:/mnt/nfs/work/nhatminh2947/working/pommerman/models",
                         "/home/lucius/working/projects/pommerman/nv03"],
                        stdout=subprocess.PIPE)
print('Downloaded models in nv03')

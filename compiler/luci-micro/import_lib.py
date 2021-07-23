import subprocess

print("Clone flatbuffers")
subprocess.run(["git", "clone", "https://github.com/google/flatbuffers.git"])

print("Clone eigen")
subprocess.run(["git", "clone", "https://gitlab.com/libeigen/eigen.git"])

print("Clone gemmlowp")
subprocess.run(["git", "clone", "https://github.com/google/gemmlowp.git"])

print("Clone tensorflow")
subprocess.run(["git", "clone", "https://github.com/tensorflow/tensorflow.git"])

print("Clone ruy")
subprocess.run(["git", "clone", "https://github.com/google/ruy.git"])

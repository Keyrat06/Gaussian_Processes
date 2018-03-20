import sys

HEAD_LINE = "<<<<<<< HEAD\n"
MID_LINE = "=======\n"
TAIL_LINE = ">>>>>>> 494a4f294c9a360e3b5d22ebb7037c0f095ed268\n"

def clean(file_path):
	output = []
	with open(file_path) as file_to_clean:
		TAKING = True
		for line in file_to_clean.readlines():
			if line == HEAD_LINE or line == TAIL_LINE:
				TAKING = True
			elif line == MID_LINE:
				TAKING = False
			elif TAKING:
				output.append(line)
			else:
				pass

	with open(file_path+"new", "w") as  new_file:
		for line in output:
			new_file.write(line)

if __name__ == "__main__":
	clean(sys.argv[1])
all: compile

compile:
	mkdir -p bin
	nvcc ./src/main.cu -o ./bin/main -lm -lineinfo

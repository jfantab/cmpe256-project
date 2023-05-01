all: msd
	time ./msd spec.json

msd: msd.cpp heap.h json.hpp
	g++ -std=c++17 -O3 msd.cpp -o msd

msd-debug: msd.cpp heap.h json.hpp
	g++ -std=c++17 -g msd.cpp -o msd-debug

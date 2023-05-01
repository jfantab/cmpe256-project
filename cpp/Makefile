all: msd
	time ./msd spec_item.json
	time ./msd spec_user.json
	time ./msd spec_ensemble.json

item: msd
	time ./msd spec_item.json

user: msd
	time ./msd spec_user.json

lmf: msd
	time ./msd spec_lmf.json

als: msd
	time ./msd spec_als.json

ensemble: msd
	time ./msd spec_ensemble.json

implicit: msd
	time python msd_implicit.py

embedding: msd
	time ./msd spec_embedding.json

msd: msd.cpp heap.h json.hpp
	g++ -std=c++17 -O3 msd.cpp -o msd

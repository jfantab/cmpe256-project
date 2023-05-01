#ifndef __HEAH_H__
#define __HEAH_H__

#include <functional>

using namespace std;

typedef pair<double, int> HeapElement;

class Heap {
	HeapElement *_heap;
	int _max_count;
	int _count;
public:
	Heap() : _heap(nullptr), _max_count(0), _count(0) {}
	~Heap() {
		assert(_heap == nullptr && _count == 0);
	}
	void acquire(HeapElement *heap, int max_count) {
		assert(_heap == nullptr && _count == 0);  // only once;
		_heap = heap;
		_max_count = max_count;
	}
	void release() {
		_heap = nullptr;
		_max_count = 0;
		_count = 0;
	}
	static int parent(int i) { return (i - 1) >> 1; }
	static int left_child(int i) { return (i << 1) + 1; }
	static int right_child(int i) { return (i << 1) + 2; }
	void insert(HeapElement e) {
		assert(_count < _max_count);
		_heap[_count++] = e;
		int i = _count - 1;
		while (i > 0 && _heap[i].first < _heap[parent(i)].first) {
			swap(_heap[i], _heap[parent(i)]);
			i = parent(i);
		}
	}
	bool delete_min(HeapElement &e) {
		if (_count == 0) return false;
		e = _heap[0];
		_heap[0] = _heap[--_count];
		int i = 0;
		while (true) {
			int left = left_child(i);
			int right = right_child(i);
			int smallest = i;
			if (left < _count && _heap[left] < _heap[smallest]) {
				smallest = left;
			}
			if (right < _count && _heap[right] < _heap[smallest]) {
				smallest = right;
			}
			if (smallest != i) {
				swap(_heap[i], _heap[smallest]);
				i = smallest;
			} else {
				break;
			}
		}
		return true;
	}
};

#endif

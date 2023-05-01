//////////////////////////////////////////////////////////////////////////
//
// Recommender System for Million Songs Dataset Challenge
//
// -- Copyright 2023 Hardy K. S. Leung --
//
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <assert.h>
#include <functional>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <random>
#include <ctime>
#include <cstring>
#include <chrono>
#include "json.hpp"
#include "heap.h"

//////////////////////////////////////////////////////////////////////////

class Dataset;
class Recommendations;

//////////////////////////////////////////////////////////////////////////

using namespace std;

enum Method {
	user_based, item_based
};

enum Algo {
	Popularity,
	Randomized,
	EmbeddingBased,
	UserBased,
	ItemBased,
	LmfLatents,
	AlsLatents,
	Ensemble,
};

unordered_map<string, Algo> algo_hash = {
	{ "popularity", Algo::Popularity },
	{ "randomized", Algo::Randomized },
	{ "embedding", Algo::EmbeddingBased },
	{ "user", Algo::UserBased },
	{ "item", Algo::ItemBased },
	{ "lmf", Algo::LmfLatents },
	{ "als", Algo::AlsLatents },
	{ "ensemble", Algo::Ensemble }
};

//////////////////////////////////////////////////////////////////////////

inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }

typedef tuple<int, int, float> UIR;
typedef pair<int, int> PII;
typedef pair<int, int> Range;

mutex cout_mutex;

//////////////////////////////////////////////////////////////////////////

struct Score {
public:
	float score;
	int index;
	Score() : score(0), index(0) {}
	Score(float s, int i) : score(s), index(i) {}
	Score(const Score &s) {
		score = s.score;
		index = s.index;
	}
	Score &operator=(const Score &s) {
		score = s.score;
		index = s.index;
		return *this;
	}
	bool operator<(const Score &s) const { return score > s.score; }
};

//////////////////////////////////////////////////////////////////////////

class Global {
public:
	enum { recK = 500 };
	static bool recommend_test_users;
	static bool zero_one;
	static float alpha;
	static bool Jaccard;
	static float q;
	static bool output_recommendations;
	static vector<pair<string, float> > ensemble_inputs;
	static float ensemble_input_multiplier;
};

bool Global::recommend_test_users = false;
bool Global::zero_one = true;  // default;
float Global::alpha = 0.5;  // default;
float Global::q = 3.0;  // default;
bool Global::Jaccard = false;
bool Global::output_recommendations = false;  // default;
vector<pair<string, float> > Global::ensemble_inputs;  // default;
float Global::ensemble_input_multiplier = 1.0;  // default;

//////////////////////////////////////////////////////////////////////////

class DebugSimilarity {
	enum { K = 20 };
	const float _fraction = 0.1;
	vector<float> _cumu;
	int _count;
	mutex _mutex;
public:
	DebugSimilarity() : _cumu(K), _count(0) {}
	void debug(vector<Score> &scores);
};

DebugSimilarity debug_similarity;  // singleton;

void DebugSimilarity::debug(vector<Score> &scores) {
	const int NK = int(_fraction * scores.size() / K);
	float temp[K];  // temporary;
	for (int i = 0; i < K; i++) {
		int p = NK * i;
		nth_element(scores.begin(), scores.begin() + p, scores.end());
		temp[i] = scores[p].score;
	}

	_mutex.lock();
	for (int i = 0; i < K; i++) _cumu[i] = temp[i];
	_count++;
	if (_count % 100 == 0) {
		stringstream ss;
		for (int i = 0; i < K; i++) {
			ss << (_cumu[i] / _count) << ",";
		}
		cout_mutex.lock();
		cout << "debug " << _count << ": " << ss.str() << endl;
		cout << "------------------------------------" << endl;
		cout_mutex.unlock();
	}
	_mutex.unlock();
}

//////////////////////////////////////////////////////////////////////////
//
// class Performance -- progress tracking;
//
//////////////////////////////////////////////////////////////////////////

class Performance {
	clock_t _start;
	float _work;
public:
	Performance() : _start(clock()), _work(0) {}
	void incr_work(float work) { _work += work; }
	float workrate() const {
		float elapsed = (clock() - _start) / CLOCKS_PER_SEC;
		return _work / elapsed;
	}
};

//////////////////////////////////////////////////////////////////////////
//
// class Similarities -- |K|\times|U| or |K|\times|I| similarities
//
//////////////////////////////////////////////////////////////////////////

class Similarities {
	int _sparse_count;
	int _dense_count;
	int _allocation;
	int _topK;
	vector<Score> _similarities;
	vector<int> _dense_indices;
public:
	Similarities() :
		_sparse_count(0), _dense_count(0), _allocation(0), _topK(0) {}
	Similarities(const vector<int> &indices, int topK) {
		reset(indices, topK);
	}
	void reset(const vector<int> &indices, int topK) {
		int max_index = -1;
		for (auto index: indices) {
			if (max_index < index) max_index = index;
		}
		_topK = topK;
		_sparse_count = max_index + 1;
		_dense_count = indices.size();
		_allocation = _dense_count * _topK;
		_similarities.resize(_allocation);
		_dense_indices.resize(_sparse_count);
		fill(_similarities.begin(), _similarities.end(), Score(0, INT_MAX));
		fill(_dense_indices.begin(), _dense_indices.end(), _allocation);

		int p = 0;
		for (auto index: indices) {
			assert(index >= 0 && index < _sparse_count);
			assert(_dense_indices[index] == _allocation);  // must be "nullptr";
			_dense_indices[index] = p;
			p += _topK;
		}
		assert(p == _allocation);
	}
	void set(int index, int k, Score sim) {
		assert(index < _sparse_count && k < _topK);
		const int dense_index = _dense_indices[index];
		assert(dense_index < _allocation);
		_similarities[dense_index + k] = sim;
	}
	Score get(int index, int k) const {
		assert(index < _sparse_count && k < _topK);
		const int dense_index = _dense_indices[index];
		assert(dense_index < _allocation);
		return _similarities[dense_index + k];
	}
};

//////////////////////////////////////////////////////////////////////////
//
// class Jobs -- for multi-threaded application with mutex;
//
//////////////////////////////////////////////////////////////////////////

class Jobs {
	int _i;
	vector<pair<int, int> > _ranges;
	mutex _mutex;
	Performance _perf;

public:
	Jobs() : _i(0) {}
	Jobs(vector<pair<int, int> > &ranges) : _i(0), _ranges(ranges) {}
	~Jobs() {}
	void reset(vector<pair<int, int> > &ranges) {
		_i = 0;
		_ranges = ranges;
	}
	bool next(int &from, int &to) {
		bool has_job = true;
		_mutex.lock();
		if (_i == _ranges.size()) {
			from = 0;
			to = 0;
			has_job = false;
		} else {
			from = _ranges[_i].first;
			to = _ranges[_i].second;
			_i++;
			has_job = true;
			_perf.incr_work(to - from);
		}
		_mutex.unlock();
		return has_job;
	}
	float workrate() const { return _perf.workrate(); }
};

//////////////////////////////////////////////////////////////////////////

bool compare_uir_uv(const UIR uir1, const UIR uir2) {
	int delta;
	if ((delta = get<0>(uir1) - get<0>(uir2)) != 0) return delta < 0;
	if ((delta = get<1>(uir1) - get<1>(uir2)) != 0) return delta < 0;
	if ((delta = get<2>(uir1) - get<2>(uir2)) != 0) return delta < 0;
	return false;
}

bool compare_uir_iv(const UIR uir1, const UIR uir2) {
	int delta;
	if ((delta = get<1>(uir1) - get<1>(uir2)) != 0) return delta < 0;
	if ((delta = get<0>(uir1) - get<0>(uir2)) != 0) return delta < 0;
	if ((delta = get<2>(uir1) - get<2>(uir2)) != 0) return delta < 0;
	return false;
}

//////////////////////////////////////////////////////////////////////////
//
// class Dataset -- the main dataset class;
//
//////////////////////////////////////////////////////////////////////////

class Dataset {

public:
	enum UserType {
		Train, Valid, Test
	};

public:
	unordered_map<string, int> _user_hash;
	unordered_map<string, int> _item_hash;
	vector<pair<string, UserType> > _inv_users;
	vector<string> _inv_items;
	vector<int> _item_clusters;

	vector<UIR> _uirs_hidden_valid;
	vector<UIR> _uirs_hidden_test;
	vector<UIR> _uirs_uv;
	vector<UIR> _uirs_iv;
	vector<Range> _ranges_uv;
	vector<Range> _ranges_iv;
	int _user_count;
	int _valid_user_count;  // User::Valid only;
	int _test_user_count;  // User::Test only;
	int _item_count;
	int _N;

private:
	void _sort_and_remove_duplicates();
	void _parse(const string &filename);
public:
	Dataset(
		const string &songs_filename,
		const string &test_users_filename,
		const vector<string> &visible_utility_filenames,
		const vector<string> &hidden_utility_filenames,
		const string &song_tracks_filename,
		float train_test_split);
	const vector<UIR> &uirs_uv() const { return _uirs_uv; }
	const vector<UIR> &uirs_iv() const { return _uirs_iv; }
	const vector<Range> &ranges_uv() const { return _ranges_uv; }
	const vector<Range> &ranges_iv() const { return _ranges_iv; }
	const vector<pair<string, UserType> > &inv_users() const {
		return _inv_users;
	}
	const unordered_map<string, int> item_hash() const {
		return _item_hash;
	}
	const vector<string> &inv_items() const {
		return _inv_items;
	}
	int user_count() const { return _user_count; }
	int item_count() const { return _item_count; }
	int N() const { return _N; }
	const vector<UIR> &uirs_hidden_valid() const { return _uirs_hidden_valid; }
	const vector<UIR> &uirs_hidden_test() const { return _uirs_hidden_test; }

	int collect_user_neighbors(
		const int u, vector<PII> &index_pairs) const;
	int collect_item_neighbors(
		const int i, vector<PII> &index_pairs) const;
	void check_users(const string &userfile);
	void check_items(const string &itemfile);
	void unit_test();
	int common_user_count(const int i, const int j) const;

	float similarity_ij(int i, int j) const;  // slow;
	float similarity_uv(int u, int v) const;  // slow;
	float similarity_ij(
		const string &is, const string &js) const;  // slow;
	float similarity_uv(
		const string &us, const string &vs) const;  // slow;

	void apply_tfidf_weights();
	void apply_bm25_weights();

	void stat();
	vector<int> rec_users() const;
	void write_recommendations(
		const string &filename, const Recommendations &recommendations) const;
	void read_recommendations(
		const string &filename, Recommendations &recommendations) const;
	void write_uirs(const string &filename);
	const vector<int> &item_clusters() const { return _item_clusters; }
};

//////////////////////////////////////////////////////////////////////////
//
// class Recommendations -- keep track of one unit of recommendations
//
//////////////////////////////////////////////////////////////////////////

class Recommendations {

	int _user_count;
	int _valid_count;
	int _test_count;
	vector<Range> _ranges;  // _ranges[u] gives the storage range;
	vector<Score> _recs;
	
public:
	Recommendations(const Dataset &datasset);
	~Recommendations() {}
	void set(int u, int k, Score recs);
	Score get(int u, int k) const;
	void clear();
};

//////////////////////////////////////////////////////////////////////////
//
// class RecommendationCollection -- a list of Recommendations
//
//////////////////////////////////////////////////////////////////////////

class RecommendationCollection {
	vector<pair<Recommendations *, float> > _collection;
public:
	RecommendationCollection() {}
	~RecommendationCollection() {
		for (auto r: _collection) delete r.first;
		_collection.clear();
	}
	int size() const { return _collection.size(); }
	pair<Recommendations *, float> operator[](const int i) const {
		return _collection[i];
	}
	void load(const Dataset &dataset);
};

//////////////////////////////////////////////////////////////////////////
//
// class Dataset -- implementation;
//
//////////////////////////////////////////////////////////////////////////

Dataset::Dataset(
	const string &songs_filename,
	const string &test_users_filename,
	const vector<string> &visible_utility_filenames,
	const vector<string> &hidden_utility_filenames,
	const string &song_tracks_filename,
	float train_test_split) :
	_user_count(0), _valid_user_count(0), _test_user_count(0), _item_count(0), _N(0) {

	// (1) setup;

	_user_count = 0;
	_item_count = 0;
	auto random_engine = default_random_engine(42);

	// (2) read songs;

	cout << "parsing " << songs_filename << " ..." << endl;
	ifstream fp_songs(songs_filename);
	string s;
	while (fp_songs >> s) {
		if (_item_hash.find(s) == _item_hash.end()) {
			_inv_items.push_back(s);
			_item_hash[s] = _item_count++;
		} else {
			assert(0);  // not acceptable;
		}
	}
	fp_songs.close();
	cout << "parsing " << songs_filename << " ... done" << endl;
	cout << "found " << _item_count << " new items" << endl;

	// (2) read test users;

	cout << "parsing " << test_users_filename << " ..." << endl;
	ifstream fp_test_users(test_users_filename);
	string u;
	while (fp_test_users >> u) {
		if (_user_hash.find(u) == _user_hash.end()) {
			_inv_users.push_back(make_pair(u, UserType::Test));
			_user_hash[u] = _user_count++;
			_test_user_count++;
		} else {
			assert(0);  // not acceptable;
		}
	}
	fp_test_users.close();
	cout << "parsing " << test_users_filename << " ... done" << endl;
	cout << "found " << _user_count << " new users" << endl;

	// (2) read triplets (visible);

	for (string filename: visible_utility_filenames) {
		int previous_user_count = _user_count;
		int previous_item_count = _item_count;
		int previous_rating_count = _uirs_uv.size();
		cout << "parsing (visible) " << filename << " ..." << endl;
		ifstream fp_in(filename);
		string u, i;
		int r;
		while (fp_in >> u >> i >> r) {
			if (_user_hash.find(u) == _user_hash.end()) {
				_inv_users.push_back(make_pair(u, UserType::Train));
				_user_hash[u] = _user_count++;
			}
			if (_item_hash.find(i) == _item_hash.end()) {
				_inv_items.push_back(i);
				_item_hash[i] = _item_count++;
			}
			const int u_int = _user_hash[u];
			const int i_int = _item_hash[i];
			_uirs_uv.push_back(make_tuple(u_int, i_int, r));
		}
		fp_in.close();
		cout << "parsing (visible) " << filename << " ... done" << endl;
		cout << "found " <<
			(_user_count - previous_user_count) << " new users" << endl;
		cout << "found " <<
			(_item_count - previous_item_count) << " new items" << endl;
		cout << "found " <<
			(_uirs_uv.size() - previous_rating_count) << " new ratings" << endl;
	}

	_sort_and_remove_duplicates();
	_N = _uirs_uv.size();

	cout << "total: found " << _user_count << " users" << endl;
	cout << "total: found " << _item_count << " items" << endl;
	cout << "total: found " << _uirs_uv.size() << " ratings" << endl;

	// (3) read triplets (hidden);

	for (string filename: hidden_utility_filenames) {
		int previous_user_count = _user_count;
		int previous_item_count = _item_count;
		int previous_rating_count = _uirs_uv.size();
		cout << "parsing (hidden) " << filename << " ..." << endl;
		ifstream fp_in(filename);
		string u, i;
		int r;
		while (fp_in >> u >> i >> r) {
			if (_user_hash.find(u) == _user_hash.end()) {
				assert(0);  // hidden user must be found;
			}
			if (_item_hash.find(i) == _item_hash.end()) {
				assert(0);  // hidden item must be found;
			}
			const int u_int = _user_hash[u];
			const int i_int = _item_hash[i];
			_uirs_hidden_test.push_back(make_tuple(u_int, i_int, r));
		}
		fp_in.close();
		cout << "parsing (hidden) " << filename << " ... done" << endl;
	}
	cout << "sorting uirs_hidden_test ... (" <<
		_uirs_hidden_test.size() << ")" << endl;
	sort(_uirs_hidden_test.begin(),
		_uirs_hidden_test.end(), compare_uir_uv);
	for (int index = 1; index < _uirs_hidden_test.size(); index++) {
		const int previous_u = get<0>(_uirs_hidden_test[index - 1]);
		const int previous_i = get<1>(_uirs_hidden_test[index - 1]);
		const int u = get<0>(_uirs_hidden_test[index]);
		const int i = get<1>(_uirs_hidden_test[index]);
		assert(previous_u <= u);
		assert(previous_u < u || previous_i < i);
	}
	cout << "sorting uirs_hidden_test ... done" << endl;

	// (2) train test split (actually train --> train + valid);

	cout << "train test split ..." << endl;
    vector<int> random_indices;
	for (int i = 0; i < _user_count; i++) {
		if (_inv_users[i].second == UserType::Test)
			continue;  // this is a test user;
		random_indices.push_back(i);
	}
	assert(random_indices.size() == _user_count - _test_user_count);
	shuffle(random_indices.begin(), random_indices.end(), random_engine);
	_valid_user_count = int(float(random_indices.size()) * train_test_split);
	int train_user_count = _user_count - _test_user_count - _valid_user_count;
	cout << "train_user_count = " << train_user_count << endl;
	cout << "valid_user_count = " << _valid_user_count << endl;
	cout << "test_user_count = " << _test_user_count << endl;
	assert(train_user_count + _valid_user_count + _test_user_count ==
		_user_count);

	for (int i = 0; i < _valid_user_count; i++) {
		int actual_i = random_indices[i];
		assert(_inv_users[actual_i].second == UserType::Train);
		_inv_users[actual_i].second = UserType::Valid;  // turn it into a valid;
	}

	auto tuple_INT_MAX = make_tuple(INT_MAX, INT_MAX, INT_MAX);
	int i_end = 0;
	while (i_end < _uirs_uv.size()) {
		int i_start = i_end;
		i_end++;
		const int u = get<0>(_uirs_uv[i_start]);
		while (i_end < _uirs_uv.size() && get<0>(_uirs_uv[i_end]) == u)
			i_end++;
		if (_inv_users[u].second == UserType::Valid) {
			const int count = i_end - i_start;
			const int half_count = count / 2;
			random_indices.resize(count);
			for (int i = 0; i < count; i++)
				random_indices[i] = i_start + i;
			shuffle(random_indices.begin(), random_indices.end(),
				default_random_engine(42));
			for (int i = 0; i < half_count; i++) {
				const int ii = random_indices[i];
				assert(ii >= i_start && ii < i_end);
				auto UIR = _uirs_uv[ii];
				_uirs_hidden_valid.push_back(UIR);
				_uirs_uv[ii] = tuple_INT_MAX;
					// to be removed;
			}			
		}
	}
	cout << "sorting uirs_hidden_valid ... (" <<
		_uirs_hidden_valid.size() << ")" << endl;
	sort(_uirs_hidden_valid.begin(),
		_uirs_hidden_valid.end(), compare_uir_uv);
	cout << "sorting uirs_hidden_valid ... done" << endl;

	int j = 0;
	for (int i = 0; i < _uirs_uv.size(); i++) {
		if (_uirs_uv[i] != tuple_INT_MAX) {
			_uirs_uv[j++] = _uirs_uv[i];
		}
	}
	_uirs_uv.resize(j);
	assert(_N == _uirs_uv.size() + _uirs_hidden_valid.size());
	_N = _uirs_uv.size();
	cout << "train test split ... done" << endl;

	// (4) build _uirs_iv;

	_uirs_iv = _uirs_uv;  // a copy;
	cout << "sorting uirs_iv ..." << endl;
	sort(_uirs_iv.begin(), _uirs_iv.end(), compare_uir_iv);
	cout << "sorting uirs_iv ... done" << endl;

	// (3) create both uirs_uv (sorted by u, then i), and
	//	 uirs_iv (sorted by u, then i);

	cout << "computing ranges ..." << endl;
	_ranges_uv = vector<Range>(_user_count, make_pair(0, 0));
	_ranges_iv = vector<Range>(_item_count, make_pair(0, 0));
	for (int index = 0; index < _N; index++) {
		const int u = get<0>(_uirs_uv[index]);
		if (_ranges_uv[u].second == 0) {
			_ranges_uv[u].first = index;  // first time;
		}
		_ranges_uv[u].second = index + 1;
	}
	for (int index = 0; index < _N; index++) {
		const int i = get<1>(_uirs_iv[index]);
		if (_ranges_iv[i].second == 0) {
			_ranges_iv[i].first = index;  // first time;
		}
		_ranges_iv[i].second = index + 1;
	}
	cout << "computing ranges ... done" << endl;

	// (5) statistics;

	long long int uu_count = 0;
	for (int u = 0; u < _user_count; u++) {
		const int u_count = _ranges_uv[u].second - _ranges_uv[u].first;
		uu_count += u_count * u_count;
	}
	long long int ii_count = 0;
	for (int i = 0; i < _item_count; i++) {
		const int i_count = _ranges_iv[i].second - _ranges_iv[i].first;
		ii_count += i_count * i_count;
	}
	cout << "total: UU_count = " << uu_count << endl;
	cout << "total: II_count = " << ii_count << endl;
}

//////////////////////////////////////////////////////////////////////////
//
// class Dataset -- implementation;
//
//////////////////////////////////////////////////////////////////////////

float Dataset::similarity_ij(int i, int j) const {

	const auto &uirs_ui_from = _ranges_iv[i].first;
	const auto &uirs_ui_to = _ranges_iv[i].second;
	const auto &uirs_uj_from = _ranges_iv[j].first;
	const auto &uirs_uj_to = _ranges_iv[j].second;
	const int card_I = uirs_ui_to - uirs_ui_from;
	const int card_J = uirs_uj_to - uirs_uj_from;
	if (card_I == 0 || card_J == 0) return 0;

	int card_IJ = 0;
	int uirs_ui = uirs_ui_from;
	int uirs_uj = uirs_uj_from;
	while (true) {
		if (uirs_ui == uirs_ui_to || uirs_uj == uirs_uj_to)
			break;
		int ui = get<0>(_uirs_iv[uirs_ui]);
		int uj = get<0>(_uirs_iv[uirs_uj]);
		if (ui == uj) {
			card_IJ++;
			uirs_ui++;
			uirs_uj++;
		} else if (ui < uj) {
			uirs_ui++;
		} else {
			uirs_uj++;
		}
	}
	if (Global::Jaccard) {
		return card_IJ > 0 ?
			float(card_IJ) / float(card_I + card_J - card_IJ) : 0;
	} else {
		return card_IJ /
			(pow(card_I, Global::alpha) * pow(card_J, 1 - Global::alpha));
	}
}

float Dataset::similarity_uv(int u, int v) const {

	const auto &uirs_ui_from = _ranges_uv[u].first;
	const auto &uirs_ui_to = _ranges_uv[u].second;
	const auto &uirs_vi_from = _ranges_uv[v].first;
	const auto &uirs_vi_to = _ranges_uv[v].second;
	const int card_U = uirs_ui_to - uirs_ui_from;
	const int card_V = uirs_vi_to - uirs_vi_from;
	if (card_U == 0 || card_V == 0) return 0;

	int card_UV = 0;
	int uirs_ui = uirs_ui_from;
	int uirs_vi = uirs_vi_from;
	while (true) {
		if (uirs_ui == uirs_ui_to || uirs_vi == uirs_vi_to)
			break;
		int ui = get<1>(_uirs_uv[uirs_ui]);
		int vi = get<1>(_uirs_uv[uirs_vi]);
		if (ui == vi) {
			card_UV++;
			uirs_ui++;
			uirs_vi++;
		} else if (ui < vi) {
			uirs_ui++;
		} else {
			uirs_vi++;
		}
	}
	if (Global::Jaccard) {
		return card_UV > 0 ?
			float(card_UV) / float(card_U + card_V - card_UV) : 0;
	} else {
		return card_UV /
			(pow(card_U, Global::alpha) * pow(card_V, 1 - Global::alpha));
	}
}

void Dataset::apply_tfidf_weights() {

	// treat each item as a document, and each user as a term;

	vector<float> idf(_user_count, 0);
	for (int u = 0; u < _user_count; u++) {
		int rated_item_count = _ranges_uv[u].second - _ranges_uv[u].first;
		idf[u] = log(_item_count / max(1, rated_item_count));
	}

	// (2) update each sparse entry;

	for (int index = 0; index < _N; index++) {  // to the _uirs_uv[] matrix;
		const float t = get<2>(_uirs_uv[index]);
		const float tf = sqrt(t);
		const float weight = tf * idf[get<0>(_uirs_uv[index])];
		get<2>(_uirs_uv[index]) = weight;
		// cout << "tfidf: " << t << " --> " << weight << endl;
	}
	for (int index = 0; index < _N; index++) {  // to the _uirs_iv[] matrix;
		const float t = get<2>(_uirs_iv[index]);
		const float tf = sqrt(t);
		const float weight = tf * idf[get<0>(_uirs_iv[index])];
		get<2>(_uirs_iv[index]) = weight;
	}
}

void Dataset::apply_bm25_weights() {

	// treat each item as a document, and each user as a term;

	vector<float> idf(_user_count, 0);
	for (int u = 0; u < _user_count; u++) {
		int rated_item_count = _ranges_uv[u].second - _ranges_uv[u].first;
		idf[u] = log(_item_count / max(1, rated_item_count));
	}

	vector<float> normalizers(_item_count, 0);
	for (int i = 0; i < _item_count; i++) {
		float rating_sum = 0;
		int i_from = _ranges_iv[i].first;
		int i_to = _ranges_iv[i].second;
		for (int index = _ranges_iv[i].first; index < _ranges_iv[i].second;
				index++) {
			rating_sum += get<2>(_uirs_iv[index]);
		}
		normalizers[i] = rating_sum;
	}

	float item_average = accumulate(
		normalizers.begin(), normalizers.end(), 0, plus<int>()) / _item_count;
	const float B = 0.8;
	const float K1 = 100;
	for (int i = 0; i < _item_count; i++) {
		normalizers[i] = (1.0 - B) + B * (normalizers[i] / item_average);
	}

	// (2) update each sparse entry;

	for (int index = 0; index < _N; index++) {  // to the _uirs_uv[] matrix;
		const int u = get<0>(_uirs_uv[index]);
		const int i = get<1>(_uirs_uv[index]);
		const float t = get<2>(_uirs_uv[index]);
		const float weight =
			t * (K1 + 1.0) / (K1 * normalizers[i] + t) * idf[u];
		get<2>(_uirs_uv[index]) = weight;
		// cout << "bm25: " << t << " --> " << weight << endl;
	}
	for (int index = 0; index < _N; index++) {  // to the _uirs_iv[] matrix;
		const int u = get<0>(_uirs_iv[index]);
		const int i = get<1>(_uirs_iv[index]);
		const float t = get<2>(_uirs_iv[index]);
		const float weight =
			t * (K1 + 1.0) / (K1 * normalizers[i] + t) * idf[u];
		get<2>(_uirs_iv[index]) = weight;
	}
}

float Dataset::similarity_ij(
	const string &is, const string &js) const {

	if (_item_hash.find(is) == _item_hash.end()) return 0;
	if (_item_hash.find(js) == _item_hash.end()) return 0;
	const int i = _item_hash.find(is)->second;
	const int j = _item_hash.find(js)->second;
	return similarity_ij(i, j);
}

float Dataset::similarity_uv(
	const string &us, const string &vs) const {

	if (_user_hash.find(us) == _user_hash.end()) return 0;
	if (_user_hash.find(vs) == _user_hash.end()) return 0;
	const int u = _user_hash.find(us)->second;
	const int v = _user_hash.find(vs)->second;
	return similarity_uv(u, v);
}

void Dataset::check_users(const string &userfile) {
	cout << "parsing " << userfile << + " ..." << endl;
	ifstream fp_in(userfile);
	string u;
	int notfound_count = 0;
	int all_count = 0;
	vector<string> notfounds;
	while (fp_in >> u) {
		all_count++;
		if (_user_hash.find(u) == _user_hash.end()) {
			notfound_count++;
			notfounds.push_back(u);
		}
	}
	cout << "check_users() found " << notfound_count <<
		" unknown users (out of " << all_count << ")" << endl;
	if (notfound_count > 0) {
		for (int i = 0; i < min(10, notfounds.size()); i++) {
			cout << "	example: " << notfounds[i] << endl;
		}
	}
}

void Dataset::check_items(const string &itemfile) {
	cout << "parsing " << itemfile << + " ..." << endl;
	ifstream fp_in(itemfile);
	string s;
	int notfound_count = 0;
	int all_count = 0;
	vector<string> notfounds;
	while (fp_in >> s) {
		all_count++;
		if (_item_hash.find(s) == _item_hash.end()) {
			notfound_count++;
			notfounds.push_back(s);
		}
	}
	cout << "check_items() found " << notfound_count <<
		" unknown items (out of " << all_count << ")" << endl;
	if (notfound_count > 0) {
		for (int i = 0; i < min(10, notfounds.size()); i++) {
			cout << "	example: " << notfounds[i] << endl;
		}
	}
}

void Dataset::_sort_and_remove_duplicates() {

	int N = _uirs_uv.size();
	bool inorder = false;
	for (int i = 1; i < N; i++) {
		if (!compare_uir_uv(_uirs_uv[i - 1], _uirs_uv[i])) {
			inorder = false;
			break;
		}
	}
	if (!inorder) {
		cout << "sorting uirs_uv ..." << endl;
		sort(_uirs_uv.begin(), _uirs_uv.end(), compare_uir_uv);
		cout << "sorting uirs_uv ... done" << endl;
	}

	// (3) remove duplicate ratings;

	cout << "removing duplicates ..." << endl;
	int del_count = 0;
	int j = 0;  // last;
	for (int i = 1; i < N; i++) {
		if (get<0>(_uirs_uv[i]) == get<0>(_uirs_uv[j]) &&
			get<1>(_uirs_uv[i]) == get<1>(_uirs_uv[j])) {
			_uirs_uv[j] = _uirs_uv[i];
			del_count++;
		} else {
			_uirs_uv[++j] = _uirs_uv[i];
		}
	}
	N = _uirs_uv.size() - del_count;
	_uirs_uv.resize(N);
	cout << "deleted " << del_count << " multiple ratings (kept last)";
	cout << endl; 

	// (4) return user_count and item_count;

	int user_max = 0;
	int item_max = 0;
	for (int i = 0; i < N; i++) {
		if (user_max < get<0>(_uirs_uv[i]))
			user_max = get<0>(_uirs_uv[i]);
		if (item_max < get<1>(_uirs_uv[i]))
			item_max = get<1>(_uirs_uv[i]);
	}

	_user_count = user_max + 1;
	_item_count = item_max + 1;
	_N = _uirs_uv.size();
}

int Dataset::collect_user_neighbors(
	const int u, vector<pair<int, int> > &neighbors) const {

	// Return a list of all possible pairs of indices
	// (index_uv, index_iv), such that the first is an index into
	// uirs_uv, and the second is an index into uirs_iv. Let's say
	// the indices point to (u1, i1, r1) and (u2, i2, r2), we would
	// have: u1 = u, and i1 = i2.

	neighbors.clear();
	const int from_uv = _ranges_uv[u].first;
	const int to_uv = _ranges_uv[u].second;
	for (int index_uv = from_uv; index_uv < to_uv; index_uv++) {
		const int i = get<1>(_uirs_uv[index_uv]);
		const int from_iv = _ranges_iv[i].first;
		const int to_iv = _ranges_iv[i].second;
		for (int index_iv = from_iv; index_iv < to_iv; index_iv++) {
			const int other_u = get<0>(_uirs_iv[index_iv]);
			if (u != other_u) {
				neighbors.push_back(make_pair(index_uv, index_iv));
			}
		}
	}
	return neighbors.size();
}

int Dataset::collect_item_neighbors(
	const int i, vector<pair<int, int> > &neighbors) const {

	// Return a list of all possible pairs of indices
	// (index_uv, index_iv), such that the first is an index into
	// uirs_uv, and the second is an index into uirs_iv. Let's say
	// the indices point to (u1, i1, r1) and (u2, i2, r2), we would
	// have: i2 = i, and u1 = u2.

	neighbors.clear();
	const int from_iv = _ranges_iv[i].first;
	const int to_iv = _ranges_iv[i].second;
	for (int index_iv = from_iv; index_iv < to_iv; index_iv++) {
		const int u = get<0>(_uirs_iv[index_iv]);
		const int from_uv = _ranges_uv[u].first;
		const int to_uv = _ranges_uv[u].second;
		for (int index_uv = from_uv; index_uv < to_uv; index_uv++) {
			const int other_i = get<1>(_uirs_uv[index_uv]);
			if (i != other_i) {
				neighbors.push_back(make_pair(index_uv, index_iv));
			}
		}
	}
	return neighbors.size();
}

int Dataset::common_user_count(const int i, const int j) const {

	// compute the number of users who have rated both items;

	const Range &range_i = _ranges_iv[i];
	const Range &range_j = _ranges_iv[j];
	const UIR *base = &_uirs_iv[0];
	const UIR *ip = base + range_i.first;
	const UIR *ip_end = base + range_i.second;
	const UIR *jp = base + range_j.first;
	const UIR *jp_end = base + range_j.second;
	if (ip == ip_end || jp == jp_end) return 0;
	int ui = get<0>(*ip);
	int uj = get<0>(*jp);
	int count = 0;
	while (true) {
		if (ui < uj) {
			if (++ip == ip_end) return count;
			ui = get<0>(*ip);
		} else if (ui > uj) {
			if (++jp == jp_end) return count;
			uj = get<0>(*jp);
		} else {
			count++;
			if (++ip == ip_end) return count;
			if (++jp == jp_end) return count;
			ui = get<0>(*ip);
			uj = get<0>(*jp);
		}
	}
}

void Dataset::unit_test() {}

void Dataset::stat() {
	vector<int> ui_counts;
	for (int u = 0; u < _user_count; u++) {
		ui_counts.push_back(_ranges_uv[u].second - _ranges_uv[u].first);
	}
	for (int step = 5; step < 100; step += 5) {
		int p = (step * _user_count) / 100;
		nth_element(
			ui_counts.begin(), ui_counts.begin() + p, ui_counts.end());
		cout << "STAT users " << step << "%: " << ui_counts[p] << endl;
	}
	vector<int> iu_counts;
	for (int i = 0; i < _item_count; i++) {
		iu_counts.push_back(_ranges_iv[i].second - _ranges_iv[i].first);
	}
	for (int step = 5; step < 100; step += 5) {
		int p = (step * _item_count) / 100;
		nth_element(
			iu_counts.begin(), iu_counts.begin() + p, iu_counts.end());
		cout << "STAT items " << step << "%: " << iu_counts[p] << endl;
	}
}

vector<int> Dataset::rec_users() const {
	vector<int> rec_users;
	const bool recommend_test_users = Global::recommend_test_users;
	for (int u = 0; u < _user_count; u++) {
		auto user_type = _inv_users[u].second;
		if (user_type == Dataset::UserType::Train)
			continue; // no need to recommend;
		if (user_type == Dataset::UserType::Test && !recommend_test_users)
			continue; // no need to recommend;
		rec_users.push_back(u);
	}
	return rec_users;
}

void Dataset::read_recommendations(
	const string &filename, Recommendations &recommendations) const {

	cout << "reading recommendations " << filename << " ..." << endl;
	recommendations.clear();
	ifstream fp(filename);
	string line;
	string token;
	int line_count = 0;
	vector<string> tokens;
	while (getline(fp, line)) {
		stringstream ss(line);
		tokens.clear();
		while (getline(ss, token, ','))
			tokens.push_back(token);
		assert(tokens.size() == 2 * Global::recK + 1);
		int u = stoi(tokens[0]);
		for (int k = 0; k < Global::recK; k++) {
			int index = stoi(tokens[2 * k + 1]);
			int score = stof(tokens[2 * k + 2]);
			recommendations.set(u, k, Score(score, index));
		}
	}
	cout << "reading recommendations " << filename << " ... DONE" << endl;
}

void Dataset::write_recommendations(
	const string &filename, const Recommendations &recommendations) const {

	ofstream fp(filename);
	auto rec_users = this->rec_users();
	for (auto u: rec_users) {
		fp << u;
		for (int k = 0; k < Global::recK; k++) {
			auto score = recommendations.get(u, k);
			fp << "," << score.index;
			fp << "," << std::fixed << std::setprecision(6) << score.score;
		}
		fp << endl;
	}
	fp.close();
}

void Dataset::write_uirs(const string &filename) {
	cout << "writing " << filename << " ..." << endl;
	ofstream fp(filename);
	for (auto uir: _uirs_uv) {
		fp << get<0>(uir) << "," << get<1>(uir) << "," << get<2>(uir) << endl;
	}
	cout << "writing " << filename <<
		" ... (" << _uirs_uv.size() << ") DONE" << endl;
}

//////////////////////////////////////////////////////////////////////////
//
// class RecommendationCollection -- implementation;
//
//////////////////////////////////////////////////////////////////////////

void RecommendationCollection::load(const Dataset &dataset) {
	assert(_collection.size() == 0);
	const auto &ensemble_inputs = Global::ensemble_inputs;
	const int ensemble_input_count = ensemble_inputs.size();
	assert(ensemble_input_count > 0);
	for (int e = 0; e < ensemble_input_count; e++) {
		Recommendations *recommendations = new Recommendations(dataset);
		const string filename = ensemble_inputs[e].first;
		const float weight = ensemble_inputs[e].second;
		dataset.read_recommendations(filename, *recommendations);
		_collection.push_back(make_pair(recommendations, weight));
			// the recommendations are owned by this class;
	}
}

//////////////////////////////////////////////////////////////////////////
//
// class Recommendations -- implementation;
//
//////////////////////////////////////////////////////////////////////////

Recommendations::Recommendations(const Dataset &dataset) :
	_user_count(dataset.user_count()),
	_valid_count(0), _test_count(0) {

	_ranges.resize(_user_count);
	fill(_ranges.begin(), _ranges.end(), make_pair(INT_MAX, INT_MAX));
	int current_p = 0;
	const auto &inv_users = dataset.inv_users();
	for (int u = 0; u < _user_count; u++) {
		if (inv_users[u].second == Dataset::UserType::Valid) {
			_valid_count++;
			_ranges[u] = make_pair(current_p, current_p + Global::recK);
			current_p += Global::recK;
		} else if (inv_users[u].second == Dataset::UserType::Test) {
			if (Global::recommend_test_users) {
				_test_count++;
				_ranges[u] = make_pair(current_p, current_p + Global::recK);
				current_p += Global::recK;
			}
		}
	}
	_recs.resize(current_p);
	fill(_recs.begin(), _recs.end(), Score(0, INT_MAX));
}

void Recommendations::clear() {
	fill(_recs.begin(), _recs.end(), Score(0, INT_MAX));
}

void Recommendations::set(int u, int k, Score rec) {
	assert(u >= 0 && u < _user_count);
	const int from = _ranges[u].first;
	assert(from != INT_MAX);
	assert(k >= 0 && k < Global::recK);
	_recs[_ranges[u].first + k] = rec;
}

Score Recommendations::get(int u, int k) const {
	assert(u >= 0 && u < _user_count);
	const int from = _ranges[u].first;
	assert(from != INT_MAX);
	assert(k >= 0 && k < Global::recK);
	return _recs[_ranges[u].first + k];
}

//////////////////////////////////////////////////////////////////////////
//
// class Evaluator -- Evaluate mAP
//
//////////////////////////////////////////////////////////////////////////

class Evaluator {
	mutex _mutex;
	vector<Dataset::UserType> _user_types;
	vector<float> _mAPs;
	vector<float> _sum_precision_at_K;  // statistics;
	float _sum_mAP_valid;
	float _sum_mAP_test;
	float _sum_hit_valid;
	float _sum_hit_test;
	int _count_mAP_valid;
	int _count_mAP_test;
	vector<UIR> _uirs_hidden_valid;  // a copy;
	vector<UIR> _uirs_hidden_test;  // a copy;
	vector<Range> _ranges_hidden_valid;
	vector<Range> _ranges_hidden_test;

	void _evaluate(const vector<int> &users,
		const Recommendations &recommendations,
		bool &evaluated_valid, bool &evaluated_test);
public:
	Evaluator(const Dataset &dataset);
	~Evaluator() {}
	void evaluate(const vector<int> &users,
		const Recommendations &recommendations, const bool silent);
	tuple<float, float, int> mAP(Dataset::UserType user_type);
	void reset() {
		_sum_mAP_valid = 0;
		_sum_mAP_test = 0;
		_sum_hit_valid = 0;
		_sum_hit_test = 0;
		_count_mAP_valid = 0;
		_count_mAP_test = 0;
		fill(
			_sum_precision_at_K.begin(),
			_sum_precision_at_K.end(), 0);
	}
	float precision_at_K(int k) const;
	string precision_string() const;  // debugging;
};

Evaluator::Evaluator(const Dataset &dataset) :
	_sum_mAP_valid(0),
	_sum_mAP_test(0),
	_sum_hit_valid(0),
	_sum_hit_test(0),
	_count_mAP_valid(0),
	_count_mAP_test(0),
	_sum_precision_at_K(Global::recK, 0) {

	const int user_count = dataset.user_count();
	_uirs_hidden_valid = dataset.uirs_hidden_valid();  // copy;
	_uirs_hidden_test = dataset.uirs_hidden_test();  // copy;
	_mAPs.resize(user_count);
	fill(_mAPs.begin(), _mAPs.end(), -1);
	_user_types.resize(user_count);
	fill(_user_types.begin(), _user_types.end(), Dataset::UserType::Train);
	for (int u = 0; u < user_count; u++) {
		_user_types[u] = dataset.inv_users()[u].second;
	}

	_ranges_hidden_valid.resize(user_count, make_pair(0, 0));
	_ranges_hidden_test.resize(user_count, make_pair(0, 0));
	fill(_ranges_hidden_valid.begin(), _ranges_hidden_valid.end(),
		make_pair(0, 0));
	fill(_ranges_hidden_test.begin(), _ranges_hidden_test.end(),
		make_pair(0, 0));
	const auto &inv_users = dataset.inv_users();

	for (int index = 0; index < _uirs_hidden_valid.size(); index++) {
		auto rating = _uirs_hidden_valid[index];
		const int u = get<0>(rating);
		if (_ranges_hidden_valid[u].second == 0) {
			_ranges_hidden_valid[u].first = index;  // first time;
		}
		_ranges_hidden_valid[u].second = index + 1;
	}

	for (int index = 0; index < _uirs_hidden_test.size(); index++) {
		auto rating = _uirs_hidden_test[index];
		const int u = get<0>(rating);
		if (_ranges_hidden_test[u].second == 0) {
			_ranges_hidden_test[u].first = index;  // first time;
		}
		_ranges_hidden_test[u].second = index + 1;
	}
}

void Evaluator::evaluate(const vector<int> &users,
	const Recommendations &recommendations, const bool silent) {

	bool evaluated_valid = false, evaluated_test = false;
	_evaluate(users, recommendations, evaluated_valid, evaluated_test);
	auto mAP_valid = mAP(Dataset::UserType::Valid);
	auto mAP_test = mAP(Dataset::UserType::Test);
	if (!silent) {
		cout_mutex.lock();
		if (evaluated_valid) {
			cout << "evaluation (Valid): " <<
				get<0>(mAP_valid) << " (" << get<2>(mAP_valid) <<
				"), hit = " << get<1>(mAP_valid) << endl;
		}
		if (evaluated_test) {
			cout << "evaluation (Test): " <<
				get<0>(mAP_test) << " (" << get<2>(mAP_test) <<
				"), hit = " << get<1>(mAP_test) << endl;
		}
		cout_mutex.unlock();
	}
}

void Evaluator::_evaluate(const vector<int> &users,
	const Recommendations &recommendations,
	bool &evaluated_valid, bool &evaluated_test) {

	evaluated_valid = false;
	evaluated_test = false;
	vector<int> rec_items;
	vector<int> ground_truth_items;
	float this_sum_mAP_valid = 0;
	float this_sum_mAP_test = 0;
	float this_sum_hit_valid = 0;
	float this_sum_hit_test = 0;
	int this_count_mAP_valid = 0;
	int this_count_mAP_test = 0;
	
	vector<float> working_precision_at_K(Global::recK, 0);
		// for calculation of a working vector of precision_at_K;

	for (auto u: users) {
		assert(_mAPs[u] < 0);  // must be the first time;
		const int hidden_valid_count =
			_ranges_hidden_valid[u].second - _ranges_hidden_valid[u].first;
		const int hidden_test_count =
			_ranges_hidden_test[u].second - _ranges_hidden_test[u].first;
		assert(hidden_valid_count == 0 || hidden_test_count == 0);
			// either valid or test;

		fill(
			working_precision_at_K.begin(),
			working_precision_at_K.end(), 0);
		ground_truth_items.clear();
		const auto user_type = _user_types[u];
		if (user_type == Dataset::UserType::Valid) {
			for (int index = _ranges_hidden_valid[u].first;
					index < _ranges_hidden_valid[u].second; index++) {
				auto uir = _uirs_hidden_valid[index];
				assert(get<0>(uir) == u);
				ground_truth_items.push_back(get<1>(uir));
			}
		} else if (user_type == Dataset::UserType::Test) {
			for (int index = _ranges_hidden_test[u].first;
					index < _ranges_hidden_test[u].second; index++) {
				auto uir = _uirs_hidden_test[index];
				assert(get<0>(uir) == u);
				ground_truth_items.push_back(get<1>(uir));
			}
		} else {
			assert(0);
		}

		if (ground_truth_items.size() == 0)
			continue;  // no information whatsoever;

		rec_items.clear();
		for (int k = 0; k < Global::recK; k++) {
			rec_items.push_back(recommendations.get(u, k).index);
		}

		// now evaluate mAP;

		int hit_count = 0;
		float sumAP = 0;
		float current_precision_at_K = 0;
		for (int k = 0; k < Global::recK; k++) {
			int rec = rec_items[k];
			if (binary_search(ground_truth_items.begin(),
				ground_truth_items.end(), rec)) {
				hit_count++;
				current_precision_at_K = float(hit_count) / (k + 1);
				sumAP += current_precision_at_K;
			}
			working_precision_at_K[k] = current_precision_at_K;
		}

		for (int k = 0; k < Global::recK; k++) {
			_sum_precision_at_K[k] += working_precision_at_K[k];
				// statistics across all _evaluate(u);
		}

		const float mAP = sumAP / ground_truth_items.size();
		assert(mAP >= 0);
		_mAPs[u] = mAP;
		if (user_type == Dataset::UserType::Valid) {
			this_sum_mAP_valid += mAP;
			this_sum_hit_valid += float(hit_count) / ground_truth_items.size();
			this_count_mAP_valid++;
			evaluated_valid = true;  // an actual non-empty evaluation;
		} else {
			this_sum_mAP_test += mAP;
			this_sum_hit_test += float(hit_count) / ground_truth_items.size();
			this_count_mAP_test++;
			evaluated_test = true;  // an actual non-empty evaluation;
		}
	}

	_mutex.lock();
	_sum_mAP_valid += this_sum_mAP_valid;
	_sum_hit_valid += this_sum_hit_valid;
	_count_mAP_valid += this_count_mAP_valid;
	_sum_mAP_test += this_sum_mAP_test;
	_sum_hit_test += this_sum_hit_test;
	_count_mAP_test += this_count_mAP_test;
	_mutex.unlock();
}

tuple<float, float, int> Evaluator::mAP(Dataset::UserType user_type) {
	_mutex.lock();
	auto result = user_type == Dataset::UserType::Valid ?
		make_tuple(
			_sum_mAP_valid / max(1, _count_mAP_valid),
			_sum_hit_valid / max(1, _count_mAP_valid),
			_count_mAP_valid) :
		make_tuple(
			_sum_mAP_test / max(1, _count_mAP_test),
			_sum_hit_test / max(1, _count_mAP_test),
			_count_mAP_test);
	_mutex.unlock();
	return result;
}

float Evaluator::precision_at_K(int k) const {
	assert(k >= 0 && k < Global::recK);
	int count = _count_mAP_valid + _count_mAP_test;
	return count == 0 ? 0 : (_sum_precision_at_K[k] / count);
}

string Evaluator::precision_string() const {
	stringstream ss;
	for (int k = 0; k < Global::recK; k++) {
		ss << precision_at_K(k);
		ss << ",";
		if ((k + 1) % 10 == 0)
			ss << endl;
	}
	return ss.str();
}

//////////////////////////////////////////////////////////////////////////
//
// class Embeddings -- from lyrics (a track-specific concept);
//
//////////////////////////////////////////////////////////////////////////

class Embeddings {  // song embeddings;
	int _dimension;
	int _track_count;
	unordered_map<string, int> _track_hash;
	float *_embeddings;
	int _embedding_allocation;
	vector<float *> _item_to_embeddings;

public:
	Embeddings(
		const Dataset &dataset,
		const string &embeddings_filename,
		const string &song_tracks_filename,
		const vector<string> &skips);
	~Embeddings() {
		if (_embeddings) delete[] _embeddings;
	}
	int dimension() const { return _dimension; }
	int track_count() const { return _track_count; }
	const unordered_map<string, int> &track_hash() const {
		return _track_hash;
	}
	float *embeddings(int i) const { return _item_to_embeddings[i]; }
		// using the item index;
	static float similarity(const float *ei, const float *ej, int dimension);
};

float Embeddings::similarity(
	const float *ei, const float *ej, int dimension) {

	float sum = 0;
	for (int d = 0; d < dimension; d++) {
		sum += ei[d] * ej[d];
	}
	return sum;
}

Embeddings::Embeddings(
	const Dataset &dataset,
	const string &embeddings_filename,
	const string &song_tracks_filename,
	const vector<string> &skips) :
	_dimension(0), _track_count(0),
	_embeddings(nullptr), _embedding_allocation(0) {

	// read embeddings from a csv file with header. The dimension of the
	// embeddings is inferred;

	const int item_count = dataset.item_count();
	_item_to_embeddings.resize(dataset.item_count());
	fill(_item_to_embeddings.begin(), _item_to_embeddings.end(), nullptr);
	const auto &item_hash = dataset.item_hash();
	const auto &inv_items = dataset.inv_items();

	ifstream fp(embeddings_filename);
	assert(fp.is_open());
	string line;
	assert(getline(fp, line));  // read the header;
	stringstream ss(line);
	string token;

	vector<bool> is_dimension_data;
	int column_count = 0;
	assert(getline(ss, token, ','));  // first column;
	is_dimension_data.push_back(false);  // first column;
	column_count++;  // first column;

	while (getline(ss, token, ',')) {
		const bool skipped = find(
			skips.begin(), skips.end(), token) != skips.end();
		bool is_dimension = column_count > 0 && !skipped;
		is_dimension_data.push_back(is_dimension);
		column_count++;
	}
	assert(column_count == is_dimension_data.size());

	_track_count = 0;
	while (getline(fp, line)) {
		_track_count++;  // except the first line;
	}
	fp.close();

	_dimension = count(
		is_dimension_data.begin(), is_dimension_data.end(), true);

	cout << "parsing embeddings of " << _dimension << " dimensions and "
		<< _track_count << " tracks" << endl;

	vector<string> tokens;
	fp.open(embeddings_filename);
	assert(getline(fp, line));  // skip the header;

	assert(_embeddings == nullptr);
	_embedding_allocation = _track_count * _dimension;
	_embeddings = new float[_embedding_allocation];
	fill(_embeddings, _embeddings + _embedding_allocation, 0);
	string track;
	for (int track_i = 0; track_i < _track_count; track_i++) {
		assert(getline(fp, line));
		stringstream ss(line);
		assert(getline(ss, track, ','));
		_track_hash[track] = track_i;  // track name first;
		float sum_square = 0;

		int d = 0;
		for (int c = 1; c < column_count; c++) {  // first column already read;
			assert(getline(ss, token, ','));
			if (!is_dimension_data[c])
				continue;
			float value = stof(token);
			const int e = track_i * _dimension + d;
			assert(d < _dimension && e < _embedding_allocation);
			_embeddings[e] = value;
			sum_square += value * value;
			d++;
		}	
		assert(d == _dimension);  // must have read all dimensions;

		if (sum_square > 0) {
			float inv_sq = 1.0 / sqrt(sum_square);
			for (int d = 0; d < _dimension; d++) {
				const int e = track_i * _dimension + d;
				assert(e < _embedding_allocation);
				_embeddings[e] *= inv_sq;
					// normalize;
			}
		}
	}
    fp.close();

	// (2) read song-tracks

	ifstream fp_song_tracks(song_tracks_filename);
	bool successful = true;
	int line_number = 0;
	int matching_song_tracks_count = 0;
	int song_with_zero_track_count = 0;
	int song_with_multiple_track_count = 0;
	while (getline(fp_song_tracks, line)) {
		stringstream ss(line);
		tokens.clear();
		while (getline(ss, token, '\t')) {
			tokens.push_back(token);
		}
		const int token_count = tokens.size();
		assert(token_count >= 1);
		assert(line_number < item_count);
		assert(inv_items[line_number] == tokens[0]);  // must be a match;
		if (token_count == 1) {
			song_with_zero_track_count++;
			// cout << "found " << tokens[0] << " without any tracks" << endl;
		} else {
			if (token_count > 2) song_with_multiple_track_count++;
			// if (token_count > 2) {
			// 	cout << "found " << tokens[0] << " with multiple tracks: ";
			// 	for (int t = 1; t < token_count; t++)
			// 		cout << tokens[t] << " ";
			// 	cout << endl;
			// }
			const string song = tokens[0];
			assert(item_hash.find(song) != item_hash.end());
				// the song must be found;
			const int i = item_hash.find(song)->second;

			bool found = false;
			for (int token_i = 1; token_i < tokens.size(); token_i++) {
				const string track = tokens[token_i];
				if (_track_hash.find(track) != _track_hash.end()) {
					const int t = _track_hash.find(track)->second;
					assert(_item_to_embeddings[i] == nullptr);  // brand new;
					assert(t * _dimension < _embedding_allocation);
					_item_to_embeddings[i] = &_embeddings[t * _dimension];
						// direct pointer;
					matching_song_tracks_count++;
					break;  // found the first matching track;
				}
			}
		}
		line_number++;
	}

	cout << "found " << song_with_zero_track_count <<
		" songs with no tracks ..." << endl;
	cout << "found " << song_with_multiple_track_count <<
		" songs with multiple tracks ..." << endl;
	cout << "parsing embeddings of " << _dimension << " dimensions and "
		<< _track_count << " tracks ... (" << matching_song_tracks_count
		<< " matches) DONE" << endl;

	int embeddings_count = 0;
	for (int i = 0; i < item_count; i++) {
		if (this->embeddings(i) != nullptr) embeddings_count++;
	}
}

//////////////////////////////////////////////////////////////////////////
//
// class Latents -- a user or item concept;
//
//////////////////////////////////////////////////////////////////////////

class Latents {  // Latents 
	string _name;
	int _dimension;
	int _entity_count;
	int _allocation;
	float *_memory;
public:
	Latents();
	~Latents();
	void reset(const string &name, const string &latent_filename);
	const string &name() const { return _name; }
	int dimension() const { return _dimension; }
	int entity_count() const { return _entity_count; }
	float *memory() const { return _memory; }
	const float *latents(int e) const {
		assert(e >= 0 && e < _entity_count);
		return &_memory[e * _dimension];
	}
};

Latents::Latents() :
	_dimension(0), _entity_count(0), _allocation(0), _memory(nullptr) {
}

Latents::~Latents() {
	if (_memory) delete[] _memory;
	_allocation = 0;
}

void Latents::reset(const string &name, const string &latents_filename) {
	if (_memory != nullptr) delete[] _memory;
	_memory = nullptr;
	_allocation = 0;
	_name = name;
	_dimension = 0;
	_entity_count = 0;
	ifstream fp(latents_filename);
	assert(fp.is_open());
	string line;
	assert(getline(fp, line));  // read the header;
	stringstream ss(line);
	string token;

	int column_count = 0;
	while (getline(ss, token, ',')) {
		column_count++;
	}
	_entity_count = 0;
	while (getline(fp, line)) {
		_entity_count++;  // except the first line;
	}
	fp.close();

	_dimension = column_count - 1;
	assert(_dimension >= 0);
	_allocation = _dimension * _entity_count;
	_memory = new float[_allocation];
	fill(_memory, _memory + _allocation, 0);

	cout << "parsing latent of " << _dimension << " dimensions and "
		<< _entity_count << " entities" << endl;

	vector<string> tokens;
	fp.open(latents_filename);
	assert(getline(fp, line));  // skip the header;
	int entity_i = 0;
	int memory_i = 0;
	while (getline(fp, line)) {
		if ((entity_i + 1) % 10000 == 0)
			cout << _name << ": entity_i = " << (entity_i + 1) <<
				"/" << _entity_count << ", " << memory_i << endl;
		tokens.clear();
		stringstream ss(line);
		while (getline(ss, token, ',')) {
			tokens.push_back(token);
		}
		assert(tokens.size() == _dimension + 1);
		int entity = stoi(tokens[0]);
		assert(entity == entity_i);
		for (int t = 1; t < tokens.size(); t++) {
			assert(memory_i < _allocation);
			_memory[memory_i++] = stof(tokens[t]);
		}
		entity_i++;
	}
	assert(entity_i == _entity_count);
	assert(memory_i == _allocation);
}

//////////////////////////////////////////////////////////////////////////
//
// class Ensembler -- take a recommendation collection and combine them
//
//////////////////////////////////////////////////////////////////////////

class Ensembler {
public:
	static void ensemble(const Dataset &dataset,
		const RecommendationCollection &input_recommendation_collection,
		Recommendations &output_recommendations);
	static float ranks_to_score_v1(const vector<Score> &ranks,
		const vector<float> &weights);
	static float ranks_to_score_v2(const vector<Score> &ranks,
		const vector<float> &weights);
	static float ranks_to_score_v2_original(const vector<Score> &ranks,
		const vector<float> &weights);
	static float ranks_to_score(const vector<Score> &ranks,
		const vector<float> &weights) {
		return ranks_to_score_v1(ranks, weights);  // use v1, in the paper;
	}
};

float Ensembler::ranks_to_score_v1(const vector<Score> &ranks,
	const vector<float> &weights) {

	double highest_score = 0;
	for (int r = 0; r < ranks.size(); r++) {
		float rank = ranks[r].score;
		float weight = weights[ranks[r].index];
		float score = weight / (1 + rank);
		if (highest_score < score) highest_score = score;
	}
	return highest_score;
}

float Ensembler::ranks_to_score_v2(const vector<Score> &ranks,
	const vector<float> &weights) {

	// imagine being ranked is a signal for likelihood of being a
	// real deal, the goal is to compute the probability of the item
	// being a real deal, given the rankings;

	// the following are empirically determined;

	vector<float> coeff_item({  // 1-based coefficients;
		1.92132052e-20, -4.35633868e-17, 4.15333814e-14, -2.16592753e-11,
		6.73702353e-09, -1.28143879e-06, 1.47911755e-04, -1.02051970e-02,
		5.01992314e-01
	});

	vector<float> coeff_user({  // 1-based coefficients;
		1.59626060e-20, -3.63940725e-17, 3.49269826e-14, -1.83632902e-11,
		5.77327635e-09, -1.11463162e-06, 1.31528737e-04, -9.39713214e-03,
		4.99899384e-01
	});

	vector<float> coeff_popularity({  // 1-based coefficients;
		-5.19074863e-21, 1.09575133e-17, -9.49822877e-15, 4.33528925e-12,
		-1.10228042e-09, 1.47661514e-07, -7.17024764e-06, -4.90233022e-04,
		9.81425521e-02
	});

	auto fit = [&](float k, int r) -> float {
		assert(r >= 0 && r < 3);  // fixme, should retrieve the correct polynomial;
		vector<float> *coeff = r == 0 ?
			&coeff_item : (r == 1 ? &coeff_user : &coeff_popularity);
		float y = 0;
		for (auto c: *coeff) { y = y * k + c; }
		return y;
	};

	bool seen_coeffs[3];
	fill(seen_coeffs, seen_coeffs + 3, false);
	float pi = 1.0;  // see paper;
	for (auto r: ranks) {
		assert(r.index >= 0 && r.index < 3);
			// this routine needs to be upgraded to be more usable;
		seen_coeffs[r.index] = true;
		float probability = fit(r.score + 1, r.index);  // 0-based to 1-based;
		pi *= 1 - probability;
	}
	for (int c = 0; c < 3; c++) {
		if (!seen_coeffs[c]) {
			float probability = fit(Global::recK, c);  // 0-based to 1-based;
			pi *= 1 - probability;
		}
	}

	const float probability = 1 - pi;
	return probability;
}

float Ensembler::ranks_to_score_v2_original(const vector<Score> &ranks,
	const vector<float> &weights) {

	// imagine being ranked is a signal for likelihood of being a
	// real deal, the goal is to compute the probability of the item
	// being a real deal, given the rankings;

	const float unranked_probability = 0.05;
	const float min_probability = 0.1;
	const float max_probability = 0.5;
	float total_weights = 0.0;
	for (auto weight: weights) total_weights += weight;
	float cumu = 1.0;
	const float invRecK = 1.0 / Global::recK;
	float sum_weights = 0;
	for (auto r: ranks) {
		float probability = max_probability -
			r.score * invRecK * (max_probability - min_probability);
				// probability that this is the real deal,
				// linearly interpreted from max_probaility when
				// rank = 0 to min_probability when rank = RecK;
		float weight = weights[r.index];
		cumu *= pow(1 - probability, weight);
		sum_weights += weight;
	}
	const float remaining_weights = total_weights - sum_weights;
	cumu *= pow(1 - unranked_probability, remaining_weights);
	
	return 1 - cumu;
}

void Ensembler::ensemble(const Dataset &dataset,
	const RecommendationCollection &input_recommendation_collection,
	Recommendations &output_recommendations) {

	const int recommender_count = input_recommendation_collection.size();
	vector<float> input_weights;
	for (int index = 0; index < recommender_count; index++) {
		const float weight = input_recommendation_collection[index].second;
		input_weights.push_back(
			pow(weight, Global::ensemble_input_multiplier));
	}

	cout << "ensemble weights: ";
	for (auto w: input_weights) cout << w << ",";
	cout << endl;

	const int item_count = dataset.item_count();
	const int recK = Global::recK;
	assert(recommender_count > 0);
	float total_weights = 0;
	for (auto weight: input_weights) total_weights += weight;

	const float unrankedP = 0.05;
	const float minP = 0.1;
	const float maxP = 0.5;

	cout << "running ensemble with " << recommender_count <<
		 " recommendations ..." << endl;

	typedef tuple<float, int, int> SRI;  // <score, r, i>
	vector<SRI> recs;
	vector<Score> ranks;  // <score, r> pairs;

	vector<int> rec_users = dataset.rec_users();
	for (auto u: rec_users) {
		recs.clear();
		for (int index = 0; index < recommender_count; index++) {
			const auto &r = *input_recommendation_collection[index].first;
			for (int k = 0; k < recK; k++) {
				const int i = r.get(u, k).index;
					// recommending this item at rank k;
				recs.push_back(make_tuple(k, index, i));  // k is the rank;
			}
		}
		sort(recs.begin(), recs.end(), [](const SRI &a, const SRI &b) {
			return get<2>(a) < get<2>(b);  // sort by item;
		});
		int merged_i = 0;
		int index = 0;
		while (index < recs.size()) {
			int i = get<2>(recs[index]);
			int index_end = index + 1;
			while (index_end < recs.size() && get<2>(recs[index_end]) == i)
				index_end++;
					// at the end, [index, index_end) all point to i;
			ranks.clear();
			for (int ii = index; ii < index_end; ii++)
				ranks.push_back(Score(get<0>(recs[ii]), get<1>(recs[ii])));
			const float score = Ensembler::ranks_to_score(
				ranks, input_weights);  // higher score is better;

			recs[merged_i++] = make_tuple(score, INT_MAX, i);  
			index = index_end;
		}
		recs.resize(merged_i);  // combine multiple ratings;
		assert(recs.size() >= recK);  // invariant;
		sort(recs.begin(), recs.end(), [](const SRI &a, const SRI &b) {
			return get<0>(a) > get<0>(b);  // sort by score (higher is better);
		});
		for (int k = 0; k < recK; k++)
			output_recommendations.set(u, k, Score(get<0>(recs[k]), get<2>(recs[k])));
	}

	cout << "running ensemble with " << recommender_count <<
		 " recommendations ... DONE" << endl;
}

//////////////////////////////////////////////////////////////////////////
//
// class SimilarityDebugger
//
//////////////////////////////////////////////////////////////////////////

class SimilarityDebugger {
	enum { Resolution = 20 };
	vector<float> _stat;
	int _count;
	const int _frequency;
public:
	SimilarityDebugger(int frequency = 100) :
		_count(0), _frequency(100), _stat(Resolution, 0) {}
	~SimilarityDebugger() {}
	void update(float *values) { // values must be an aarray of size Resolution;
		for (int i = 0; i < Resolution; i++)
			_stat[i] += values[i];
	}
};

void similarity_item_based_v4(const Dataset *p_dataset,
	const int topK, const int max_batch_size,
	Jobs *p_jobs, const int workerId,
	Similarities *p_topK_similarities, const bool silent) {

	Similarities &topK_similarities = *p_topK_similarities;
	const Dataset &dataset = *p_dataset;
	const vector<UIR> &uirs_uv = dataset.uirs_uv();
	const vector<UIR> &uirs_iv = dataset.uirs_iv();
	const vector<Range> &ranges_uv = dataset.ranges_uv();
	const vector<Range> &ranges_iv = dataset.ranges_iv();
	const int item_count = dataset.item_count();
	const int N = dataset.N();
	auto &jobs = *p_jobs;
	vector<Score> sortable(item_count);
	const vector<int> &item_clusters = dataset.item_clusters();
	const bool need_clusters = item_clusters.size() > 0;
	if (need_clusters)
		assert(item_clusters.size() == item_count);  // consistency;

	const int memory_size = item_count * max_batch_size;
	vector<int> memory(memory_size, 0);

	if (!silent) {
		cout_mutex.lock();
		cout << "running similarity (item-based) on worker " <<
			workerId << " ..." << endl;
		cout_mutex.unlock();
	}
	
	int i_from, i_to;
	while (jobs.next(i_from, i_to)) {
		assert(i_to - i_from <= max_batch_size);
		fill(memory.begin(), memory.end(), 0);

		// Part (a) of the algorithm is as follows.
		//
		// foreach i in [i_from, i_to):
		//     foreach u who rated i:
		//         foreach j rated by u:
		//             memory(ij)++;

		for (int i = i_from; i < i_to; i++) {
			const int from_iv = ranges_iv[i].first;
			const int to_iv = ranges_iv[i].second;
			// cout << "working on " << i << " with " << (to_iv - from_iv) <<
			// 	" items" << endl;
			for (int index_iv = from_iv; index_iv < to_iv;
					index_iv++) {
				const int u = get<0>(uirs_iv[index_iv]);
				const int from_uv = ranges_uv[u].first;
				const int to_uv = ranges_uv[u].second;
				for (int index_uv = from_uv; index_uv < to_uv;
						index_uv++) {
					const int j = get<1>(uirs_uv[index_uv]);
					const int ij = (i - i_from) * item_count + j;
					assert(ij < memory_size);
					memory[ij]++;
				}
			}
		}

		// Part (b) of the algorithm is as follows.
		//
		// foreach i in [i_from, i_to) do:
		//     foreach j do:
		//         calculate sim(ij) using memory(ij);
		//     keep the topK sim(ij) neighbors (using nth, then sort);

		const bool Jaccard = Global::Jaccard;
		const float alpha = Global::alpha;
		int total_sortable_count = 0;
		int total_topK_count = 0;
		for (int i = i_from; i < i_to; i++) {
			const int ij_0 = (i - i_from) * item_count;
			int sortable_count = 0;
			const int i_count = ranges_iv[i].second - ranges_iv[i].first;
			const float i_denominator = pow(i_count, alpha);
			const int cluster_id_i =
				need_clusters ? item_clusters[i] : INT_MAX;
			for (int j = 0, ij = ij_0; j < item_count; j++, ij++) {
				const int ij_count = memory[ij];
				if (ij_count > 0) {
					const int j_count = ranges_iv[j].second - ranges_iv[j].first;
					const float j_denominator = pow(j_count, 1 - alpha);
					float sim_ij = Jaccard ?
						float(ij_count) / (i_count + j_count - ij_count) :
						float(ij_count) / (i_denominator * j_denominator);
					const int cluster_id_j = 
						need_clusters ? item_clusters[j] : INT_MAX;
					if (need_clusters && cluster_id_i == cluster_id_j)
						sim_ij *= 1.2;
					sortable[sortable_count++] = Score(sim_ij, j);
				}
			}
			int topK_i = topK;
			if (sortable_count > topK) {
				nth_element(
					sortable.begin(),
					sortable.begin() + topK,
					sortable.begin() + sortable_count);
			} else {
				topK_i = sortable_count;  // not enough entries;
			}

			sort(sortable.begin(), sortable.begin() + topK_i);
			total_topK_count += topK_i;
			total_sortable_count += sortable_count;

			for (int k = 0; k < topK_i; k++) {
				topK_similarities.set(i, k, sortable[k]);
			}
			for (int k = topK_i; k < topK; k++) {
				topK_similarities.set(i, k, Score(0, INT_MAX));
			}
		}
	}
}

void similarity_user_based_v4(const Dataset *p_dataset,
	const int topK, const int max_batch_size,
	Jobs *p_jobs, const int workerId,
	Similarities *p_topK_similarities,
	const vector<int> &rec_users, const bool silent) {

	// note that we are going through the users in non-contiguous order;

	Similarities &topK_similarities = *p_topK_similarities;
	const Dataset &dataset = *p_dataset;
	const vector<UIR> &uirs_uv = dataset.uirs_uv();
	const vector<UIR> &uirs_iv = dataset.uirs_iv();
	const vector<Range> &ranges_uv = dataset.ranges_uv();
	const vector<Range> &ranges_iv = dataset.ranges_iv();
	const int user_count = dataset.user_count();
	const int N = dataset.N();
	auto &jobs = *p_jobs;
	vector<Score> sortable(user_count);

	const int memory_size = user_count * max_batch_size;
	vector<int> memory(memory_size, 0);

	if (!silent) {
		cout_mutex.lock();
		cout << "running similarity (user-based) on worker " <<
			workerId << " ..." << endl;
		cout_mutex.unlock();
	}
	
	int rec_from, rec_to;
	while (jobs.next(rec_from, rec_to)) {
		assert(rec_to - rec_from <= max_batch_size);
		fill(memory.begin(), memory.end(), 0);

		// Part (a) of the algorithm is as follows.
		//
		// foreach u in [u_from, u_to):
		//     foreach i rated by u:
		//         foreach v who rated i:
		//             memory(uv)++;

		for (int rec = rec_from; rec < rec_to; rec++) {
			const int u = rec_users[rec];
				// note the users are not arranged in contiguous block;

			const int from_uv = ranges_uv[u].first;
			const int to_uv = ranges_uv[u].second;
			cout_mutex.lock();
			// cout << "working on " << u << " with " << (to_uv - from_uv) <<
			// 	" users" << endl;
			cout_mutex.unlock();
			for (int index_uv = from_uv; index_uv < to_uv;
					index_uv++) {
				const int i = get<1>(uirs_uv[index_uv]);
				const int from_iv = ranges_iv[i].first;
				const int to_iv = ranges_iv[i].second;
				for (int index_iv = from_iv; index_iv < to_iv;
						index_iv++) {
					const int v = get<0>(uirs_iv[index_iv]);
					const int rv = (rec - rec_from) * user_count + v;
					assert(rv < memory_size);
					memory[rv]++;
				}
			}
		}

		// Part (b) of the algorithm is as follows.
		//
		// foreach u in [u_from, u_to) do:
		//     foreach v do:
		//         calculate sim(uv) using memory(rv);
		//     keep the topK sim(uv) neighbors (using nth, then sort);

		const bool Jaccard = Global::Jaccard;
		const float alpha = Global::alpha;
		int total_sortable_count = 0;
		int total_topK_count = 0;
		for (int rec = rec_from; rec < rec_to; rec++) {
			const int u = rec_users[rec];
				// note the users are not arranged in contiguous block;
			
			const int rv_0 = (rec - rec_from) * user_count;
			int sortable_count = 0;
			const int u_count = ranges_uv[u].second - ranges_uv[u].first;
			const float u_denominator = pow(u_count, alpha);
			for (int v = 0, rv = rv_0; v < user_count; v++, rv++) {
				const int uv_count = memory[rv];
				if (uv_count > 0) {
					const int v_count = ranges_uv[v].second - ranges_uv[v].first;
					const float v_denominator = pow(v_count, 1 - alpha);
					float sim_uv = Jaccard ?
						float(uv_count) / (u_count + v_count - uv_count) :
						float(uv_count) / (u_denominator * v_denominator);
					sortable[sortable_count++] = Score(sim_uv, v);
				}
			}
			int topK_u = topK;
			if (sortable_count > topK) {
				nth_element(
					sortable.begin(),
					sortable.begin() + topK,
					sortable.begin() + sortable_count);
			} else {
				topK_u = sortable_count;  // not enough entries;
			}

			sort(sortable.begin(), sortable.begin() + topK_u);

			total_topK_count += topK_u;
			total_sortable_count += sortable_count;

			for (int k = 0; k < topK_u; k++) {
				topK_similarities.set(u, k, sortable[k]);
			}
			for (int k = topK_u; k < topK; k++) {
				topK_similarities.set(u, k, Score(0, INT_MAX));
			}
		}
	}
}

void recommend_v4_neighborhood_based(
	Jobs *p_jobs, const int workerId, Algo algo, const Dataset *p_dataset,
	const int topK,
	Similarities *p_topK_similarities,
	const vector<int> *p_rec_users,
	Recommendations *p_recommendations, Evaluator *p_evaluator,
	const bool silent) {

	// rec_users is a subset of the users to receive
	// recommendations;

	const Dataset &dataset = *p_dataset;
	Recommendations &recommendations = *p_recommendations;
	Evaluator &evaluator = *p_evaluator;
	const vector<int> &rec_users = *p_rec_users;
	const int item_count = dataset.item_count();
	assert(Global::recK < item_count);
	const vector<Range> &ranges_uv = dataset.ranges_uv();
	const vector<Range> &ranges_iv = dataset.ranges_iv();
	Jobs &jobs = *p_jobs;
	vector<int> current_rec_users;
	const Similarities &topK_similarities = *p_topK_similarities;
	const float q = Global::q;

	assert(algo == Algo::ItemBased || algo == Algo::UserBased);
	vector<Score> scores(item_count);
	const vector<UIR> &uirs_uv = dataset.uirs_uv();
	const float LargeNegative = -1e9;
	const bool zero_one = Global::zero_one;

	int rec_users_from, rec_users_to;
	while (jobs.next(rec_users_from, rec_users_to)) {

		current_rec_users.clear();
		for (int index = rec_users_from; index < rec_users_to; index++) {
			current_rec_users.push_back(rec_users[index]);
		}

		for (auto u: current_rec_users) {
			for (int index = 0; index < item_count; index++) {
				scores[index] = Score(0, index);  // setup;
			}
			const int from_uv = ranges_uv[u].first;
			const int to_uv = ranges_uv[u].second;
			for (int index_uv = from_uv; index_uv < to_uv; index_uv++) {
				const int i = get<1>(uirs_uv[index_uv]);
				scores[i].score = LargeNegative;  // skip existing songs;
			}
			if (algo == Algo::ItemBased) {
				for (int index_uv = from_uv; index_uv < to_uv; index_uv++) {
					const int i = get<1>(uirs_uv[index_uv]);
					for (int k = 0; k < topK; k++) {
						Score sim_k = topK_similarities.get(i, k);
						const int j = sim_k.index;
						if (j == INT_MAX) break;  // no more;
						if (scores[j].score < 0) continue;  // existing item; 
						const float score = sim_k.score;
						float play = get<2>(uirs_uv[index_uv]);
						if (zero_one) play = play > 0 ? 1 : 0;
						scores[j].score += pow(score, q) * play;
							// item j is a topK of item i; 
					}
				}
			} else {
				for (int k = 0; k < topK; k++) {
					Score sim_k = topK_similarities.get(u, k);
					const int v = sim_k.index;
					if (v == INT_MAX) break;  // no more;
					const float score = sim_k.score;
					const int from_vv = ranges_uv[v].first;
					const int to_vv = ranges_uv[v].second;
					for (int index_vv = from_vv; index_vv < to_vv; index_vv++) {
						const int i = get<1>(uirs_uv[index_vv]);
						if (scores[i].score < 0) continue;  // existing item; 
						float play = get<2>(uirs_uv[index_vv]);
						if (zero_one) play = play > 0 ? 1 : 0;
						scores[i].score += pow(score, q) * play;
							// item i is rated by a topK user v;
					}
				}
			}
		
			nth_element(
				scores.begin(), scores.begin() + Global::recK, scores.end());
			sort(scores.begin(), scores.begin() + Global::recK);
			for (int k = 0; k < Global::recK; k++) {
				recommendations.set(u, k, scores[k]);
			}
		}

		evaluator.evaluate(current_rec_users, recommendations, silent);
	}
}

void recommend_v4_embedding_based(
	Jobs *p_jobs, const int workerId, Algo algo, const Dataset *p_dataset,
	const Embeddings *p_embeddings,
	const vector<int> *p_rec_users,
	Recommendations *p_recommendations, Evaluator *p_evaluator,
	const bool silent) {

	// rec_users is a subset of the users to receive
	// recommendations;

	const Embeddings &embeddings = *p_embeddings;
	Recommendations &recommendations = *p_recommendations;
	Evaluator &evaluator = *p_evaluator;
	const int dimension = embeddings.dimension();
	const Dataset &dataset = *p_dataset;
	const vector<int> &rec_users = *p_rec_users;
	const int item_count = dataset.item_count();
	assert(Global::recK < item_count);
	const vector<Range> &ranges_uv = dataset.ranges_uv();
	Jobs &jobs = *p_jobs;
	vector<int> current_rec_users;

	vector<Score> scores(item_count);
	const vector<UIR> &uirs_uv = dataset.uirs_uv();
	const float LargeNegative = -1e9;
	const bool zero_one = Global::zero_one;
	const float q = Global::q;

	int rec_users_from, rec_users_to;
	while (jobs.next(rec_users_from, rec_users_to)) {

		cout_mutex.lock();
		cout << "working on [" << rec_users_from << ", " <<
			rec_users_to << "] (worker " << workerId << ")" << endl;
		cout_mutex.unlock();

		current_rec_users.clear();
		for (int index = rec_users_from; index < rec_users_to; index++) {
			current_rec_users.push_back(rec_users[index]);
		}

		for (auto u: current_rec_users) {
			for (int j = 0; j < item_count; j++) {
				scores[j] = Score(0, j);  // cleanup;
			}
			const int from_uv = ranges_uv[u].first;
			const int to_uv = ranges_uv[u].second;
			for (int index_uv = from_uv; index_uv < to_uv; index_uv++) {
				const int i = get<1>(uirs_uv[index_uv]);
				scores[i].score = LargeNegative;  // skip existing songs;
			}
			for (int index_uv = from_uv; index_uv < to_uv; index_uv++) {
				const int i = get<1>(uirs_uv[index_uv]);
				float *i_embeddings = embeddings.embeddings(i);
				if (i_embeddings == nullptr) continue;  // no embedding;
				for (int j = 0; j < item_count; j++) {
					if (scores[j].score < 0) continue;  // existing item; 
					float *j_embeddings = embeddings.embeddings(j);
					if (j_embeddings == nullptr) continue;  // no embedding;
					float play = get<2>(uirs_uv[index_uv]);
					const float score = Embeddings::similarity(
						i_embeddings, j_embeddings, dimension);
					if (isnan(score)) {
						cout << "i = " << i << ", j = " << j << endl;
					}
					if (zero_one) play = play > 0 ? 1 : 0;
					scores[j].score += pow(score, q) * play;
						// item j is a topK of item i; 
				}
			}
	
			nth_element(
				scores.begin(), scores.begin() + Global::recK, scores.end());
			sort(scores.begin(), scores.begin() + Global::recK);
			for (int k = 0; k < Global::recK; k++) {
				recommendations.set(u, k, scores[k]);
			}
		}

		evaluator.evaluate(current_rec_users, recommendations, silent);
	}
}

void recommend_v4_latents_based(
	Jobs *p_jobs, const int workerId, Algo algo, const Dataset *p_dataset,
	const Latents *p_user_latents, const Latents *p_item_latents,
	const vector<int> *p_rec_users,
	Recommendations *p_recommendations, Evaluator *p_evaluator,
	const bool silent) {

	// rec_users is a subset of the users to receive
	// recommendations;

	const Latents &user_latents = *p_user_latents;
	const Latents &item_latents = *p_item_latents;
	Recommendations &recommendations = *p_recommendations;
	Evaluator &evaluator = *p_evaluator;
	const int dimension = user_latents.dimension();
	assert(dimension == item_latents.dimension());
	const Dataset &dataset = *p_dataset;
	const vector<int> &rec_users = *p_rec_users;
	const int item_count = dataset.item_count();
	assert(Global::recK < item_count);
	const vector<Range> &ranges_uv = dataset.ranges_uv();
	Jobs &jobs = *p_jobs;
	vector<int> current_rec_users;

	vector<Score> scores(item_count);
	const vector<UIR> &uirs_uv = dataset.uirs_uv();
	const float LargeNegative = -1e9;

	int rec_users_from, rec_users_to;
	while (jobs.next(rec_users_from, rec_users_to)) {

		cout_mutex.lock();
		cout << "working on [" << rec_users_from << ", " <<
			rec_users_to << "] (worker " << workerId << ")" << endl;
		cout_mutex.unlock();

		current_rec_users.clear();
		for (int index = rec_users_from; index < rec_users_to; index++) {
			current_rec_users.push_back(rec_users[index]);
		}

		for (auto u: current_rec_users) {
			const float *u_latents = user_latents.latents(u);
			for (int j = 0; j < item_count; j++) {
				scores[j] = Score(0, j);  // cleanup;
			}
			const int from_uv = ranges_uv[u].first;
			const int to_uv = ranges_uv[u].second;
			for (int index_uv = from_uv; index_uv < to_uv; index_uv++) {
				const int i = get<1>(uirs_uv[index_uv]);
				scores[i].score = LargeNegative;  // skip existing songs;
			}
			for (int i = 0; i < item_count; i++) {
				if (scores[i].score < 0) continue;  // existing item;
				const float *i_latents = item_latents.latents(i);
				float inner = 0;
				auto uu = u_latents, ii = i_latents;
				for (int d = 0; d < dimension; d++)
					inner += *uu++ * *ii++;
				scores[i].score = inner;
			}
			nth_element(
				scores.begin(), scores.begin() + Global::recK, scores.end());
			sort(scores.begin(), scores.begin() + Global::recK);
			for (int k = 0; k < Global::recK; k++) {
				recommendations.set(u, k, scores[k]);
			}
		}

		evaluator.evaluate(current_rec_users, recommendations, silent);
	}
}

void recommend_v4_popularity_based(
	Jobs *p_jobs, const int workerId, Algo algo, const Dataset *p_dataset,
	const vector<int> *p_rec_users,
	Recommendations *p_recommendations, Evaluator *p_evaluator,
	const bool silent) {

	// rec_users is a subset of the users to receive
	// recommendations;

	assert(algo == Algo::Popularity || algo == Algo::Randomized);
	const Dataset &dataset = *p_dataset;
	Recommendations &recommendations = *p_recommendations;
	Evaluator &evaluator = *p_evaluator;
	const vector<int> &rec_users = *p_rec_users;
	const int item_count = dataset.item_count();
	assert(Global::recK < item_count);
	const vector<Range> &ranges_uv = dataset.ranges_uv();
	const vector<Range> &ranges_iv = dataset.ranges_iv();
	Jobs &jobs = *p_jobs;

	vector<Score> popular_items(item_count);
	for (int i = 0; i < item_count; i++) {
		popular_items[i] = Score(ranges_iv[i].second - ranges_iv[i].first, i);
	}
	if (algo == Algo::Popularity) {
		sort(popular_items.begin(), popular_items.end());
	} else if (algo == Algo::Randomized) {
		shuffle(popular_items.begin(), popular_items.end(),
			default_random_engine(42));
	} else {
		assert(0);
	}

	vector<Score> scores(item_count);
	const vector<UIR> &uirs_uv = dataset.uirs_uv();
	const float LargeNegative = -1e9;

	vector<int> current_rec_users;
	vector<int> existing_items;
	int rec_users_from, rec_users_to;
	while (jobs.next(rec_users_from, rec_users_to)) {

		current_rec_users.clear();
		for (int index = rec_users_from; index < rec_users_to; index++) {
			current_rec_users.push_back(rec_users[index]);
		}

		for (auto u: current_rec_users) {
			const int from_uv = ranges_uv[u].first;
			const int to_uv = ranges_uv[u].second;
			existing_items.clear();
			for (int index_uv = from_uv; index_uv < to_uv; index_uv++) {
				const int i = get<1>(uirs_uv[index_uv]);
				existing_items.push_back(i);
			}
			sort(existing_items.begin(), existing_items.end());

			int k = 0;
			for (auto popular_i: popular_items) {
				if (binary_search(
						existing_items.begin(), existing_items.end(),
						popular_i.index)) {
					continue;  // don't recommend existing item;
				}
				recommendations.set(u, k, popular_i);
				k++;
				if (k == Global::recK) break;  // all done;
			}
			assert(k == Global::recK);  // must have recommended enough items; 
		}

		evaluator.evaluate(current_rec_users, recommendations, silent);
	}
}

void ensemble_v4(const Dataset &dataset,
	const RecommendationCollection &input_recommendation_collection,
	Recommendations &output_recommendations, Evaluator &evaluator,
	const bool silent) {

	Ensembler::ensemble(dataset,
		input_recommendation_collection, output_recommendations);

	auto rec_users = dataset.rec_users();
	evaluator.evaluate(rec_users, output_recommendations, silent);
}

void run_v4(const Dataset &dataset, Algo algo,
	const int topK, const int thread_count,
	const Embeddings &embeddings,
	Recommendations &recommendations,
	Evaluator &evaluator, const bool silent) {
	
	// (3) going through each user, and each item of the user,
	//	 retrieve all other users rating the same item;

	Latents user_latents, item_latents;
	bool use_lmf_latents = algo == Algo::LmfLatents;
	bool use_als_latents = algo == Algo::AlsLatents;
	if (use_lmf_latents || use_als_latents) {
		if (use_lmf_latents) {
			user_latents.reset("lmf_user", "lmf_user_factors.csv");
			item_latents.reset("lmf_item", "lmf_item_factors.csv");
		} else {
			user_latents.reset("als_user", "als_user_factors.csv");
			item_latents.reset("als_item", "als_item_factors.csv");
		}
		assert(user_latents.entity_count() == dataset.user_count());
		assert(item_latents.entity_count() == dataset.item_count());
	}

	auto make_windows = [](int N, int batch_size) -> vector<pair<int, int> > {
		vector<pair<int, int> > windows;
		int i = 0;
		while (i < N) {
			const int i_end = min(N, i + batch_size);
			windows.push_back(make_pair(i, i_end));
			i = i_end;
		}
		return windows;
    };

	vector<int> rec_users = dataset.rec_users();

	Similarities topK_similarities;
	const bool item_based = algo == Algo::ItemBased;
	const bool user_based = algo == Algo::UserBased;
	const bool neighborhood_based = item_based || user_based;
	if (neighborhood_based) {
		const int batch_size = item_based ? 128 : 32;  // smaller batch size for user-based;
		Jobs similarity_jobs;
		if (item_based) {
			auto windows = make_windows(dataset.item_count(), batch_size);
			similarity_jobs.reset(windows);
			const int item_count = dataset.item_count();
			vector<int> items(item_count, 0);
			for (int i = 0; i < item_count; i++) items[i] = i;
			topK_similarities.reset(items, topK);  // all items;
		} else {
			auto windows = make_windows(rec_users.size(), batch_size);
			similarity_jobs.reset(windows);
			topK_similarities.reset(rec_users, topK);  // rec_users only;
		}
		vector<thread> threads(thread_count);
		for (int i = 0; i < thread_count; i++) {
			if (item_based) {
				threads[i] = thread(similarity_item_based_v4,
					&dataset, topK, batch_size, &similarity_jobs,
					i, &topK_similarities, silent);
			} else {
				threads[i] = thread(similarity_user_based_v4,
					&dataset, topK, batch_size, &similarity_jobs,
					i, &topK_similarities, rec_users, silent);
			}
		}
		for (int i = 0; i < thread_count; i++) {
			threads[i].join();
		}
	}

	/////////////////////////////////////////////////////////////////

	if (!silent) {
		cout << "making recommendations for " << rec_users.size() <<
			" users ..." << endl;
	}
	const int rec_batch_count = rec_users.size() / (25 * thread_count);
				// average 25 jobs per thread;
	int i = 0;
	vector<PII> rec_windows;
	while (i < rec_users.size()) {
		const int i_end = min(i + rec_batch_count, rec_users.size());
		rec_windows.push_back(make_pair(i, i_end));
		i = i_end;
	}
	Jobs rec_jobs(rec_windows);
	
	const int rec_thread_count = thread_count;
	vector<thread> rec_threads(rec_thread_count);

	for (int i = 0; i < rec_thread_count; i++) {
		if (algo == Algo::Popularity || algo == Algo::Randomized) {
			rec_threads[i] = thread(recommend_v4_popularity_based,
				&rec_jobs, i, algo, &dataset,
				&rec_users, &recommendations, &evaluator, silent);
		} else if (algo == Algo::EmbeddingBased) {
			rec_threads[i] = thread(recommend_v4_embedding_based,
				&rec_jobs, i, algo, &dataset,
				&embeddings,
				&rec_users, &recommendations, &evaluator, silent);
		} else if (algo == Algo::AlsLatents || algo == Algo::LmfLatents) {
			rec_threads[i] = thread(recommend_v4_latents_based,
				&rec_jobs, i, algo, &dataset,
				&user_latents, &item_latents,
				&rec_users, &recommendations, &evaluator, silent);
		} else if (algo == Algo::ItemBased || algo == Algo::UserBased) {
			rec_threads[i] = thread(recommend_v4_neighborhood_based,
				&rec_jobs, i, algo, &dataset,
				topK, &topK_similarities,
				&rec_users, &recommendations, &evaluator, silent);
		}
	}
	for (int i = 0; i < rec_thread_count; i++) {
		rec_threads[i].join();
	}
}

//////////////////////////////////////////////////////////////////////////

void run_all_v4(const string &name, const Dataset &dataset,
	const string &algo_name,
	const Embeddings &embeddings,
	const RecommendationCollection &recommendation_collection,
	const bool silent) {

	Evaluator evaluator(dataset);
	Recommendations recommendations(dataset);
	Algo algo = algo_hash[algo_name];

	cout << "run(" << name << "): <" << algo_name << ">, alpha = " <<
		Global::alpha << ", q = " << Global::q << ", zero_one(" <<
		(Global::zero_one ? "true" : "false") << ")" << endl;

	const auto cpu_time_before = clock();
	const auto wall_time_before = chrono::high_resolution_clock::now();
	if (algo == Algo::Ensemble) {
		ensemble_v4(dataset, recommendation_collection,
			recommendations, evaluator, silent);
	} else {
		run_v4(dataset, algo, 500, 8, embeddings,
			recommendations, evaluator, silent);
	}
	const auto cpu_time_after = clock();
	const auto wall_time_after = chrono::high_resolution_clock::now();
	const int cpu_elapsed_time =
		int((cpu_time_after - cpu_time_before) / CLOCKS_PER_SEC);
    const auto wall_elapsed_time =
		chrono::duration_cast<chrono::nanoseconds>(
			wall_time_after - wall_time_before).count() * 1e-9;
	cout << "elapsed time: wall = " << int(wall_elapsed_time) << "s, " <<
		"cpu = " << int(cpu_elapsed_time) << "s" << endl;

	float mAP_valid = get<0>(evaluator.mAP(Dataset::UserType::Valid));
	float mAP_test = get<0>(evaluator.mAP(Dataset::UserType::Test));
	cout << "summary(" << name << "): <" << algo_name << ">, alpha = " <<
		Global::alpha << ", q = " << Global::q << ", zero_one(" <<
		(Global::zero_one ? "true" : "false") << ")" << endl;
	cout << "summary(" << name << "): valid = " << mAP_valid <<
		", test = " << mAP_test << endl;

	// cout << evaluator.precision_string() << endl;

	if (Global::output_recommendations) {
		auto t = std::time(nullptr);
		auto tm = *std::localtime(&t);
		ostringstream oss;
		oss << "rec_" << put_time(&tm, "%m%d_%H%M") << "_" <<
			algo_name << "_" << int(100 * Global::alpha) <<
			"_" << int(100 * Global::q) <<
			(Global::zero_one ? "_b" : "_r") << "_" <<
			int(1000000 * mAP_valid) << ".csv";
    	string filename = oss.str();
		dataset.write_recommendations(filename, recommendations);
	}
}

void gridsearch_v4(const Dataset &dataset,
	const string &algo_name, Embeddings &embeddings,
	const RecommendationCollection &input_recommendation_collection,
	const vector<string> &parameters, const vector<vector<float> > &values) {

	Global::output_recommendations = false;  // don't output anything;
	const int parameter_count = parameters.size();
	vector<int> indices(parameter_count, 0);
	vector<int> index_ends(parameter_count, 0);
	
	int cv_count = 1;
	for (int p = 0; p < parameter_count; p++) {
		index_ends[p] = values[p].size();
		cv_count *= index_ends[p];
	}

	int run_id = 0;
	for (int cv = 0; cv < cv_count; cv++) {

		// (1) set parameters;

		for (int p = 0; p < parameter_count; p++) {
			const string &parameter = parameters[p];
			const float value = values[p][indices[p]];
			if (parameter == "q") {
				Global::q = value;
			} else if (parameter == "zero_one") {
				Global::zero_one = bool(value);
			} else if (parameter == "alpha") {
				Global::alpha = value;
			} else if (parameter == "jaccard") {
				Global::Jaccard = bool(value);
			} else if (parameter == "ensemble_input_multiplier") {
				Global::ensemble_input_multiplier = value;
			}
		}

		// (2) run;

		stringstream run_name_ss;
		run_name_ss << "cv-search-" << to_string(run_id++);
		for (int p = 0; p < parameter_count; p++) {
			run_name_ss << "-" << parameters[p] << "(" <<
				std::fixed << std::setprecision(2) << values[p][indices[p]] << ")";
		}
		const string run_name = run_name_ss.str();
		run_all_v4(run_name, dataset, algo_name,
			embeddings, input_recommendation_collection, true);

		// (2) increment index;

		for (int p = 0; p < parameter_count; p++) {
			++indices[p];
			if (indices[p] < index_ends[p]) {
				break;
			} else {
				indices[p] = 0;  // and carry;
			}
		}
	}
}

int main(int argc, char **argv) {

	vector<string> visible_utility_filenames;
	vector<string> hidden_utility_filenames;
	string songs_filename = "";
	string test_users_filename = "";

	float train_test_split = 0.0;  // default to no split;
	string song_tracks_filename = "";
	string output_canon_train = "";
	string algo_name = "popularity";

	if (argc == 2) {
		ifstream fp_json(argv[1]);
		stringstream ss;
		ss << fp_json.rdbuf();
		string s = ss.str();
		auto js = nlohmann::json::parse(s);
		train_test_split = js["train_test_split"].get<float>();
		songs_filename = js["songs"].get<string>();
		test_users_filename = js["users"].get<string>();
		song_tracks_filename = js["song_tracks"].get<string>();
		Global::recommend_test_users = js["recommend_test_users"].get<bool>();
		Global::alpha = js["alpha"].get<float>();
		Global::q = js["q"].get<float>();
		Global::zero_one = js["zero_one"].get<bool>();
		Global::Jaccard = js["jaccard"].get<bool>();

		output_canon_train = js["output_canon_train"].get<string>();
		if (output_canon_train != "" && output_canon_train[0] == '#')
			 output_canon_train = "";  // commented out;
		Global::output_recommendations =
			js["output_recommendations"].get<bool>();

		algo_name = js["algorithm"].get<string>();
		transform(algo_name.begin(), algo_name.end(), algo_name.begin(),
			[](unsigned char c){ return tolower(c); });  // tolower();
		assert(algo_hash.find(algo_name) != algo_hash.end());
		cout << "train_test_split = " << train_test_split << endl;
		cout << "q = " << Global::q << endl;

		auto visible_utilities = js["visible_utilities"];
		visible_utility_filenames.clear();
		for (int i = 0; i < js["visible_utilities"].size(); i++) {
			auto visible_utility = js["visible_utilities"][i].get<string>();
			visible_utility_filenames.push_back(visible_utility);
			cout << "visible_utility = " << visible_utility << endl;
		}
		auto hidden_utilities = js["hidden_utilities"];
		hidden_utility_filenames.clear();
		for (int i = 0; i < js["hidden_utilities"].size(); i++) {
			auto hidden_utility = js["hidden_utilities"][i].get<string>();
			hidden_utility_filenames.push_back(hidden_utility);
			cout << "hidden_utility = " << hidden_utility << endl;
		}
		auto ensemble_inputs = js["ensemble_inputs"];
		Global::ensemble_inputs.clear();
		assert(js["ensemble_inputs"].size() % 2 == 0);  // even number;
		for (int i = 0; i < js["ensemble_inputs"].size(); i += 2) {
			// filename, weight pair;
			auto filename = js["ensemble_inputs"][i].get<string>();
			auto weight = js["ensemble_inputs"][i + 1].get<float>();
			Global::ensemble_inputs.push_back(make_pair(filename, weight));
			cout << "ensemble_input = " << filename << ", " << weight << endl;
		}
		Global::ensemble_input_multiplier =
			js["ensemble_input_multiplier"].get<float>();
		cout << "ensemble_input_multiplier = " <<
			Global::ensemble_input_multiplier << endl;
	}

	Dataset dataset(
		songs_filename,
		test_users_filename,
		visible_utility_filenames,
		hidden_utility_filenames,
		song_tracks_filename,
		train_test_split);

	dataset.apply_tfidf_weights();  // better weights for scoring;
	// dataset.apply_bm25_weights();  // not a good weight;

	Embeddings embeddings(dataset,
		"song_embeddings.csv",
		song_tracks_filename,
		vector<string>({ "Musix_match_track_id" }));  // John;

	RecommendationCollection recommendation_collection;
	if (algo_name == "ensemble") {
		recommendation_collection.load(dataset);
	}

	if (output_canon_train != "")
		dataset.write_uirs(output_canon_train);

	// gridsearch_v4(dataset, algo_name, embeddings, recommendation_collection,
	// { "q", "alpha" },
	// {
	// vector<float>({ 1.5, 2.5, 3.5, }),
	// vector<float>({ 0.75, 0.85, 0.95 }),
	// });

	// gridsearch_v4(dataset, algo_name, embeddings, recommendation_collection,
	// 	{ "q", "alpha" },
	// 	{
	// 		vector<float>({ 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 }),
	// 		vector<float>({ 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 }),
	// 	});

	// gridsearch_v4(dataset, algo_name, embeddings, recommendation_collection,
	// 	{ "q", "alpha" },
	// 	{
	// 		vector<float>({ 1.5, 2.0, 2.5, 3.0, 3.5 }),
	// 		vector<float>({ 0.75, 0.8, 0.85, 0.9, 0.95 }),
	// 	});

	// gridsearch_v4(dataset, algo_name, embeddings, recommendation_collection,
	// 	{ "q", "alpha" },
	// 	{
	// 		vector<float>({ 2.5, 3.0, 3.5, 4.0, 4.5 }),
	// 		vector<float>({ 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75 }),
	// 	});

	// gridsearch_v4(dataset, algo_name, embeddings, recommendation_collection,
	// 	{ "ensemble_input_multiplier" },
	// 	{
	// 		vector<float>({
	// 			0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
	// 			1.75, 2.0, 2.25, 2.5, 2.75, 3.0,
	// 			3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0 }),
	// 	});

	run_all_v4("run-all-v4", dataset, algo_name,
		embeddings, recommendation_collection, false);

	return 0;
}

#pragma once

using std::string;
using std::map;
using std::vector;

class Config {
private:
	void parseConfig(map<string, string> &cfg, string &data) {
		size_t pos = data.find_first_not_of(" \t");
		if (pos != string::npos) {
			data = data.substr(pos);
		}
		pos = data.find("=");
		if (pos != string::npos) {
			std::istringstream keystream(data.substr(0, pos));
			std::istringstream valuestream(data.substr(pos + 1));
			string key, value;
			keystream >> key;
			valuestream >> value;
			if (key.size() && value.size() && key[0] != '#') {
				if (key == "inherit") {
					std::ifstream infile(value);
					string data;
					while (getline(infile, data)) {
						parseConfig(cfg, data);
					}
					infile.close();
					return;
				}
				if (cfg[key].size()) {
					value = cfg[key];
				}
				std::istringstream f_valuestream(value);
				float f_value;
				f_valuestream >> f_value;
				if (f_valuestream.eof() && !f_valuestream.fail()) {
					f[key] = f_value;
					i[key] = std::round(f_value);
				}
			}
		}
	};
	void loadConfig(string &path, map<string, string> &cfg) {
		std::ifstream infile(path + "/config.ini");
		string data;
		while (getline(infile, data)) {
			parseConfig(cfg, data);
		}
		infile.close();
	};
	void loadSource() {
		size_t pos;
		string data;
		std::ifstream infile(path + "/sources.dat");
		while (getline(infile, data)) {
			pos = data.find_first_not_of(" \t");
			if (pos != string::npos) {
				data = data.substr(pos);
			}
			float *source = host::create(7);
			std::istringstream sourcestream(data);
			for (size_t i = 0; i < 7; i++) {
				sourcestream >> source[i];
			}
			if (sourcestream.eof() && !sourcestream.fail()) {
				src.push_back(source);
			}
		}
		infile.close();
	};
	void loadStation() {
		size_t pos;
		string data;
		std::ifstream infile(path + "/stations.dat");
		while (getline(infile, data)) {
			pos = data.find_first_not_of(" \t");
			if (pos != string::npos) {
				data = data.substr(pos);
			}
			float *station = host::create(2);
			std::istringstream stationstream(data);
			for (size_t i = 0; i < 2; i++) {
				stationstream >> station[i];
			}
			if (stationstream.eof() && !stationstream.fail()) {
				rec.push_back(station);
			}
		}
		infile.close();
	};

public:
	vector<float*> src;
	vector<float*> rec;
	map<string, float> f;
	map<string, int> i;
	string path;
	Config(map<string, string> &cfg) {
		path = cfg["config"];
		loadConfig(path, cfg);
		loadSource();
		loadStation();
		if (i["clean"]) {
			removeDirectory(path + "/output");
		}
		createDirectory(path + "/output");
		if (this->i["nthread"]) {
			device::nthread = this->i["nthread"];
		}
	};
};

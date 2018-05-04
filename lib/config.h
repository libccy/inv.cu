#pragma once

using std::string;
using std::map;
using std::vector;

class Config {
public:
	vector<float*> src;
	vector<float*> rec;
	map<string, string> s;
	map<string, float> f;
	map<string, int> i;
	Config(map<string, string> cfg) {
		string data;
		enum Section {null, config, dir, sources, stations};
		map<string, Section> sectionMap {
			{"config", config},
			{"path", dir},
			{"sources", sources},
			{"stations", stations}
		};
		Section section = null;
		size_t pos;
		std::ifstream infile(cfg["config"]);
		size_t section_count = 0;

		while (getline(infile, data)) {
			pos = data.find_first_not_of(" \t");
			if (pos != string::npos) {
				data = data.substr(pos);
			}
			if (data[0] == '[') {
				section = null;
				pos = data.find(']');
				if (pos != string::npos) {
					section = sectionMap[data.substr(1, pos - 1)];
					if (section) {
						section_count++;
					}
				}
			}
			switch (section) {
				case config: case dir: {
					pos = data.find("=");
					if (pos != string::npos) {
						std::istringstream keystream(data.substr(0, pos));
						std::istringstream valuestream(data.substr(pos + 1));
						string key, value;
						keystream >> key;
						valuestream >> value;
						if (key.size() && value.size() && key[0] != '#') {
							if (cfg[key].size()) {
								value = cfg[key];
							}
							if (section == dir) {
								s[key] = value;
							}
							else {
								std::istringstream f_valuestream(value);
								float f_value;
								f_valuestream >> f_value;
								if (f_valuestream.eof() && !f_valuestream.fail()) {
									f[key] = f_value;
									i[key] = std::round(f_value);
								}
							}
						}
					}
					break;
				}
				case sources: {
					float *source = host::create(7);
					std::istringstream sourcestream(data);
					for (size_t i = 0; i < 7; i++) {
						sourcestream >> source[i];
					}
					if (sourcestream.eof() && !sourcestream.fail()) {
						src.push_back(source);
					}
					break;
				}
				case stations: {
					float *station = host::create(2);
					std::istringstream stationstream(data);
					for (size_t i = 0; i < 2; i++) {
						stationstream >> station[i];
					}
					if (stationstream.eof() && !stationstream.fail()) {
						rec.push_back(station);
					}
					break;
				}
			}
		}

		infile.close();

		if (section_count != 4) {
			std::cout << "error: invalid config file" << std::endl;
		}
	};
};

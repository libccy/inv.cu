#ifdef _WIN32
	#include <direct.h>
#elif defined __linux__
	#include <sys/stat.h>
	#define _mkdir(dir) mkdir(dir, 0777)
#endif

using std::string;

static void createDirectory(string path){
	size_t len = path.size();
	for (size_t i = 0; i < len; i++) {
		if (path[i] == '/') {
			_mkdir(path.substr(0, i).c_str());
		}
	}
	_mkdir(path.c_str());
}

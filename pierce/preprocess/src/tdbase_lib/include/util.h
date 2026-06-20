#ifndef TENG_UTIL_H_
#define TENG_UTIL_H_

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <sstream>
#include <vector>
#include <thread>
#include <iostream>
#include <cstdarg>
#include <ctime>
#include <string>
#include <fstream>
#include <streambuf>
#include <random>
#include <sys/stat.h>
#include <sys/types.h>
#include <mutex>
#include <filesystem>

#if defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <windows.h>
#else
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace tdbase{

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * timer functions
 * */
#define TENG_RANDOM_NUMBER 0315
#define INIT_TIME struct timeval start = get_cur_time();

using uint = unsigned int;

#if defined(_WIN32) || defined(_WIN64)
inline int gettimeofday(struct timeval* tp, struct timezone* tzp)
{
static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

SYSTEMTIME  system_time;
FILETIME    file_time;
uint64_t    time;

GetSystemTime(&system_time);
SystemTimeToFileTime(&system_time, &file_time);
time = ((uint64_t)file_time.dwLowDateTime);
time += ((uint64_t)file_time.dwHighDateTime) << 32;

tp->tv_sec = (long)((time - EPOCH) / 10000000L);
tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
return 0;
}
#endif


inline struct timeval get_cur_time(){
struct timeval t1;
gettimeofday(&t1, NULL);
return t1;
}
inline double get_time_elapsed(struct timeval &t1, bool update_start = false){
struct timeval t2;
    double elapsedTime;
gettimeofday(&t2, NULL);
elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
if(update_start){
t1 = get_cur_time();
}
return elapsedTime;
}

inline std::string time_string(){
struct timeval tv;
gettimeofday(&tv, NULL);
struct tm *nowtm;
char tmbuf[100];
char buf[256];
#if defined(_WIN32) || defined(_WIN64)
time_t tvsec = tv.tv_sec;
nowtm = localtime(&tvsec);
#else
nowtm = localtime(&tv.tv_sec);
#endif

strftime(tmbuf, sizeof tmbuf, "%H:%M:%S", nowtm);
sprintf(buf,"%s.%04ld", tmbuf, (long)(tv.tv_usec/1000));
return std::string(buf);
}

class Timer{
struct timeval t;
public:
Timer(){
t = get_cur_time();
}
double time_elapsed(bool update_start = false){
return get_time_elapsed(t, update_start);
}
void reset(){
t = get_cur_time();
}
};

/*
 * log functions
 * */

static std::mutex print_lock;
inline double logt(const char *format, struct timeval start, ...){

print_lock.lock();
va_list args;
va_start(args, start);
	char sprint_buf[4096];
	vsnprintf(sprint_buf, sizeof(sprint_buf), format, args);
va_end(args);
long tid = 0;
#if defined(__linux__)
tid = syscall(__NR_gettid);
#elif defined(_WIN32) || defined(_WIN64)
tid = GetCurrentThreadId();
#endif

fprintf(stderr,"%s thread %ld:\t%s", time_string().c_str(), tid,sprint_buf);

double mstime = get_time_elapsed(start, true);
if(mstime>1000){
fprintf(stderr," takes %f s\n", mstime/1000);
}else{
fprintf(stderr," takes %f ms\n", mstime);
}
fflush(stderr);

print_lock.unlock();
return mstime;
}

inline void log(const char *format, ...){
print_lock.lock();
va_list args;
va_start(args, format);
	char sprint_buf[4096];
	vsnprintf(sprint_buf, sizeof(sprint_buf), format, args);
va_end(args);
long tid = 0;
#if defined(__linux__)
tid = syscall(__NR_gettid);
#elif defined(_WIN32) || defined(_WIN64)
tid = GetCurrentThreadId();
#endif
fprintf(stderr,"%s thread %ld:\t%s\n", time_string().c_str(), tid,sprint_buf);
fflush(stderr);
print_lock.unlock();
}


inline int get_num_threads(){
return std::thread::hardware_concurrency();
}

static std::mutex plock;
inline void process_lock(){
plock.lock();
}

inline void process_unlock(){
plock.unlock();
}

/*
 * some random number based functions
 *
 * */

inline int get_rand_number(int max_value){
std::random_device dev;
std::mt19937 rng(dev());
std::uniform_int_distribution<std::mt19937::result_type> dist(0, max_value);
return dist(rng);
}

inline double get_rand_double(){
return get_rand_number(RAND_MAX)*1.0/RAND_MAX;
}

inline bool tryluck(float possibility){
assert(possibility>=0);
return possibility>=1.0||get_rand_double()<=possibility;
}

inline bool flip_coin(){
return tryluck(0.5);
}

/*
 * file or folder operations
 * Using std::filesystem (C++17)
 */
namespace fs = std::filesystem;

inline bool is_dir(const char* path) {
    if (!path) return false;
    std::error_code ec;
    return fs::is_directory(path, ec);
}

inline bool is_file(const char* path) {
    if (!path) return false;
    std::error_code ec;
    return fs::is_regular_file(path, ec);
}

inline void list_files(const char *path, std::vector<std::string> &f_list){
    if (is_file(path)) {
        f_list.push_back(std::string(path));
        return;
    }
    
    if (is_dir(path)) {
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(path, ec)) {
            if (entry.is_regular_file()) {
                f_list.push_back(entry.path().string());
            } else if (entry.is_directory()) {
                list_files(entry.path().string().c_str(), f_list);
            }
        }
    }
}


inline long file_size(const char *file){
    std::error_code ec;
    if (fs::exists(file, ec) && fs::is_regular_file(file, ec)) {
        return static_cast<long>(fs::file_size(file, ec));
    }
    return -1;
}

inline long file_size(std::vector<std::string> &f_list){
long size = 0;
for(std::string s:f_list){
long ls = file_size(s.c_str());
if(ls>0){
size += ls;
}
}
return size;
}

inline bool file_exist(const char *path) {
    std::error_code ec;
    return fs::exists(path, ec);
}

inline std::string read_line(){
std::string input_line;
std::getline(std::cin, input_line);
return input_line;
}

inline std::string read_file(const char *path){
std::ifstream t(path);
std::string str((std::istreambuf_iterator<char>(t)),
 std::istreambuf_iterator<char>());
t.close();
return str;
}

inline void write_file(const std::string &content, const char *path){
std::filebuf fb;
fb.open(path, std::ios::out | std::ios::trunc);
if(fb.is_open())
{
std::ostream os(&fb);
os << content.c_str();
}else{
std::cerr<<"cannot find path "<<path<<std::endl;
}
}


inline void tokenize( const std::string& str, std::vector<std::string>& result,
const std::string& delimiters = " ,;:\t",
const bool keepBlankFields=false,
const std::string& quote="\"\'"
){
    if (!result.empty()){
    result.clear();
    }

    if (delimiters.empty())
return ;

    std::string::size_type pos = 0; 
    char ch = 0; 

    char current_quote = 0; 
    bool quoted = false; 
    std::string token;  
    bool token_complete = false; 
    std::string::size_type len = str.length();

while(len > pos){
ch = str.at(pos);

bool add_char = true;
if ( false == quote.empty()){
if (std::string::npos != quote.find_first_of(ch)){
if (!quoted){
quoted = true;
current_quote = ch;
add_char = false;
} else {
if (current_quote == ch){
quoted = false;
current_quote = 0;
add_char = false;
}
}
}
}

if (!delimiters.empty()&&!quoted){
if (std::string::npos != delimiters.find_first_of(ch)){
token_complete = true;
add_char = false;
}
}

if (add_char){
token.push_back(ch);
}

if (token_complete){
if (token.empty())
{
if (keepBlankFields)
result.push_back("");
}
else
result.push_back( token );
token.clear();
token_complete = false;
}
++pos;
    } 
    if ( false == token.empty() ) {
    result.push_back( token );
    } else if(keepBlankFields && std::string::npos != delimiters.find_first_of(ch) ){
    result.push_back("");
    }
}


inline void remove_slash(std::string &str){
if(str.at(str.size() - 1) == '/'){
str = str.substr(0, str.size() - 1);
}
}

}
#endif

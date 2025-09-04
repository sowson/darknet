#ifndef SYSTEM_H
#define SYSTEM_H

#include <execinfo.h>

//#define MEM_LEAKS_DETECTION
#ifdef MEM_LEAKS_DETECTION
void *tracked_calloc(size_t count, size_t size, const char *file, int line);
void tracked_free(void *ptr, const char *file, int line);
#define CALLOC(count, size) tracked_calloc((count), (size), __FILE__, __LINE__)
#define FREE(ptr) tracked_free(ptr, __FILE__, __LINE__)
#else
#define CALLOC(count, size) calloc(count, size)
#define FREE(ptr) free(ptr)
#endif

int init_notified_file_name(char *dirname, const char **patterns, int (*system_notified_file_name) (char* file_name));

//struct inotify_event {
//  int      wd;       /* Watch descriptor */
//  uint32_t mask;     /* Mask describing event */
//  uint32_t cookie;   /* Unique cookie associating related
//                        events (for rename(2)) */
//  uint32_t len;      /* Size of name field */
//  char     name[];   /* Optional null-terminated name */
//};

int monitor_directory(const char *directory, const char **patterns, const size_t pattern_count, int (*system_notified_file_name) (char* file_name));

int exists(const char *fname);
int empty(char *dirname);

#endif // SYSTEM_H

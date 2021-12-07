#ifndef SYSTEM_H
#define SYSTEM_H

int init_notified_file_name(char *dirname, int (*process_file) (const char* file_name));

#ifdef __linux__
//struct inotify_event {
//  int      wd;       /* Watch descriptor */
//  uint32_t mask;     /* Mask describing event */
//  uint32_t cookie;   /* Unique cookie associating related
//                        events (for rename(2)) */
//  uint32_t len;      /* Size of name field */
//  char     name[];   /* Optional null-terminated name */
//};

int monitor_directory(const char *directory, const char **patterns, int (*process_file) (const char* file_name), const size_t pattern_count);
#endif

int exists(const char *fname, const char* ext);
int empty(char *dirname);

#endif // SYSTEM_H

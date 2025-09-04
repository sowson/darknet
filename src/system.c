#include "darknet.h"
#include <stdio.h>
#if !defined(WIN32)
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#endif

#ifndef MEM_TRACK
#define MEM_TRACK

#include "system.h"

#ifndef WIN32
#include <execinfo.h>
#endif

typedef struct allocation_node {
    void *pointer;
    size_t size;
    const char *file;
    int line;
    time_t timestamp;
    struct allocation_node *next;
} allocation_node;

static allocation_node *allocation_head = NULL;

void add_allocation(void *ptr, size_t size, const char *file, int line) {
    allocation_node *node = (allocation_node *)malloc(sizeof(allocation_node));
    if (!node) {
        fprintf(stderr, "failed to track allocation at %s:%d\n", file, line);
        exit(EXIT_FAILURE);
    }
    if (0) {
        allocation_node **current = &allocation_head;
        while (*current) {
            if (strcmp((*current)->file, file) == 0 && (*current)->line == line) {
                allocation_node *to_free = *current;
                *current = (*current)->next;
                free(to_free);
                break;
            }
            current = &(*current)->next;
        }
    }
    node->pointer = ptr;
    node->size = size;
    node->file = file;
    node->line = line;
    node->timestamp = time(NULL);
    node->next = allocation_head;
    allocation_head = node;
}

void remove_allocation(void *ptr, const char *file, int line) {
    allocation_node **current = &allocation_head;
    while (*current) {
        if ((void*)(*current)->pointer == (void*)ptr) {
            allocation_node *to_free = *current;
            *current = (*current)->next;
            free(to_free);
            return;
        }
        current = &(*current)->next;
    }
    fprintf(stderr, "warning: attempt to free untracked or already freed pointer %p at %s:%d\n", ptr, file, line);
    exit(EXIT_FAILURE);
}

void check_for_leaks() {
    allocation_node *current = allocation_head;
    time_t now = time(NULL);
    int detected = 0;
    while (current) {
        double seconds_elapsed = difftime(now, current->timestamp);
        if (seconds_elapsed >= 10) {
            detected = 1;
            if (detected) {
                void *buffer[1000];
#ifndef WIN32
                int size = backtrace(buffer, 1000);
                char **symbols = backtrace_symbols(buffer, size);
                if (symbols) {
                    fprintf(stderr, "call Stack:\n");
                    for (int i = 0; i < size; ++i) {
                        fprintf(stderr, "%s\n", symbols[i]);
                    }
                    free(symbols);
                } else {
                    fprintf(stderr, "failed to retrieve call stack symbols!\n");
                }
#endif
            }
            fprintf(stderr, "  leak detected! pointer: %p, Size: %zu, allocated at %s:%d, time elapsed: %.0f seconds\n",
                    current->pointer, current->size, current->file, current->line, seconds_elapsed);
        }
        current = current->next;
    }
    if (detected) exit(EXIT_FAILURE);
}

void print_allocations() {
    allocation_node *current = allocation_head;
    fprintf(stderr, "active allocations:\n");
    while (current) {
        fprintf(stderr, " pointer: %p, size: %zu, allocated at %s:%d, timestamp: %ld\n",
                current->pointer, current->size, current->file, current->line, current->timestamp);
        current = current->next;
    }
}

void* tracked_calloc(size_t count, size_t size, const char *file, int line) {
    void *ptr = calloc(count, size);
    if (!ptr) {
        fprintf(stderr, "memory allocation failed at %s:%d (count: %zu, size: %zu)\n", file, line, count, size);
        exit(EXIT_FAILURE);
    }
    add_allocation(ptr, count * size, file, line);
    //fprintf(stdut, "allocated memory at %p in %s:%d (count: %zu, size: %zu)\n", &ptr, file, line, count, size);
    return ptr;
}

void* tracked_track(void *ptr, const char *file, int line) {
    if (!ptr) {
        fprintf(stderr, "memory allocation failed at %s:%d (size: %zu)\n", file, line, sizeof(ptr));
        exit(EXIT_FAILURE);
    }
    add_allocation(ptr, sizeof(ptr), file, line);
    //fprintf(stdout, "allocated memory at %p in %s:%d (size: %zu)\n", ptr, file, line, sizeof(ptr));
    return ptr;
}

void tracked_free(void *ptr, const char *file, int line) {
    if (ptr) {
        remove_allocation(ptr, file, line);
        free(ptr);
        //fprintf(stderr, "freed memory at %p in %s:%d\n", ptr, file, line);
    } else {
        fprintf(stderr, "attempt to free NULL pointer in %s:%d\n", file, line);
    }
#ifndef WIN32
    check_for_leaks();
#endif
}

#endif

#if defined(__linux__)
#include <assert.h>
#include <errno.h>
#include <fnmatch.h>
#include <limits.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/inotify.h>
#include <unistd.h>
#include "system.h"

#ifdef __GNUC__
#  define ALIGNAS(TYPE) __attribute__ ((aligned(__alignof__(TYPE))))
#else
#  define ALIGNAS(TYPE) /* empty */
#endif

static volatile int initialized = 0;
static volatile int interrupted = 0;

static void interruption_handler(const int s)
{
  if (s == SIGINT)
    interrupted = 1;
}

int monitor_directory(const char *directory, const char **patterns, const size_t pattern_count, int (*system_notified_file_name) (char* file_name))
{
  int notifyfd = -1;
  int watchfd = -1;
  int ret = 0;
  const char * errmsg = "unknown error";

  notifyfd = inotify_init();
  if (notifyfd < 0)
  {
    errmsg = "inotify_init";
    goto catch;
  }

  watchfd = inotify_add_watch(notifyfd, directory, IN_MOVED_TO);
  if (watchfd < 0)
  {
    errmsg = "inotify_add_watch";
    goto catch;
  }

  while (1)
  {
    char buffer[1024];
    char fname[1024];
    ssize_t count = read(notifyfd, buffer, 1024);
    if (count < 0)
    {
      if (interrupted)
        goto finally;
      errmsg = "read";
        goto catch;
    }
    const struct inotify_event *event_ptr = (const struct inotify_event *)buffer;
    assert(event_ptr->wd == watchfd);
    assert(event_ptr->mask & IN_MOVE);
    if (event_ptr->len)
    {
      size_t i;
      for (i = 0; i < pattern_count; ++i)
      {
        switch (fnmatch(patterns[i], event_ptr->name, FNM_PATHNAME))
        {
          case 0:
          strncpy(fname, event_ptr->name, event_ptr->len);
          if (system_notified_file_name(fname) != 0) {
              errmsg = "system_notified_file_name";
              goto catch;
          }
          break;
          case FNM_NOMATCH:
            break;
          default:
            errmsg = "fnmatch";
            goto catch;
          }
      }
   }
 }
 finally:
  if (watchfd >= 0)
  {
    int status = close(watchfd);
    watchfd = -1;
    if (status < 0)
    {
      errmsg = "close(watchfd)";
      goto catch;
    }
  }
  if (notifyfd >= 0)
  {
    int status = close(notifyfd);
    notifyfd = -1;
    if (status < 0)
    {
      errmsg = "close(notifyfd)";
      goto catch;
    }
  }
  return ret;
 catch:
  if (errmsg && errno)
    perror(errmsg);
  ret = -1;
  goto finally;
}
#endif

int exists(const char *fname)
{
    FILE *file;
    if ((file = fopen(fname, "r")))
    {
        fclose(file);
        return 1;
    }
    return 0;
}

#if defined(__linux__)
int init_notified_file_name(char *dirname, const char **patterns, int (*system_notified_file_name) (char* file_name)) {
  if (!initialized) {
    struct sigaction sa;
    sa.sa_handler = interruption_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    initialized = 1;
  }
  if(monitor_directory(dirname, patterns, 1, system_notified_file_name) == 0)
    return 0;
  else
    return 1;
}

#elif defined(__APPLE__) || defined(__unix__)

#include <assert.h>
#include <errno.h>
#include <fnmatch.h>
#include <limits.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/event.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "system.h"

int empty(char *dirname) {
    int n = 0;
    struct dirent *d;
    DIR *dir = opendir(dirname);
    if (dir == NULL) // not a dir or doesn't exist
        return 1;
    while ((d = readdir(dir)) != NULL) {
        if(++n > 2)
            break;
    }
    closedir(dir);
    if (n <= 2) //dir empty
        return 1;
    else
        return 0;
}

#ifdef __GNUC__
#  define ALIGNAS(TYPE) __attribute__ ((aligned(__alignof__(TYPE))))
#else
#  define ALIGNAS(TYPE) /* empty */
#endif

static volatile int initialized = 0;
static volatile int interrupted = 0;

static void interruption_handler(const int s)
{
    if (s == SIGINT)
        interrupted = 1;
}

int monitor_directory(const char *directory, const char **patterns, const size_t pattern_count, int (*system_notified_file_name) (char* file_name))
{
    int kq = -1;
    int dir_fd = -1;
    int ret = 0;
    const char *errmsg = "unknown error";

    dir_fd = open(directory, O_RDONLY);
    if (dir_fd < 0) {
        perror("open directory");
        return -1;
    }

    kq = kqueue();
    if (kq == -1) {
        perror("kqueue");
        close(dir_fd);
        return -1;
    }

    // Set up the kevent structure to monitor directory for file children (IN_MOVED_TO equivalent)
    struct kevent change;
    EV_SET(&change, dir_fd, EVFILT_VNODE, EV_ADD | EV_ENABLE, NOTE_WRITE | NOTE_RENAME, 0, NULL);

    while (1) {
        if (interrupted) {
            goto finally;
        }

        struct timespec timeout;
        timeout.tv_sec = 0;
        timeout.tv_nsec = 100000;

        struct kevent event;
        int nev = kevent(kq, &change, 1, &event, 1, /* &timeout */ NULL);
        if (nev == -1) {
            errmsg = "kevent";
            goto catch;
        }

        if (event.fflags & NOTE_WRITE || event.fflags & NOTE_RENAME) {
            DIR *dir = opendir(directory);
            if (dir == NULL) {
                perror("opendir");
                ret = -1;
                goto finally;
            }

            struct dirent *de;
            while ((de = readdir(dir)) != NULL) {
                if (de->d_type != DT_REG) {
                    continue;
                }

                for (size_t i = 0; i < pattern_count; ++i) {
                    if (fnmatch(patterns[i], de->d_name, FNM_PATHNAME) == 0) {
                        char fname[PATH_MAX];
                        strncpy(fname, de->d_name, PATH_MAX);

                        if (system_notified_file_name(fname) != 0) {
                            errmsg = "system_notified_file_name";
                            ret = -1;
                            closedir(dir);
                            goto catch;
                        }

                        //break;
                        remove(fname);
                    }
                }
            }

            closedir(dir);
        }
    }

    finally:
    if (dir_fd >= 0) {
        close(dir_fd);
        dir_fd = -1;
    }

    if (kq >= 0) {
        close(kq);
        kq = -1;
    }

    return ret;

    catch:
    if (errmsg && errno) {
        perror(errmsg);
    }
    ret = -1;
    goto finally;
}

int init_notified_file_name(char *dirname, const char **patterns, int (*system_notified_file_name) (char* file_name)) {
    if (!initialized) {
        struct sigaction sa;
        sa.sa_handler = interruption_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGINT, &sa, NULL);
        initialized = 1;
    }

    if (monitor_directory(dirname, patterns, 1, system_notified_file_name) == 0)
        return 0;
    else
        return 1;
}

#elif defined(WIN32)

#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <fnmatch.h>

// Define ALIGNAS macro if needed
#ifdef __GNUC__
#  define ALIGNAS(TYPE) __attribute__ ((aligned(__alignof__(TYPE))))
#else
#  define ALIGNAS(TYPE) /* empty */
#endif

static volatile int initialized = 0;
static volatile int interrupted = 0;

// Console control handler to handle CTRL+C (SIGINT)
BOOL WINAPI ConsoleHandler(DWORD signal) {
    if (signal == CTRL_C_EVENT) {
        interrupted = 1;
    }
    return TRUE;
}

// Function to check if a directory is empty
int empty(char *dirname) {
    WIN32_FIND_DATAA findFileData;
    HANDLE hFind = INVALID_HANDLE_VALUE;
    char searchPath[MAX_PATH];

    // Prepare the search path with wildcard
    snprintf(searchPath, MAX_PATH, "%s\\*", dirname);

    // Find the first file in the directory
    hFind = FindFirstFileA(searchPath, &findFileData);

    if (hFind == INVALID_HANDLE_VALUE) {
        // If the directory cannot be opened, assume it's not empty
        perror("FindFirstFileA");
        return 0;
    }

    // Iterate through the files
    do {
        const char *name = findFileData.cFileName;
        if (strcmp(name, ".") != 0 && strcmp(name, "..") != 0) {
            // Found a file other than "." and ".."
            FindClose(hFind);
            return 0; // Directory is not empty
        }
    } while (FindNextFileA(hFind, &findFileData) != 0);

    // Close the directory handle
    FindClose(hFind);
    return 1; // Directory is empty
}

// Function to monitor directory changes using ReadDirectoryChangesW
int monitor_directory_windows(const char *directory, const char **patterns, const size_t pattern_count, int (*system_notified_file_name)(const char* file_name)) {
    HANDLE hDir = INVALID_HANDLE_VALUE;
    HANDLE hEvent = NULL;
    BYTE buffer[1024];
    DWORD bytesTransferred = 0;
    BOOL success;
    int ret = 0;

    // Open the directory handle
    hDir = CreateFileA(
        directory,
        FILE_LIST_DIRECTORY,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED,
        NULL
    );

    if (hDir == INVALID_HANDLE_VALUE) {
        perror("CreateFileA");
        return -1;
    }

    // Create an event for overlapped I/O
    hEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (hEvent == NULL) {
        perror("CreateEvent");
        CloseHandle(hDir);
        return -1;
    }

    // Initialize the OVERLAPPED structure
    OVERLAPPED overlapped;
    memset(&overlapped, 0, sizeof(OVERLAPPED));
    overlapped.hEvent = hEvent;

    while (!interrupted) {
        // Initiate an asynchronous directory change notification
        success = ReadDirectoryChangesW(
            hDir,
            &buffer,
            sizeof(buffer),
            FALSE, // FALSE: do not monitor subdirectories
            FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_DIR_NAME,
            &bytesTransferred,
            &overlapped,
            NULL // No completion routine
        );

        if (!success) {
            fprintf(stderr, "ReadDirectoryChangesW failed: %lu\n", GetLastError());
            ret = -1;
            break;
        }

        // Wait for the event to be signaled
        DWORD waitStatus = WaitForSingleObject(hEvent, INFINITE);
        if (waitStatus == WAIT_OBJECT_0) {
            // Retrieve the result of the overlapped operation
            success = GetOverlappedResult(hDir, &overlapped, &bytesTransferred, FALSE);
            if (!success) {
                fprintf(stderr, "GetOverlappedResult failed: %lu\n", GetLastError());
                ret = -1;
                break;
            }

            // Process each FILE_NOTIFY_INFORMATION record in the buffer
            DWORD offset = 0;
            while (offset < bytesTransferred) {
                FILE_NOTIFY_INFORMATION *pNotify = (FILE_NOTIFY_INFORMATION *)(buffer + offset);

                // Extract the file name from the notification
                int nameLength = pNotify->FileNameLength / sizeof(WCHAR);
                WCHAR wname[MAX_PATH];

                if (nameLength >= MAX_PATH) {
                    // Truncate if the file name is too long
                    nameLength = MAX_PATH - 1;
                }

                wcsncpy(wname, pNotify->FileName, nameLength);
                wname[nameLength] = L'\0'; // Null-terminate the string

                // Convert the wide-character file name to ANSI
                char fname[MAX_PATH];
                int nChars = WideCharToMultiByte(CP_ACP, 0, wname, -1, fname, sizeof(fname), NULL, NULL);
                if (nChars == 0) {
                    fprintf(stderr, "WideCharToMultiByte failed: %lu\n", GetLastError());
                    // Skip this file and continue with the next notification
                    goto next_entry;
                }

                // Determine the type of action
                switch (pNotify->Action) {
                    case FILE_ACTION_ADDED:
                    case FILE_ACTION_RENAMED_NEW_NAME:
                        // Check if the file name matches any of the provided patterns
                        for (size_t i = 0; i < pattern_count; ++i) {
                            if (fnmatch(patterns[i], fname, FNM_PATHNAME) == 0) {
                                // Invoke the callback with the matching file name
                                if (system_notified_file_name(fname) != 0) {
                                    fprintf(stderr, "system_notified_file_name failed for file: %s\n", fname);
                                    ret = -1;
                                    goto cleanup;
                                }

                                // Construct the full path of the file to remove
                                char fullPath[MAX_PATH];
                                snprintf(fullPath, MAX_PATH, "%s\\%s", directory, fname);

                                // Remove the file after processing
                                if (remove(fullPath) != 0) {
                                    perror("remove");
                                }

                                // Once a pattern is matched, no need to check other patterns
                                break;
                            }
                        }
                        break;

                    // You can handle other actions (e.g., FILE_ACTION_REMOVED) here if needed
                }

            next_entry:
                if (pNotify->NextEntryOffset == 0)
                    break; // No more entries
                offset += pNotify->NextEntryOffset;
            }
        }
        else {
            // Handle wait failure
            fprintf(stderr, "WaitForSingleObject failed: %lu\n", GetLastError());
            ret = -1;
            break;
        }
    }

cleanup:
    if (hEvent != NULL) {
        CloseHandle(hEvent);
    }

    if (hDir != INVALID_HANDLE_VALUE) {
        CloseHandle(hDir);
    }

    return ret;
}

// Function to initialize and start monitoring the directory
int init_notified_file_name(char *dirname, const char **patterns, int (*system_notified_file_name) (const char* file_name)){
    if (!initialized) {
        // Set up the console control handler for graceful shutdown
        if (!SetConsoleCtrlHandler(ConsoleHandler, TRUE)) {
            fprintf(stderr, "SetConsoleCtrlHandler failed\n");
            return 1;
        }
        initialized = 1;
    }

    // Assuming pattern_count is 1 as per your original Linux code
    size_t pattern_count = 1;

    // Start monitoring the directory
    if (monitor_directory_windows(dirname, patterns, pattern_count, system_notified_file_name) == 0)
        return 0;
    else
        return 1;
}

#endif
#include "darknet.h"
#include <stdio.h>
#if !defined(WIN32)
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
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

int monitor_directory(const char *directory, const char **patterns, const size_t pattern_count)
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

  watchfd = inotify_add_watch(notifyfd, directory, IN_MOVE);
  if (watchfd < 0)
  {
    errmsg = "inotify_add_watch";
    goto catch;
  }

  while (1)
  {
    char buffer[1024];
    char fname[1024];
    usleep(100);
    ssize_t count = read(notifyfd, buffer, sizeof(buffer));
    if (count < 0)
    {
      if (interrupted)
        goto finally;
      errmsg = "read";
        goto catch;
    }
    const struct inotify_event *event_ptr = (const struct inotify_event *) buffer;
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

int exists(const char *fname, const char* ext)
{
    FILE *file;
    if (strstr(fname, ext) && (file = fopen(fname, "r")))
    {
        fclose(file);
        return 1;
    }
    return 0;
}

#if defined(__linux__)

int init_notified_file_name(char *dirname) {
  if (!initialized) {
    struct sigaction sa;
    sa.sa_handler = interruption_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    initialized = 1;
  }
  const char* patterns[] = {"*.jpg"};
  if(monitor_directory(dirname, patterns, 1) == 0)
    return 0;
  else
    return 1;
}

#elif defined(__APPLE__)

int init_notified_file_name(char *dirname){
    return 0;
}

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

#elif defined(WIN32)

int init_notified_file_name(char *dirname){
    return 0; // TODO: Implement!
}

int empty(char *dirname) {
    return 0; // TODO: Implement!
}

#endif

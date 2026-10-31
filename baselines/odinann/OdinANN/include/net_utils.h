#ifndef NET_UTILS_H_
#define NET_UTILS_H_

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <string>

inline std::string getIP(const std::string &if_name) {
  struct ifreq ifr;
  int fd = socket(AF_INET, SOCK_DGRAM, 0);

  ifr.ifr_addr.sa_family = AF_INET;
  strncpy(ifr.ifr_name, if_name.c_str(), IFNAMSIZ - 1);

  ioctl(fd, SIOCGIFADDR, &ifr);
  close(fd);

  return std::string(inet_ntoa(((struct sockaddr_in *) &ifr.ifr_addr)->sin_addr));
}

inline std::string getIPWithPattern(const std::string &pattern) {
  struct ifaddrs *ifaddr, *ifa;
  int family, s;
  char host[NI_MAXHOST];
  std::string ip;

  if (getifaddrs(&ifaddr) == -1) {
    perror("getifaddrs");
    exit(EXIT_FAILURE);
  }

  for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) {
      continue;
    }

    family = ifa->ifa_addr->sa_family;

    if (family == AF_INET) {
      s = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host, NI_MAXHOST, nullptr, 0, NI_NUMERICHOST);

      if (s != 0) {
        printf("getnameinfo() failed: %s\n", gai_strerror(s));
        exit(EXIT_FAILURE);
      }

      if (std::string(host).find(pattern) != std::string::npos) {
        ip = std::string(host);
        break;
      }
    }
  }

  freeifaddrs(ifaddr);
  return ip;
}

inline std::string getMac(const std::string &if_name) {
  static struct ifreq ifr;
  int fd = socket(AF_INET, SOCK_DGRAM, 0);

  ifr.ifr_addr.sa_family = AF_INET;
  strncpy(ifr.ifr_name, if_name.c_str(), IFNAMSIZ - 1);

  ioctl(fd, SIOCGIFHWADDR, &ifr);
  close(fd);

  return std::string(ifr.ifr_hwaddr.sa_data);
}

#endif  // NET_UTILS_H_
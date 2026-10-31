#include "navis/build/build.h"
#include "navis/util/file_utils.h"

#include <cstdio>
#include <getopt.h>
#include <string>
#include <thread>
#include <unistd.h>

using namespace navis;

struct Args {
    std::string data_path;
    std::string output_prefix;
    std::string type    = "float";
    std::string metric  = "l2";
    uint32_t R          = 96;
    uint32_t L          = 128;
    float    pq_bytes   = 64.0f;
    float    mem_budget = 0.0f;     // 0 = 75% of physical memory
    uint32_t threads    = 0;
    float    sample     = 0.01f;
    uint32_t ent_R      = 32;
    uint32_t ent_L      = 64;
    float    alpha      = 1.2f;
};

static void print_usage(const char* prog) {
    std::printf(
        "Usage: %s [options]\n"
        "\n"
        "Required:\n"
        "  -d, --data       <path>    Input vectors (.bin format)\n"
        "  -o, --output     <prefix>  Output index prefix\n"
        "\n"
        "Index parameters:\n"
        "  -t, --type       <str>     Data type: float|int8|uint8 (default: float)\n"
        "  -m, --metric     <str>     Distance: l2|cosine|ip (default: l2)\n"
        "  -R               <int>     Max out-degree (default: 96)\n"
        "  -L               <int>     Build search list size (default: 128)\n"
        "      --pq-bytes   <float>   PQ bytes per vector (default: 64)\n"
        "      --mem-budget <float>   Host memory budget in GB; larger datasets are\n"
        "                             built in shards (default: 75%% of physical memory)\n"
        "      --threads    <int>     Build threads (default: all cores)\n"
        "      --alpha      <float>   Pruning alpha (default: 1.2)\n"
        "\n"
        "Entrance graph:\n"
        "      --sample     <float>   Sample rate (default: 0.01)\n"
        "      --ent-R      <int>     Entrance graph degree (default: 32)\n"
        "      --ent-L      <int>     Entrance graph build L (default: 64)\n",
        prog);
}

static Args parse_args(int argc, char** argv) {
    Args a;
    static struct option long_opts[] = {
        {"data",       required_argument, 0, 'd'},
        {"output",     required_argument, 0, 'o'},
        {"type",       required_argument, 0, 't'},
        {"metric",     required_argument, 0, 'm'},
        {"pq-bytes",   required_argument, 0, 1001},
        {"mem-budget", required_argument, 0, 1002},
        {"threads",    required_argument, 0, 1003},
        {"alpha",      required_argument, 0, 1004},
        {"sample",     required_argument, 0, 1005},
        {"ent-R",      required_argument, 0, 1006},
        {"ent-L",      required_argument, 0, 1007},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "d:o:t:m:R:L:", long_opts, nullptr)) != -1) {
        switch (c) {
            case 'd': a.data_path     = optarg; break;
            case 'o': a.output_prefix = optarg; break;
            case 't': a.type          = optarg; break;
            case 'm': a.metric        = optarg; break;
            case 'R': a.R             = std::stoi(optarg); break;
            case 'L': a.L             = std::stoi(optarg); break;
            case 1001: a.pq_bytes     = std::stof(optarg); break;
            case 1002: a.mem_budget   = std::stof(optarg); break;
            case 1003: a.threads      = std::stoi(optarg); break;
            case 1004: a.alpha        = std::stof(optarg); break;
            case 1005: a.sample       = std::stof(optarg); break;
            case 1006: a.ent_R        = std::stoi(optarg); break;
            case 1007: a.ent_L        = std::stoi(optarg); break;
            default: print_usage(argv[0]); exit(1);
        }
    }

    if (a.data_path.empty() || a.output_prefix.empty()) {
        print_usage(argv[0]);
        exit(1);
    }

    if (a.mem_budget <= 0.0f) {
        double phys = static_cast<double>(sysconf(_SC_PHYS_PAGES)) *
                      static_cast<double>(sysconf(_SC_PAGESIZE));
        a.mem_budget = static_cast<float>(0.75 * phys / 1e9);
    }

    if (a.threads == 0) {
        a.threads = std::thread::hardware_concurrency();
        if (a.threads == 0) a.threads = 8;
    }

    return a;
}

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    Metric m = metric_from_string(a.metric);

    if (a.type == "float") {
        build::build_full_index<float>(
            a.data_path, a.output_prefix, m,
            a.R, a.L, a.pq_bytes, a.mem_budget, a.threads,
            a.sample, a.ent_R, a.ent_L, a.alpha);
    } else if (a.type == "int8") {
        build::build_full_index<int8_t>(
            a.data_path, a.output_prefix, m,
            a.R, a.L, a.pq_bytes, a.mem_budget, a.threads,
            a.sample, a.ent_R, a.ent_L, a.alpha);
    } else if (a.type == "uint8") {
        build::build_full_index<uint8_t>(
            a.data_path, a.output_prefix, m,
            a.R, a.L, a.pq_bytes, a.mem_budget, a.threads,
            a.sample, a.ent_R, a.ent_L, a.alpha);
    } else {
        std::fprintf(stderr, "Unknown type: %s\n", a.type.c_str());
        return 1;
    }

    return 0;
}

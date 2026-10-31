#include "navis/types.h"
#include <algorithm>
#include <stdexcept>

namespace navis {

Metric metric_from_string(const std::string& s) {
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "l2")             return Metric::L2;
    if (lower == "ip" || lower == "inner_product" || lower == "mips")
                                   return Metric::INNER_PRODUCT;
    if (lower == "cosine")         return Metric::COSINE;
    throw std::invalid_argument("Unknown metric: " + s);
}

} // namespace navis

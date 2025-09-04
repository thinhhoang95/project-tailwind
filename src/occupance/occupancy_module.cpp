#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace py = pybind11;

namespace {

// Helper to safely get attribute; returns None if missing
inline py::object getattr_default(const py::object &obj, const char *name, const py::object &default_value) {
    if (py::hasattr(obj, name)) {
        try {
            return obj.attr(name);
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

inline long long to_int64_default(const py::object &o, long long default_value = 0) {
    try {
        return py::cast<long long>(o);
    } catch (...) {
        return default_value;
    }
}

inline int to_int_default(const py::object &o, int default_value = 0) {
    try {
        return py::cast<int>(o);
    } catch (...) {
        return default_value;
    }
}

inline double to_double_default(const py::object &o, double default_value = 0.0) {
    try {
        return py::cast<double>(o);
    } catch (...) {
        return default_value;
    }
}

} // namespace

py::dict compute_occupancy_cpp(
    const py::object &flight_list,
    const py::object &delays,
    const py::object &indexer,
    const py::object &tv_filter_py
) {
    // Determine TVs of interest
    std::unordered_set<std::string> tv_of_interest;
    if (!tv_filter_py.is_none()) {
        try {
            for (auto tv_obj : py::reinterpret_borrow<py::iterable>(tv_filter_py)) {
                try {
                    tv_of_interest.emplace(py::cast<std::string>(tv_obj));
                } catch (...) {
                    // Skip elements that cannot be cast to string
                }
            }
        } catch (...) {
            // If not iterable, fall back to empty filter set
        }
    } else {
        py::object mapping = getattr_default(indexer, "tv_id_to_idx", py::none());
        if (!mapping.is_none()) {
            try {
                py::object keys = mapping.attr("keys")();
                for (auto k : py::reinterpret_borrow<py::iterable>(keys)) {
                    try {
                        tv_of_interest.emplace(py::cast<std::string>(k));
                    } catch (...) {}
                }
            } catch (...) {
                // Ignore if keys() not available
            }
        }
    }

    // Number of time bins and bin length in seconds
    const int T = to_int_default(getattr_default(indexer, "num_time_bins", py::int_(0)), 0);
    const int bin_len_s = to_int_default(getattr_default(indexer, "time_bin_minutes", py::int_(0)), 0) * 60;

    // Prepare output arrays per TV
    std::unordered_map<std::string, py::array_t<long long>> tv_to_array;
    tv_to_array.reserve(tv_of_interest.size());
    for (const auto &tv : tv_of_interest) {
        py::array_t<long long> arr({(py::ssize_t)T});
        auto r = arr.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < (py::ssize_t)T; ++i) r(i) = 0;
        tv_to_array.emplace(tv, std::move(arr));
    }

    // Access flight metadata mapping
    py::object flight_metadata_obj = getattr_default(flight_list, "flight_metadata", py::dict());
    py::dict flight_metadata;
    try {
        flight_metadata = py::cast<py::dict>(flight_metadata_obj);
    } catch (...) {
        // If not a dict, attempt to construct a dict() from it
        try {
            flight_metadata = py::dict(flight_metadata_obj);
        } catch (...) {
            // Nothing to do; return empty mapping of preallocated TVs
            py::dict out;
            for (auto &kv : tv_to_array) {
                out[py::cast(kv.first)] = kv.second;
            }
            return out;
        }
    }

    // Iterate flights
    for (auto item : flight_metadata) {
        py::handle fid_h = item.first;
        py::handle meta_h = item.second;
        py::object fid = py::reinterpret_borrow<py::object>(fid_h);
        py::object meta = py::reinterpret_borrow<py::object>(meta_h);

        // Get delay in seconds
        py::object delay_min_obj;
        try {
            delay_min_obj = delays.attr("get")(fid, py::int_(0));
        } catch (...) {
            delay_min_obj = py::int_(0);
        }
        long long delay_sec = to_int64_default(delay_min_obj, 0) * 60LL;

        // Get intervals list; handle None or missing
        py::object intervals_obj;
        try {
            intervals_obj = meta.attr("get")(py::str("occupancy_intervals"), py::list());
        } catch (...) {
            intervals_obj = py::list();
        }
        if (intervals_obj.is_none()) {
            continue;
        }

        // Iterate intervals
        py::iterable intervals_iter;
        try {
            intervals_iter = py::reinterpret_borrow<py::iterable>(intervals_obj);
        } catch (...) {
            continue;
        }
        for (auto iv : intervals_iter) {
            py::object tvtw_idx_obj;
            try {
                tvtw_idx_obj = iv.attr("get")(py::str("tvtw_index"));
            } catch (...) {
                continue;
            }
            int tvtw_idx;
            try {
                tvtw_idx = py::cast<int>(tvtw_idx_obj);
            } catch (...) {
                continue;
            }

            // Decode TV and base bin
            py::object decoded;
            try {
                decoded = indexer.attr("get_tvtw_from_index")(tvtw_idx);
            } catch (...) {
                continue;
            }
            bool decoded_truthy = false;
            try {
                decoded_truthy = py::cast<bool>(decoded);
            } catch (...) {
                decoded_truthy = false;
            }
            if (!decoded_truthy) {
                continue;
            }

            std::string tv_id;
            int base_bin;
            try {
                auto tup = py::cast<py::tuple>(decoded);
                if (tup.size() < 2) {
                    continue;
                }
                tv_id = py::cast<std::string>(tup[0]);
                base_bin = py::cast<int>(tup[1]);
            } catch (...) {
                continue;
            }

            // Filter TV: always restrict to tv_of_interest
            if (tv_of_interest.find(tv_id) == tv_of_interest.end()) {
                continue;
            }

            // Entry time seconds
            double entry_s = 0.0;
            try {
                entry_s = to_double_default(iv.attr("get")(py::str("entry_time_s"), py::float_(0.0)), 0.0);
            } catch (...) {
                entry_s = 0.0;
            }

            // Whole-bin shift induced by delay
            if (bin_len_s <= 0 || T <= 0) {
                continue;
            }
            long long before = static_cast<long long>(std::floor(entry_s / static_cast<double>(bin_len_s)));
            long long after = static_cast<long long>(std::floor((entry_s + static_cast<double>(delay_sec)) / static_cast<double>(bin_len_s)));
            long long shift = after - before;
            long long b = static_cast<long long>(base_bin) + shift;
            if (b < 0 || b >= T) {
                continue;
            }

            auto it = tv_to_array.find(tv_id);
            if (it == tv_to_array.end()) {
                // If TV not preallocated (e.g., empty tv_of_interest), skip to match Python behavior
                continue;
            }
            long long *data_ptr = static_cast<long long *>(it->second.mutable_data());
            data_ptr[b] += 1;
        }
    }

    // Build output dict
    py::dict out;
    for (auto &kv : tv_to_array) {
        out[py::cast(kv.first)] = kv.second;
    }
    return out;
}

PYBIND11_MODULE(_occupancy, m) {
    m.doc() = "C++ acceleration for occupancy computation";
    m.def(
        "compute_occupancy",
        &compute_occupancy_cpp,
        py::arg("flight_list"),
        py::arg("delays"),
        py::arg("indexer"),
        py::arg("tv_filter") = py::none(),
        "Compute per-TV per-bin occupancy with delays applied"
    );
}



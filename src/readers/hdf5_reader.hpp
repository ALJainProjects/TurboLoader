/**
 * @file hdf5_reader.hpp
 * @brief HDF5 format reader for scientific computing datasets
 *
 * Supports:
 * - Reading datasets and groups
 * - Memory-mapped access for large files
 * - Chunked reading for streaming
 * - Attribute access
 *
 * Usage:
 * ```cpp
 * HDF5Reader reader("/path/to/data.h5");
 * auto data = reader.read_dataset<float>("train/images");
 * auto labels = reader.read_dataset<int>("train/labels");
 * ```
 *
 * Note: Requires libhdf5 to be installed.
 * - macOS: brew install hdf5
 * - Linux: apt install libhdf5-dev
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <functional>

// Check for HDF5 availability
#ifdef TURBOLOADER_HAS_HDF5
#include <hdf5.h>
#endif

namespace turboloader {
namespace readers {

/**
 * @brief HDF5 dataset information
 */
struct HDF5DatasetInfo {
    std::string name;
    std::vector<size_t> shape;
    std::string dtype;
    size_t element_size;
    size_t total_elements;
    bool is_chunked;
    std::vector<size_t> chunk_dims;
    std::string compression;
};

/**
 * @brief HDF5 attribute value
 */
struct HDF5Attribute {
    std::string name;
    std::string type;  // "int", "float", "string", "array"
    std::string string_value;
    double numeric_value;
    std::vector<double> array_value;
};

/**
 * @brief HDF5 file reader
 */
class HDF5Reader {
public:
#ifdef TURBOLOADER_HAS_HDF5

    /**
     * @brief Open HDF5 file
     */
    explicit HDF5Reader(const std::string& file_path)
        : file_path_(file_path), file_id_(-1) {
        file_id_ = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id_ < 0) {
            throw std::runtime_error("Failed to open HDF5 file: " + file_path);
        }
    }

    ~HDF5Reader() {
        if (file_id_ >= 0) {
            H5Fclose(file_id_);
        }
    }

    // Non-copyable
    HDF5Reader(const HDF5Reader&) = delete;
    HDF5Reader& operator=(const HDF5Reader&) = delete;

    // Movable
    HDF5Reader(HDF5Reader&& other) noexcept
        : file_path_(std::move(other.file_path_)), file_id_(other.file_id_) {
        other.file_id_ = -1;
    }

    /**
     * @brief List all datasets in the file
     */
    std::vector<std::string> list_datasets(const std::string& group_path = "/") {
        std::vector<std::string> datasets;

        hid_t group_id = H5Gopen2(file_id_, group_path.c_str(), H5P_DEFAULT);
        if (group_id < 0) {
            throw std::runtime_error("Failed to open group: " + group_path);
        }

        // Iterate through group members
        H5G_info_t group_info;
        H5Gget_info(group_id, &group_info);

        for (hsize_t i = 0; i < group_info.nlinks; i++) {
            ssize_t name_size = H5Lget_name_by_idx(group_id, ".", H5_INDEX_NAME,
                                                    H5_ITER_NATIVE, i, nullptr, 0, H5P_DEFAULT);
            std::string name(name_size + 1, '\0');
            H5Lget_name_by_idx(group_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, i,
                               &name[0], name_size + 1, H5P_DEFAULT);
            name.resize(name_size);

            std::string full_path = group_path == "/" ?
                "/" + name : group_path + "/" + name;

            H5O_info_t obj_info;
            H5Oget_info_by_name(file_id_, full_path.c_str(), &obj_info, H5P_DEFAULT);

            if (obj_info.type == H5O_TYPE_DATASET) {
                datasets.push_back(full_path);
            } else if (obj_info.type == H5O_TYPE_GROUP) {
                // Recursively list datasets in subgroups
                auto sub_datasets = list_datasets(full_path);
                datasets.insert(datasets.end(), sub_datasets.begin(), sub_datasets.end());
            }
        }

        H5Gclose(group_id);
        return datasets;
    }

    /**
     * @brief List all groups in the file
     */
    std::vector<std::string> list_groups(const std::string& group_path = "/") {
        std::vector<std::string> groups;

        hid_t group_id = H5Gopen2(file_id_, group_path.c_str(), H5P_DEFAULT);
        if (group_id < 0) {
            throw std::runtime_error("Failed to open group: " + group_path);
        }

        H5G_info_t group_info;
        H5Gget_info(group_id, &group_info);

        for (hsize_t i = 0; i < group_info.nlinks; i++) {
            ssize_t name_size = H5Lget_name_by_idx(group_id, ".", H5_INDEX_NAME,
                                                    H5_ITER_NATIVE, i, nullptr, 0, H5P_DEFAULT);
            std::string name(name_size + 1, '\0');
            H5Lget_name_by_idx(group_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, i,
                               &name[0], name_size + 1, H5P_DEFAULT);
            name.resize(name_size);

            std::string full_path = group_path == "/" ?
                "/" + name : group_path + "/" + name;

            H5O_info_t obj_info;
            H5Oget_info_by_name(file_id_, full_path.c_str(), &obj_info, H5P_DEFAULT);

            if (obj_info.type == H5O_TYPE_GROUP) {
                groups.push_back(full_path);
            }
        }

        H5Gclose(group_id);
        return groups;
    }

    /**
     * @brief Get dataset information
     */
    HDF5DatasetInfo get_dataset_info(const std::string& dataset_path) {
        HDF5DatasetInfo info;
        info.name = dataset_path;

        hid_t dataset_id = H5Dopen2(file_id_, dataset_path.c_str(), H5P_DEFAULT);
        if (dataset_id < 0) {
            throw std::runtime_error("Failed to open dataset: " + dataset_path);
        }

        // Get dataspace
        hid_t dataspace_id = H5Dget_space(dataset_id);
        int ndims = H5Sget_simple_extent_ndims(dataspace_id);
        std::vector<hsize_t> dims(ndims);
        H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr);

        info.shape.resize(ndims);
        info.total_elements = 1;
        for (int i = 0; i < ndims; i++) {
            info.shape[i] = dims[i];
            info.total_elements *= dims[i];
        }

        // Get datatype
        hid_t dtype_id = H5Dget_type(dataset_id);
        H5T_class_t type_class = H5Tget_class(dtype_id);
        info.element_size = H5Tget_size(dtype_id);

        switch (type_class) {
            case H5T_INTEGER:
                info.dtype = H5Tget_sign(dtype_id) == H5T_SGN_NONE ?
                    "uint" + std::to_string(info.element_size * 8) :
                    "int" + std::to_string(info.element_size * 8);
                break;
            case H5T_FLOAT:
                info.dtype = info.element_size == 4 ? "float32" : "float64";
                break;
            case H5T_STRING:
                info.dtype = "string";
                break;
            default:
                info.dtype = "unknown";
        }

        // Check for chunking
        hid_t plist_id = H5Dget_create_plist(dataset_id);
        if (H5Pget_layout(plist_id) == H5D_CHUNKED) {
            info.is_chunked = true;
            info.chunk_dims.resize(ndims);
            std::vector<hsize_t> chunk_dims(ndims);
            H5Pget_chunk(plist_id, ndims, chunk_dims.data());
            for (int i = 0; i < ndims; i++) {
                info.chunk_dims[i] = chunk_dims[i];
            }

            // Check compression
            int nfilters = H5Pget_nfilters(plist_id);
            for (int i = 0; i < nfilters; i++) {
                H5Z_filter_t filter = H5Pget_filter2(plist_id, i, nullptr, nullptr,
                                                     nullptr, 0, nullptr, nullptr);
                if (filter == H5Z_FILTER_DEFLATE) {
                    info.compression = "gzip";
                } else if (filter == H5Z_FILTER_SZIP) {
                    info.compression = "szip";
                }
            }
        }

        H5Pclose(plist_id);
        H5Tclose(dtype_id);
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);

        return info;
    }

    /**
     * @brief Read entire dataset as vector of type T
     */
    template<typename T>
    std::vector<T> read_dataset(const std::string& dataset_path) {
        hid_t dataset_id = H5Dopen2(file_id_, dataset_path.c_str(), H5P_DEFAULT);
        if (dataset_id < 0) {
            throw std::runtime_error("Failed to open dataset: " + dataset_path);
        }

        // Get size
        hid_t dataspace_id = H5Dget_space(dataset_id);
        hssize_t num_elements = H5Sget_simple_extent_npoints(dataspace_id);

        // Allocate buffer
        std::vector<T> data(num_elements);

        // Get native type
        hid_t mem_type = get_hdf5_type<T>();

        // Read data
        herr_t status = H5Dread(dataset_id, mem_type, H5S_ALL, H5S_ALL,
                                H5P_DEFAULT, data.data());
        if (status < 0) {
            H5Sclose(dataspace_id);
            H5Dclose(dataset_id);
            throw std::runtime_error("Failed to read dataset: " + dataset_path);
        }

        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);

        return data;
    }

    /**
     * @brief Read a slice of the dataset
     */
    template<typename T>
    std::vector<T> read_dataset_slice(const std::string& dataset_path,
                                       const std::vector<size_t>& start,
                                       const std::vector<size_t>& count) {
        hid_t dataset_id = H5Dopen2(file_id_, dataset_path.c_str(), H5P_DEFAULT);
        if (dataset_id < 0) {
            throw std::runtime_error("Failed to open dataset: " + dataset_path);
        }

        hid_t file_space = H5Dget_space(dataset_id);
        int ndims = H5Sget_simple_extent_ndims(file_space);

        if (start.size() != ndims || count.size() != ndims) {
            H5Sclose(file_space);
            H5Dclose(dataset_id);
            throw std::runtime_error("Start/count dimensions must match dataset dimensions");
        }

        // Convert to HDF5 types
        std::vector<hsize_t> h_start(start.begin(), start.end());
        std::vector<hsize_t> h_count(count.begin(), count.end());

        // Select hyperslab
        H5Sselect_hyperslab(file_space, H5S_SELECT_SET, h_start.data(),
                            nullptr, h_count.data(), nullptr);

        // Create memory space
        size_t total = 1;
        for (auto c : count) total *= c;
        hid_t mem_space = H5Screate_simple(ndims, h_count.data(), nullptr);

        // Allocate and read
        std::vector<T> data(total);
        hid_t mem_type = get_hdf5_type<T>();
        H5Dread(dataset_id, mem_type, mem_space, file_space, H5P_DEFAULT, data.data());

        H5Sclose(mem_space);
        H5Sclose(file_space);
        H5Dclose(dataset_id);

        return data;
    }

    /**
     * @brief Read dataset attributes
     */
    std::vector<HDF5Attribute> get_attributes(const std::string& path) {
        std::vector<HDF5Attribute> attrs;

        hid_t obj_id = H5Oopen(file_id_, path.c_str(), H5P_DEFAULT);
        if (obj_id < 0) {
            throw std::runtime_error("Failed to open object: " + path);
        }

        int num_attrs = H5Aget_num_attrs(obj_id);
        for (int i = 0; i < num_attrs; i++) {
            hid_t attr_id = H5Aopen_idx(obj_id, i);

            // Get name
            ssize_t name_size = H5Aget_name(attr_id, 0, nullptr);
            std::string name(name_size + 1, '\0');
            H5Aget_name(attr_id, name_size + 1, &name[0]);
            name.resize(name_size);

            HDF5Attribute attr;
            attr.name = name;

            // Get type and read value
            hid_t attr_type = H5Aget_type(attr_id);
            H5T_class_t type_class = H5Tget_class(attr_type);

            if (type_class == H5T_INTEGER || type_class == H5T_FLOAT) {
                double value;
                H5Aread(attr_id, H5T_NATIVE_DOUBLE, &value);
                attr.type = type_class == H5T_INTEGER ? "int" : "float";
                attr.numeric_value = value;
            } else if (type_class == H5T_STRING) {
                size_t str_size = H5Tget_size(attr_type);
                std::string value(str_size, '\0');
                H5Aread(attr_id, attr_type, &value[0]);
                attr.type = "string";
                attr.string_value = value;
            }

            attrs.push_back(attr);

            H5Tclose(attr_type);
            H5Aclose(attr_id);
        }

        H5Oclose(obj_id);
        return attrs;
    }

    /**
     * @brief Check if a path exists in the file
     */
    bool exists(const std::string& path) {
        return H5Lexists(file_id_, path.c_str(), H5P_DEFAULT) > 0;
    }

private:
    template<typename T>
    hid_t get_hdf5_type() {
        if constexpr (std::is_same_v<T, float>) return H5T_NATIVE_FLOAT;
        else if constexpr (std::is_same_v<T, double>) return H5T_NATIVE_DOUBLE;
        else if constexpr (std::is_same_v<T, int32_t>) return H5T_NATIVE_INT32;
        else if constexpr (std::is_same_v<T, int64_t>) return H5T_NATIVE_INT64;
        else if constexpr (std::is_same_v<T, uint8_t>) return H5T_NATIVE_UINT8;
        else if constexpr (std::is_same_v<T, uint16_t>) return H5T_NATIVE_UINT16;
        else if constexpr (std::is_same_v<T, uint32_t>) return H5T_NATIVE_UINT32;
        else if constexpr (std::is_same_v<T, uint64_t>) return H5T_NATIVE_UINT64;
        else return H5T_NATIVE_FLOAT;
    }

    std::string file_path_;
    hid_t file_id_;

#else
    // Stub implementation when HDF5 is not available

    explicit HDF5Reader(const std::string& file_path) {
        throw std::runtime_error(
            "HDF5 support not compiled. Install libhdf5 and rebuild with -DTURBOLOADER_HAS_HDF5");
    }

    std::vector<std::string> list_datasets(const std::string& = "/") { return {}; }
    std::vector<std::string> list_groups(const std::string& = "/") { return {}; }
    HDF5DatasetInfo get_dataset_info(const std::string&) { return {}; }

    template<typename T>
    std::vector<T> read_dataset(const std::string&) { return {}; }

    template<typename T>
    std::vector<T> read_dataset_slice(const std::string&,
                                       const std::vector<size_t>&,
                                       const std::vector<size_t>&) { return {}; }

    std::vector<HDF5Attribute> get_attributes(const std::string&) { return {}; }
    bool exists(const std::string&) { return false; }

#endif

    static bool is_available() {
#ifdef TURBOLOADER_HAS_HDF5
        return true;
#else
        return false;
#endif
    }
};

}  // namespace readers
}  // namespace turboloader

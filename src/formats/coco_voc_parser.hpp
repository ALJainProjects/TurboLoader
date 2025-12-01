/**
 * @file coco_voc_parser.hpp
 * @brief COCO and Pascal VOC annotation format parsers
 *
 * Supports:
 * - COCO JSON format (annotations, bounding boxes, segmentation, keypoints)
 * - Pascal VOC XML format (annotations, bounding boxes)
 *
 * Usage:
 * ```cpp
 * // COCO format
 * COCOParser coco("path/to/annotations.json");
 * auto annotations = coco.get_annotations(image_id);
 *
 * // Pascal VOC format
 * VOCParser voc("path/to/Annotations");
 * auto annotations = voc.parse("000001.xml");
 * ```
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <sstream>
#include <cmath>

namespace turboloader {
namespace formats {

/**
 * @brief Bounding box representation
 */
struct BoundingBox {
    float x;        // Top-left x coordinate
    float y;        // Top-left y coordinate
    float width;    // Box width
    float height;   // Box height
    int category_id;
    float confidence;
    std::string category_name;

    // Convert to different formats
    std::array<float, 4> to_xyxy() const {
        return {x, y, x + width, y + height};
    }

    std::array<float, 4> to_xywh() const {
        return {x, y, width, height};
    }

    std::array<float, 4> to_cxcywh() const {
        return {x + width / 2, y + height / 2, width, height};
    }

    float area() const {
        return width * height;
    }

    float iou(const BoundingBox& other) const {
        float x1 = std::max(x, other.x);
        float y1 = std::max(y, other.y);
        float x2 = std::min(x + width, other.x + other.width);
        float y2 = std::min(y + height, other.y + other.height);

        float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        float union_area = area() + other.area() - inter_area;

        return union_area > 0 ? inter_area / union_area : 0.0f;
    }
};

/**
 * @brief Segmentation mask (polygon or RLE)
 */
struct Segmentation {
    enum class Type { POLYGON, RLE };

    Type type;
    std::vector<std::vector<float>> polygons;  // For polygon type
    std::string rle_counts;                     // For RLE type
    int rle_height;
    int rle_width;

    bool empty() const {
        return polygons.empty() && rle_counts.empty();
    }
};

/**
 * @brief Keypoint annotation
 */
struct Keypoint {
    float x;
    float y;
    int visibility;  // 0: not labeled, 1: labeled but not visible, 2: labeled and visible

    bool is_visible() const { return visibility == 2; }
    bool is_labeled() const { return visibility > 0; }
};

/**
 * @brief Complete annotation for an object
 */
struct Annotation {
    int id;
    int image_id;
    int category_id;
    std::string category_name;
    BoundingBox bbox;
    Segmentation segmentation;
    std::vector<Keypoint> keypoints;
    float area;
    bool is_crowd;

    // Additional attributes
    std::unordered_map<std::string, std::string> attributes;
};

/**
 * @brief Image metadata
 */
struct ImageInfo {
    int id;
    std::string file_name;
    int width;
    int height;
    std::string coco_url;
    std::string flickr_url;
    std::string date_captured;
};

/**
 * @brief Category information
 */
struct Category {
    int id;
    std::string name;
    std::string supercategory;
    std::vector<std::string> keypoint_names;
    std::vector<std::pair<int, int>> skeleton;
};

/**
 * @brief Simple JSON parser for COCO format
 * Lightweight implementation without external dependencies
 */
class SimpleJSONParser {
public:
    enum class TokenType {
        OBJECT_START, OBJECT_END,
        ARRAY_START, ARRAY_END,
        STRING, NUMBER, BOOL_TRUE, BOOL_FALSE, NULL_VALUE,
        COLON, COMMA, END_OF_FILE
    };

    struct Token {
        TokenType type;
        std::string value;
    };

private:
    std::string json_;
    size_t pos_ = 0;

    void skip_whitespace() {
        while (pos_ < json_.size() && std::isspace(json_[pos_])) {
            pos_++;
        }
    }

    Token next_token() {
        skip_whitespace();
        if (pos_ >= json_.size()) {
            return {TokenType::END_OF_FILE, ""};
        }

        char c = json_[pos_];

        switch (c) {
            case '{': pos_++; return {TokenType::OBJECT_START, "{"};
            case '}': pos_++; return {TokenType::OBJECT_END, "}"};
            case '[': pos_++; return {TokenType::ARRAY_START, "["};
            case ']': pos_++; return {TokenType::ARRAY_END, "]"};
            case ':': pos_++; return {TokenType::COLON, ":"};
            case ',': pos_++; return {TokenType::COMMA, ","};
            case '"': return parse_string();
            case 't':
                if (json_.substr(pos_, 4) == "true") {
                    pos_ += 4;
                    return {TokenType::BOOL_TRUE, "true"};
                }
                break;
            case 'f':
                if (json_.substr(pos_, 5) == "false") {
                    pos_ += 5;
                    return {TokenType::BOOL_FALSE, "false"};
                }
                break;
            case 'n':
                if (json_.substr(pos_, 4) == "null") {
                    pos_ += 4;
                    return {TokenType::NULL_VALUE, "null"};
                }
                break;
            default:
                if (std::isdigit(c) || c == '-' || c == '+') {
                    return parse_number();
                }
        }

        throw std::runtime_error("Invalid JSON at position " + std::to_string(pos_));
    }

    Token parse_string() {
        pos_++;  // Skip opening quote
        std::string result;

        while (pos_ < json_.size() && json_[pos_] != '"') {
            if (json_[pos_] == '\\' && pos_ + 1 < json_.size()) {
                pos_++;
                switch (json_[pos_]) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    default: result += json_[pos_];
                }
            } else {
                result += json_[pos_];
            }
            pos_++;
        }
        pos_++;  // Skip closing quote

        return {TokenType::STRING, result};
    }

    Token parse_number() {
        size_t start = pos_;
        if (json_[pos_] == '-' || json_[pos_] == '+') pos_++;

        while (pos_ < json_.size() && (std::isdigit(json_[pos_]) ||
               json_[pos_] == '.' || json_[pos_] == 'e' || json_[pos_] == 'E' ||
               json_[pos_] == '+' || json_[pos_] == '-')) {
            pos_++;
        }

        return {TokenType::NUMBER, json_.substr(start, pos_ - start)};
    }

public:
    explicit SimpleJSONParser(const std::string& json) : json_(json), pos_(0) {}

    // Parse methods for COCO format
    std::vector<Annotation> parse_annotations();
    std::vector<ImageInfo> parse_images();
    std::vector<Category> parse_categories();
};

/**
 * @brief COCO annotation format parser
 */
class COCOParser {
public:
    /**
     * @brief Load COCO annotations from JSON file
     */
    explicit COCOParser(const std::string& annotation_file) {
        load(annotation_file);
    }

    COCOParser() = default;

    /**
     * @brief Load annotations from file
     */
    void load(const std::string& annotation_file) {
        std::ifstream file(annotation_file);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open annotation file: " + annotation_file);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        json_content_ = buffer.str();

        parse_json();
    }

    /**
     * @brief Get all annotations for an image
     */
    std::vector<Annotation> get_annotations(int image_id) const {
        auto it = image_annotations_.find(image_id);
        if (it != image_annotations_.end()) {
            return it->second;
        }
        return {};
    }

    /**
     * @brief Get image info by ID
     */
    const ImageInfo* get_image_info(int image_id) const {
        auto it = images_.find(image_id);
        if (it != images_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    /**
     * @brief Get image info by filename
     */
    const ImageInfo* get_image_by_filename(const std::string& filename) const {
        auto it = filename_to_image_.find(filename);
        if (it != filename_to_image_.end()) {
            return get_image_info(it->second);
        }
        return nullptr;
    }

    /**
     * @brief Get category by ID
     */
    const Category* get_category(int category_id) const {
        auto it = categories_.find(category_id);
        if (it != categories_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    /**
     * @brief Get category name by ID
     */
    std::string get_category_name(int category_id) const {
        auto cat = get_category(category_id);
        return cat ? cat->name : "";
    }

    /**
     * @brief Get all image IDs
     */
    std::vector<int> get_image_ids() const {
        std::vector<int> ids;
        ids.reserve(images_.size());
        for (const auto& [id, _] : images_) {
            ids.push_back(id);
        }
        return ids;
    }

    /**
     * @brief Get all category IDs
     */
    std::vector<int> get_category_ids() const {
        std::vector<int> ids;
        ids.reserve(categories_.size());
        for (const auto& [id, _] : categories_) {
            ids.push_back(id);
        }
        return ids;
    }

    /**
     * @brief Filter annotations by category
     */
    std::vector<Annotation> get_annotations_by_category(int category_id) const {
        std::vector<Annotation> result;
        for (const auto& [_, annotations] : image_annotations_) {
            for (const auto& ann : annotations) {
                if (ann.category_id == category_id) {
                    result.push_back(ann);
                }
            }
        }
        return result;
    }

    /**
     * @brief Get number of images
     */
    size_t num_images() const { return images_.size(); }

    /**
     * @brief Get number of annotations
     */
    size_t num_annotations() const { return total_annotations_; }

    /**
     * @brief Get number of categories
     */
    size_t num_categories() const { return categories_.size(); }

private:
    void parse_json();

    std::string json_content_;
    std::unordered_map<int, ImageInfo> images_;
    std::unordered_map<std::string, int> filename_to_image_;
    std::unordered_map<int, Category> categories_;
    std::unordered_map<int, std::vector<Annotation>> image_annotations_;
    size_t total_annotations_ = 0;
};

/**
 * @brief Pascal VOC XML annotation parser
 */
class VOCParser {
public:
    /**
     * @brief Create parser for VOC annotation directory
     */
    explicit VOCParser(const std::string& annotation_dir)
        : annotation_dir_(annotation_dir) {}

    VOCParser() = default;

    /**
     * @brief Parse a single VOC XML annotation file
     */
    std::vector<Annotation> parse(const std::string& xml_filename) const {
        std::string full_path = annotation_dir_.empty() ?
            xml_filename : annotation_dir_ + "/" + xml_filename;

        std::ifstream file(full_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open VOC annotation: " + full_path);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        return parse_xml(buffer.str());
    }

    /**
     * @brief Parse VOC XML from string
     */
    std::vector<Annotation> parse_xml(const std::string& xml_content) const {
        std::vector<Annotation> annotations;

        // Simple XML parsing for VOC format
        size_t pos = 0;
        int ann_id = 0;

        // Parse image size
        int img_width = 0, img_height = 0;
        auto size_start = xml_content.find("<size>");
        if (size_start != std::string::npos) {
            img_width = parse_int_tag(xml_content, "width", size_start);
            img_height = parse_int_tag(xml_content, "height", size_start);
        }

        // Parse objects
        while ((pos = xml_content.find("<object>", pos)) != std::string::npos) {
            auto obj_end = xml_content.find("</object>", pos);
            if (obj_end == std::string::npos) break;

            std::string obj_xml = xml_content.substr(pos, obj_end - pos);

            Annotation ann;
            ann.id = ann_id++;
            ann.category_name = parse_string_tag(obj_xml, "name");

            // Parse bounding box
            auto bndbox_start = obj_xml.find("<bndbox>");
            if (bndbox_start != std::string::npos) {
                float xmin = parse_float_tag(obj_xml, "xmin", bndbox_start);
                float ymin = parse_float_tag(obj_xml, "ymin", bndbox_start);
                float xmax = parse_float_tag(obj_xml, "xmax", bndbox_start);
                float ymax = parse_float_tag(obj_xml, "ymax", bndbox_start);

                ann.bbox.x = xmin;
                ann.bbox.y = ymin;
                ann.bbox.width = xmax - xmin;
                ann.bbox.height = ymax - ymin;
                ann.bbox.category_name = ann.category_name;
                ann.bbox.confidence = 1.0f;
                ann.area = ann.bbox.area();
            }

            // Parse difficulty/truncated/occluded flags
            ann.is_crowd = parse_int_tag(obj_xml, "difficult") == 1;
            ann.attributes["truncated"] = std::to_string(parse_int_tag(obj_xml, "truncated"));
            ann.attributes["occluded"] = std::to_string(parse_int_tag(obj_xml, "occluded"));
            ann.attributes["pose"] = parse_string_tag(obj_xml, "pose");

            annotations.push_back(ann);
            pos = obj_end + 1;
        }

        return annotations;
    }

    /**
     * @brief Get image filename from annotation
     */
    std::string get_image_filename(const std::string& xml_content) const {
        return parse_string_tag(xml_content, "filename");
    }

    /**
     * @brief Get image size from annotation
     */
    std::pair<int, int> get_image_size(const std::string& xml_content) const {
        auto size_start = xml_content.find("<size>");
        if (size_start != std::string::npos) {
            int width = parse_int_tag(xml_content, "width", size_start);
            int height = parse_int_tag(xml_content, "height", size_start);
            return {width, height};
        }
        return {0, 0};
    }

private:
    std::string parse_string_tag(const std::string& xml, const std::string& tag,
                                  size_t start = 0) const {
        std::string open_tag = "<" + tag + ">";
        std::string close_tag = "</" + tag + ">";

        auto tag_start = xml.find(open_tag, start);
        if (tag_start == std::string::npos) return "";

        tag_start += open_tag.length();
        auto tag_end = xml.find(close_tag, tag_start);
        if (tag_end == std::string::npos) return "";

        return xml.substr(tag_start, tag_end - tag_start);
    }

    int parse_int_tag(const std::string& xml, const std::string& tag,
                      size_t start = 0) const {
        std::string value = parse_string_tag(xml, tag, start);
        if (value.empty()) return 0;
        try {
            return std::stoi(value);
        } catch (...) {
            return 0;
        }
    }

    float parse_float_tag(const std::string& xml, const std::string& tag,
                          size_t start = 0) const {
        std::string value = parse_string_tag(xml, tag, start);
        if (value.empty()) return 0.0f;
        try {
            return std::stof(value);
        } catch (...) {
            return 0.0f;
        }
    }

    std::string annotation_dir_;
};

/**
 * @brief Dataset wrapper for COCO format
 */
class COCODataset {
public:
    COCODataset(const std::string& image_dir, const std::string& annotation_file)
        : image_dir_(image_dir), parser_(annotation_file) {
        image_ids_ = parser_.get_image_ids();
    }

    size_t size() const { return image_ids_.size(); }

    struct Sample {
        std::string image_path;
        std::vector<Annotation> annotations;
        ImageInfo image_info;
    };

    Sample get(size_t index) const {
        if (index >= image_ids_.size()) {
            throw std::out_of_range("Index out of range");
        }

        int image_id = image_ids_[index];
        const ImageInfo* info = parser_.get_image_info(image_id);

        Sample sample;
        if (info) {
            sample.image_path = image_dir_ + "/" + info->file_name;
            sample.image_info = *info;
        }
        sample.annotations = parser_.get_annotations(image_id);

        return sample;
    }

    const COCOParser& parser() const { return parser_; }

private:
    std::string image_dir_;
    COCOParser parser_;
    std::vector<int> image_ids_;
};

/**
 * @brief Dataset wrapper for Pascal VOC format
 */
class VOCDataset {
public:
    VOCDataset(const std::string& root_dir, const std::string& split = "train")
        : root_dir_(root_dir), split_(split),
          parser_(root_dir + "/Annotations") {
        load_image_set();
    }

    size_t size() const { return image_ids_.size(); }

    struct Sample {
        std::string image_path;
        std::vector<Annotation> annotations;
        std::string image_id;
    };

    Sample get(size_t index) const {
        if (index >= image_ids_.size()) {
            throw std::out_of_range("Index out of range");
        }

        const std::string& image_id = image_ids_[index];

        Sample sample;
        sample.image_id = image_id;
        sample.image_path = root_dir_ + "/JPEGImages/" + image_id + ".jpg";
        sample.annotations = parser_.parse(image_id + ".xml");

        return sample;
    }

private:
    void load_image_set() {
        std::string set_file = root_dir_ + "/ImageSets/Main/" + split_ + ".txt";
        std::ifstream file(set_file);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open image set file: " + set_file);
        }

        std::string line;
        while (std::getline(file, line)) {
            // Remove whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            if (!line.empty()) {
                image_ids_.push_back(line);
            }
        }
    }

    std::string root_dir_;
    std::string split_;
    VOCParser parser_;
    std::vector<std::string> image_ids_;
};

}  // namespace formats
}  // namespace turboloader

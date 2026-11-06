#ifndef PSUM_SERIALIZATION_OBJECTMANAGER_HPP
#define PSUM_SERIALIZATION_OBJECTMANAGER_HPP

#include <cxxabi.h>
#include "mas.hpp"
#include "tag_sl.hpp"
#include "container_sl.hpp"
#include "simplify_type_name.hpp"
#include <any>
#include <memory>
#include <optional>
#include <sycl/sycl.hpp>

namespace psum {

namespace serialization {

    class object_manager {
    private:
        std::filesystem::path file_path_;
        std::unique_ptr<serialization::mas_file> file_;
        mutable std::optional<sycl::queue> q_;
        std::unordered_map<std::string, std::any> objects_;
        std::unordered_map<std::string, std::function<void(mas_file&)>> savers_;

        // mng will not change associated mas file
        // if mng is associated with a file, valid saving path must be a different one
        bool check_valid_save_path(const std::filesystem::path& target) const {
            if (target.empty()) {
                // of course, empty path is not valid
                return false;
            } else if (file_path_.empty()) {
                // this mng is not associated with any file so any file path in existing directory is valid
                return std::filesystem::exists(std::filesystem::absolute(target).parent_path());
            } else {
                if (std::filesystem::exists(target)) {
                    // target file already exists, check if it is the same as the current file
                    try {
                        return std::filesystem::equivalent(file_path_, target) == false;
                    } catch (const std::exception& e) {
                        std::cerr << e.what() << std::endl;
                        return false;
                    }
                } else {
                    // target file does not exist
                    // any file path in existing directory is valid
                    return std::filesystem::exists(std::filesystem::absolute(target).parent_path());
                }
            }
        }

        template <typename T>
        void assert_supported_type() const {
            static_assert(
                std::is_default_constructible_v<T> || std::is_constructible_v<T, sycl::queue&>,
                "Only default-constructible types or queue-constructible types can be managed by object_manager.");
        }

    public:
        object_manager(const std::filesystem::path& file_path) : file_path_(file_path) {
            if (!file_path.empty()) {
                try {
                    file_ = std::make_unique<serialization::mas_file>(file_path);
                } catch (const std::exception& e) {
                    throw std::runtime_error("Failed to create object manager: " + std::string(e.what()));
                }
            }
        }

        object_manager(const std::filesystem::path& file_path, const sycl::queue& q)
            : object_manager(file_path) {
            q_ = q;
        }

        ~object_manager() = default;
        object_manager(const object_manager&) = delete;
        object_manager& operator=(const object_manager&) = delete;

        template <typename T>
        std::string type_name() const {
            static const std::string name = []{
                int status = 0;
                char* demangled = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status);
                std::string result = (status == 0 && demangled) ? demangled : typeid(T).name();
                std::free(demangled);
                return simplify_type_name(result);
            }();
            return name;
        }

        template<typename T>
        T& obj(const std::string& name) {
            assert_supported_type<T>();
            std::string prefix = type_name<T>();
            std::string key = prefix + "[" + name + "]";
            auto it = objects_.find(key);
            if (it != objects_.end()) {
                try {
                    return *std::any_cast<std::shared_ptr<T>&>(it->second);
                } catch (const std::bad_any_cast&) {
                    throw std::runtime_error("Type mismatch for object: " + name);
                }
            } else {
                typedef std::shared_ptr<T> Tp;
                // create new object
                Tp ptr_try_load = make_object<T>();
                // try to load from file or create new object
                if (!try_load_from_file(key, *ptr_try_load))
                    ptr_try_load = make_object<T>(); // create new object if loading fails
                T& obj_ref = create_new_object<T>(key, ptr_try_load);
                // for a newly created object, register a function to save it to file
                register_saver<T>(key);
                return obj_ref;
            }
        }

        void save(mas_file& target) {
            if (file_) {
                if (!check_valid_save_path(target.getFilename()))
                    throw std::runtime_error("Error: invalid mas_file path.");
            }
            try {
                for (auto& saver : savers_) {
                    saver.second(target);
                }
                // The data without save function should also be saved to the new file.
                if (file_) {
                    for (auto& mas_item : file_->getHeads()) {
                        try {
                            // trivial copy data block to the new file
                            target.writeData(mas_item.info.blockName, file_->readData(mas_item.info.blockName));
                        } catch (const mas_duplicate_block_error& e) {
                            // ignore duplicate block error
                        }
                    }
                }
            } catch (const std::exception& e) {
                target.clear();
                std::cerr << e.what() << std::endl;
                throw std::runtime_error("Error: failed to save in " + target.getFilename());
            }
        }

        void save(std::filesystem::path file_path) {
            if (!check_valid_save_path(file_path))
                throw std::runtime_error("Error: invalid file path.");
            mas_file file(file_path, serialization::mas_file::replaceMode);
            save(file);
        }

        template<typename T>
        bool contains(const std::string& name) const {
            std::string prefix = type_name<T>();
            return objects_.find(prefix + "[" + name + "]") != objects_.end();
        }

        template<typename T>
        bool available_in_file(const std::string& name) const {
            assert_supported_type<T>();
            if (!file_) {
                return false;
            } else {
                auto tmp_ptr = make_object<T>();
                serialization::mas_file tmp_mas(file_path_);
                try {
                    std::string prefix = type_name<T>();
                    std::string key = prefix + "[" + name + "]";
                    load(tmp_mas, key, *tmp_ptr);
                    return true;
                } catch (const std::exception&) {
                    return false;
                }
            }
        }

    private:
        template<typename T>
        T& create_new_object(const std::string& name, std::shared_ptr<T> ptr) {
            auto [it, inserted] = objects_.try_emplace(name, std::make_any<std::shared_ptr<T>>(ptr));
            if (!inserted) {
                throw std::runtime_error("Failed to create object: " + name);
            }
            return *std::any_cast<std::shared_ptr<T>&>(it->second);
        }    
    
        // construct T from nothing when possible, else from the manager's queue
        template <typename T>
        requires std::is_default_constructible_v<T>
        std::shared_ptr<T> make_object() const {
            return std::make_shared<T>();
        }

        template <typename T>
        requires std::is_constructible_v<T, sycl::queue&> && (!std::is_default_constructible_v<T>)
        std::shared_ptr<T> make_object() const {
            if (!q_)
                throw std::runtime_error("Error: type needs a queue but object_manager has none: " + type_name<T>());
            return std::make_shared<T>(*q_);
        }

        template<typename T>
        bool try_load_from_file(const std::string& name, T& obj_ref) {
            if (!file_) {
                return false;
            }
            try {
                load(*file_, name, obj_ref);
                return true;
            } catch (const std::exception&) {
                return false;
            }
        }

        template<typename T>
        void register_saver(const std::string& name) {
            savers_[name] = [this, name](mas_file& file) {
                using serialization::save;
                try {
                    auto it = objects_.find(name);
                    if (it != objects_.end())
                        save(file, name, *std::any_cast<std::shared_ptr<T>&>(it->second));
                    else
                        throw std::runtime_error("Object missing: " + name);
                } catch (const std::exception& e) {
                    throw std::runtime_error("Failed to save object '" + name + "': " + e.what());
                }
            };
        }
    };

}

}

#endif
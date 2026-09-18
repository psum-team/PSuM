#ifndef PSUM_SERIALIZATION_MAS_HPP
#define PSUM_SERIALIZATION_MAS_HPP

#include <Eigen/Core>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "foundation.hpp"
#include <unordered_map>

// mas is short for Multi-Array-Set

// Control level for duplicate name checking:
// 0 = silent (only check same type)
// 1 = warn on any duplicate (default)
// 2 = throw exception on any duplicate
#ifndef MAS_STRICT_NAME_CHECKING
#define MAS_STRICT_NAME_CHECKING 2
#endif

namespace psum {

namespace serialization {

	struct mas_block_info {
		std::string blockName;
		std::string blockType;
		std::vector<size_t> blockShape;
		size_t blockSize;
	};

	struct mas_block {
		mas_block_info info;
		foundation::vPtr content;
	};

	struct mas_block_head {
		mas_block_info info;
		size_t position = 0;
	};

	class mas_duplicate_block_error : public std::runtime_error {
	public:
		mas_duplicate_block_error(const std::string& msg) : std::runtime_error(msg) {}
	};

	class mas_file {
		std::string filename;
		std::vector<mas_block_head> heads;
		std::unordered_map<std::string, std::vector<int>> headmap;
		const std::string& magic_string() const {
			static const std::string _magic_string = "#MAS!#";
			return _magic_string;
		}

	public:
		// autoMode: create a new file if it doesn't exist, otherwise open the existing file
		// replaceMode: clear the existing file and create a new one
		// quickMode: open the existing file without read block heads
		enum OpenMode {autoMode, replaceMode, quickMode};

		mas_file(const std::filesystem::path& file_path, OpenMode mode = autoMode) :filename(file_path) {
			bool fileExists = std::filesystem::exists(file_path);
			if (fileExists == false) {
				if (mode == autoMode || mode == replaceMode) {
					auto parentPath = file_path.parent_path();
					// Check if the parent directory exists
					if (!parentPath.empty() && !std::filesystem::exists(parentPath)) {
						// Create the parent directory
						if (!std::filesystem::create_directories(parentPath)) {
							throw std::runtime_error("Error: failed to create directories: " + parentPath.string());
						}
					}
					// Create an empty file
					std::ofstream ofs(filename);
					if (!ofs) {
						throw std::runtime_error("Error: failed to create file: " + filename);
					}
					ofs.close();
				}
				if (mode == quickMode) throw std::runtime_error("Error: file not found: " + filename + ". Use autoMode/replaceMode to create a new file.");
			} 
			if(mode == autoMode) readHead();
			else if(mode == replaceMode) clear();
			else if(mode == quickMode) {}
		};
		~mas_file() = default;

		// clear == write an empty file
		inline void clear() {
			std::ofstream ofs(filename);
			if (!ofs) {
				throw std::runtime_error("Error: failed to clear file: " + filename);
			}
			ofs.close();
			heads.clear();
			headmap.clear();
		}
		
		template <foundation::mas_fundamental T>
		inline void writeData(const std::string& key, const T* value_data, const std::vector<size_t>& shape) {
			// key can not be empty
			if (key.empty()) throw std::invalid_argument("Error: key can't be empty");
			// '!' is used to mask a block and valid key can't start with it
			if (key[0] == '!') throw std::invalid_argument("Error: key can't start with '!'");
			std::string typestr = foundation::basic_type_name<T>();
			if (foundation::synonym_cpp2matlab_map().find(typestr) != foundation::synonym_cpp2matlab_map().end())
				typestr = foundation::synonym_cpp2matlab_map().at(typestr);

			for (auto idx : headmap[key]) {
				if (MAS_STRICT_NAME_CHECKING > 0) {
					if (MAS_STRICT_NAME_CHECKING == 1) {
						std::cerr << "Warning: block '" << key << "' already exists with type "
						          << heads[idx].info.blockType << ", new type is " << typestr << std::endl;
					} else {
						throw mas_duplicate_block_error("Error: block already exists: " + key);
					}
				}
				if (heads[idx].info.blockType == typestr)
				throw mas_duplicate_block_error("Error: block already exists: " + key + " of type " + typestr);
			}

			std::ofstream ofs(filename, std::ios::binary | std::ios::app);
			if (!ofs) {
				throw std::runtime_error("Error: failed to open file in writeData: " + filename);
			}
			size_t position = ofs.tellp();

			// magic string
			ofs.write(magic_string().data(), magic_string().size());
			// block name
			int32_t buffer;
			buffer = key.length(); ofs.write((char *)&buffer, sizeof(int32_t));
			buffer = typestr.length(); ofs.write((char *)&buffer, sizeof(int32_t));
			buffer = shape.size(); ofs.write((char *)&buffer, sizeof(int32_t));
			ofs.write(key.data(), key.length());
			ofs.write(typestr.data(), typestr.length());

			size_t element_num = 1;
			for (auto i : shape) {
				size_t buffer = i;
				ofs.write((char *)&buffer, sizeof(size_t));
				element_num *= i;
			}

			ofs.write((char*)value_data, element_num * sizeof(T));

			// check ofs status
			if (!ofs) {
				throw std::runtime_error("Error: failed to write data to file: " + filename);
			}
			// update headmap
			mas_block_info info = { key, typestr, shape, element_num};
			heads.push_back({ info, position });
			headmap[key].push_back(heads.size() - 1);
			// close file
			ofs.close();
		}

		inline std::vector<mas_block_head>& readHead() {
		    heads.clear();
			headmap.clear();
			std::ifstream ifs(filename, std::ios::binary);
			if (!ifs) {
		        throw std::runtime_error("Error: failed to open file in readHead: " + filename);
		    }
			
			while (ifs.peek() != EOF) {
		        size_t position = ifs.tellg();
		        // check magic string
		        std::string magic(magic_string().size(), '\0');
		        ifs.read(magic.data(), magic.size());
				if (!ifs) throw std::runtime_error("Error: failed to read magic string.");
			
		        if (magic != magic_string()) {
		            throw std::runtime_error("Error: magic string mismatch.");
		        }
			
		        int32_t key_len = 0, type_len = 0, shape_len = 0;
		        ifs.read(reinterpret_cast<char*>(&key_len), sizeof(int32_t));
		        ifs.read(reinterpret_cast<char*>(&type_len), sizeof(int32_t));
		        ifs.read(reinterpret_cast<char*>(&shape_len), sizeof(int32_t));
		        if (!ifs) throw std::runtime_error("Error: failed to read block information.");
			
		        std::string key(key_len, '\0');
		        std::string typestr(type_len, '\0');
		        ifs.read(key.data(), key_len);
		        ifs.read(typestr.data(), type_len);
		        if (!ifs) throw std::runtime_error("Error: failed to read key/typestr");
			
		        std::vector<size_t> shape(shape_len);
		        size_t element_num = 1;
		        for (int32_t i = 0; i < shape_len; ++i) {
		            size_t dim = 0;
		            ifs.read(reinterpret_cast<char*>(&dim), sizeof(size_t));
		            if (!ifs) throw std::runtime_error("Error: failed to read shape");
		            shape[i] = dim;
		            element_num *= shape[i];
		        }

				foundation::vPtr tmp_pointer = foundation::vPtr_typestr2_setmem().at(foundation::synonym_matlab2cpp_map().at(typestr))(1);

				size_t typesize = std::visit(
					[](auto& i)->size_t { return sizeof(typename std::decay_t<decltype(i)>::element_type);},
					tmp_pointer
				);
				
		        size_t data_size = element_num * typesize;
		        ifs.seekg(data_size, std::ios::cur);

				if (key[0] != '!') {
					mas_block_info info{key, typestr, shape, element_num};
					heads.push_back({info, position});
					headmap[key].push_back(heads.size() - 1);	
				}

		        if (!ifs) throw std::runtime_error("Error: failed to skip data");
		    }
		
		    ifs.close();
			return heads;
		}

		inline bool smashData(const std::string& key, const std::string& typestr = "*") {
			bool found = false;
			heads.erase(
				std::remove_if(heads.begin(), heads.end(),
				[&](const mas_block_head& h) {
					if (key == h.info.blockName) {
						if (typestr == "*" || h.info.blockType == typestr) {
							std::fstream fs(filename, std::ios::in | std::ios::out | std::ios::binary);
							fs.seekp(h.position + magic_string().size() + 3 * sizeof(int32_t));
							char c = '!';
							fs.write(&c, 1);
							fs.close();
							found = true;
							return true;
						}
					}
					return false;
				}),
				heads.end()
			);
			headmap.clear();
			for (size_t i = 0; i < heads.size(); ++i) {
				headmap[heads[i].info.blockName].push_back(i);
			}
			return found;
		}

		template <foundation::mas_fundamental T>
		inline void replaceData(const std::string& name, T* data, std::vector<size_t> shape) {
			std::string typestr = foundation::basic_type_name<T>();
			if (foundation::synonym_cpp2matlab_map().find(typestr) != foundation::synonym_cpp2matlab_map().end())
				typestr = foundation::synonym_cpp2matlab_map().at(typestr);
			smashData(name, typestr);
			writeData<T>(name, data, shape);
		}

		inline mas_block readData(const std::string& varname, std::string typestr = "*") {
			if (headmap.find(varname) == headmap.end())
				readHead();
			if (headmap.find(varname) == headmap.end())
				return mas_block();
			else {
				std::ifstream ifs(filename, std::ios::binary);
				if (!ifs) {
					throw std::runtime_error("Error: failed to open file in readData: " + filename);
				}
				for (auto idx : headmap[varname]) {
					if (typestr == "*" || heads[idx].info.blockType == typestr) {
						ifs.seekg(heads[idx].position + magic_string().size() + 3 * sizeof(int32_t));
						mas_block_info info = heads[idx].info;
						ifs.seekg(info.blockName.length(), std::ios::cur);
						ifs.seekg(info.blockType.length(), std::ios::cur);
						ifs.seekg(info.blockShape.size() * sizeof(size_t), std::ios::cur);
						std::string typestr = info.blockType;
						typestr = foundation::synonym_matlab2cpp_map().at(typestr);
						mas_block ans;
						ans.info = info;
						ans.content = foundation::vPtr_typestr2_setmem().at(typestr)(info.blockSize);
						visit(
							[&ifs, &info, this](auto& v) {
								ifs.read((char*)v.get(), info.blockSize * sizeof(typename std::decay_t<decltype(v)>::element_type));
							}, 
							ans.content
						);
						ifs.close();
						return ans;
					}
				}
				ifs.close();
				return mas_block();
			}
		}

		inline mas_block readData(int index) {
			if (index < 0) {
				throw std::invalid_argument("Error: index should be greater than 0");
			}
			else if ((int)heads.size() <= index) {
				throw std::out_of_range("Error: index should be smaller than length of head");
			}
			else {
				return readData(heads[index].info.blockName, heads[index].info.blockType);
			}
		}

		template<typename T>
		static inline auto castData(mas_block&& in) {
			if (in.info.blockSize < 0 || in.info.blockShape.empty())
				throw std::invalid_argument(std::string("Error: casting 0-size data is invalid.") +
											" blockSize:" + std::to_string(in.info.blockSize) +
											" blockShape:" + std::to_string(in.info.blockShape.size()) +
											" blockName:" + in.info.blockName);
			if constexpr (std::is_base_of_v<
							  Eigen::MatrixBase<std::remove_cvref_t<T>>,
							  std::remove_cvref_t<T>>)
				return EigenCast<T>(std::move(in));
			else if constexpr (std::is_base_of_v<std::string, std::remove_cvref_t<T>>)
				return stringCast(std::move(in));
			else if (auto p = std::get_if<std::unique_ptr<T[]>>(&in.content)) {
				return std::unique_ptr<T[]>(std::move(*p));
			}
			else throw std::runtime_error("Error: type mismatch in Cast.");
		}

		template<typename T>
		static inline T EigenCast(mas_block&& in) {
			// in ColMajor
			Eigen::Matrix<typename T::Scalar, -1, -1, Eigen::ColMajor> ans;
			std::unique_ptr<typename T::Scalar[]> data = castData<typename T::Scalar>(std::move(in));
			ans.resize(in.info.blockShape[0], in.info.blockShape[1]);
			for(int i=0;i<in.info.blockShape[0]*in.info.blockShape[1];i++)
				ans.data()[i] = data[i];
			return ans;
		}

		static inline std::string stringCast(mas_block&& in) {
			std::string ans;
			std::unique_ptr<char[]> data = castData<char>(std::move(in));
			size_t length = 1;
			for(auto i: in.info.blockShape)
				length *= i;
			ans.resize(length);
			for(size_t i=0;i<length;i++)
				ans[i] = data[i];
			return ans;
		}

		template <typename ScalarType, int Rows, int Cols, int Opt>
		inline void writeData(const std::string& name, const Eigen::Matrix<ScalarType, Rows, Cols, Opt>& data) {
			// in ColMajor
			Eigen::Matrix<ScalarType, -1, -1, Eigen::ColMajor> data_t = data;
			writeData(name, data_t.data(), {(size_t)data.rows(), (size_t)data.cols()});
		}

		inline void writeData(const std::string& name, const std::string& data) {
			std::string buf = data;
			writeData<char>(name, buf.data(), {(size_t)buf.size(), 1});
		}

		inline void writeData(const std::string& name, const char* data) {
			std::string buf = data;
			writeData<char>(name, buf.data(), {(size_t)buf.size(), 1});
		}

		// single element writeData
		template <foundation::mas_fundamental T>
		inline void writeData(const std::string& name, T data) {
			T tmp = data;
			writeData(name, &tmp, {1,1});
		}

		// mas_block writeData
		// you can read so you can write
		inline void writeData(const std::string& name, const mas_block& data) {
			auto info = data.info;
			visit(
				[info, this](const auto& v) {
					writeData(info.blockName, v.get(), info.blockShape);
				}, 
				data.content
			);
		}

		template <foundation::mas_fundamental T>
		inline void replaceData(const std::string& name, T data) {
			std::string typestr = foundation::basic_type_name<T>();
			if (foundation::synonym_cpp2matlab_map().find(typestr) != foundation::synonym_cpp2matlab_map().end())
				typestr = foundation::synonym_cpp2matlab_map().at(typestr);
			smashData(name, typestr);
			writeData<T>(name, data);
		}

		std::string getFilename() const {
			return filename;
		}

		std::vector<mas_block_head>& getHeads() {
			return heads;
		}

		std::unordered_map<std::string, std::vector<int>>& getHeadmap() {
			return headmap;
		}

		size_t blockNum() const {
			return heads.size();
		}

	};

	template<typename T>
	inline decltype(auto) Cast(mas_block&& in) {
		return mas_file::castData<T>(std::move(in));
	}

	template <typename T>
    void save(mas_file& fp, const std::string& name, const T& in) {
		fp.writeData(name, in);
	}

	template <typename T>
    void load(mas_file& fp, const std::string& name, T& out) {
		if constexpr (std::is_fundamental_v<T>) {
			auto tmp = Cast<T>(fp.readData(name));
			out = tmp[0];
		}
		else
			out = Cast<T>(fp.readData(name));
	}

	template <typename T>
    decltype(auto) load(mas_file& fp, const std::string& name) {
		if constexpr (std::is_fundamental_v<T>) {
			auto tmp = Cast<T>(fp.readData(name));
			return T(tmp[0]);
		}
		else
			return Cast<T>(fp.readData(name));
	}
}

}

#endif

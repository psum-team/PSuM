#ifndef PSUM_PARTICLE_BOUNDARY_STL_LOADER_HPP
#define PSUM_PARTICLE_BOUNDARY_STL_LOADER_HPP

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include "geometry.hpp"

namespace psum {

namespace particle_boundary {

namespace stl_loader {

    using geometry::triangle_data;

    inline bool stl_file_is_ascii(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open STL file: " + filename);
        }
        
        char header[5] = {0};
        file.read(header, 5);
        bool is_ascii = (std::string(header, 5) == "solid");
        
        if (is_ascii) {
            file.seekg(0);
            char header[1000] = {0};
            file.read(header, 1000);
            std::string content(header);
            if (content.find("facet") == std::string::npos && 
                content.find("vertex") == std::string::npos) {
                file.close();
                return false;
            }
        }
        
        file.close();
        return is_ascii;
    }

    inline std::vector<triangle_data> load_stl_file_ascii(const std::string& filename) {
        std::vector<triangle_data> triangles;
        
        std::ifstream stl_file(filename);
        if (!stl_file) {
            throw std::runtime_error("Failed to open ASCII STL file: " + filename);
        }
        
        auto trim = [](std::string& s) {
            s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
                return !std::isspace(ch);
            }));
            s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
                return !std::isspace(ch);
            }).base(), s.end());
        };
        
        std::string line;
        std::getline(stl_file, line);
        
        while (std::getline(stl_file, line)) {
            trim(line);
            if (line.empty()) continue;
            
            if (line.substr(0, 8) == "endsolid") {
                break;
            }
            else if (line.substr(0, 12) == "facet normal") {
                std::istringstream normal_stream(line.substr(12));
                double nx, ny, nz;
                if (!(normal_stream >> nx >> ny >> nz)) {
                    throw std::runtime_error("Failed to parse normal vector");
                }
                
                std::getline(stl_file, line); 
                trim(line);
                if (line != "outer loop") {
                    throw std::runtime_error("Expected 'outer loop'");
                }
                
                double vertices[3][3];
                for (int i = 0; i < 3; ++i) {
                    std::getline(stl_file, line);
                    trim(line);
                    if (line.substr(0, 6) != "vertex") {
                        throw std::runtime_error("Expected vertex definition");
                    }
                    std::istringstream vertex_stream(line.substr(6));
                    double x, y, z;
                    if (!(vertex_stream >> x >> y >> z)) {
                        throw std::runtime_error("Failed to parse vertex");
                    }
                    vertices[i][0] = x;
                    vertices[i][1] = y;
                    vertices[i][2] = z;
                }
                
                std::getline(stl_file, line); 
                trim(line);
                if (line != "endloop") {
                    throw std::runtime_error("Expected 'endloop'");
                }
                
                std::getline(stl_file, line); 
                trim(line);
                if (line != "endfacet") {
                    throw std::runtime_error("Expected 'endfacet'");
                }
                
                triangle_data trig;
                double ax = vertices[1][0] - vertices[0][0];
                double ay = vertices[1][1] - vertices[0][1];
                double az = vertices[1][2] - vertices[0][2];
                double bx = vertices[2][0] - vertices[1][0];
                double by = vertices[2][1] - vertices[1][1];
                double bz = vertices[2][2] - vertices[1][2];
                
                double cross_x = ay * bz - az * by;
                double cross_y = az * bx - ax * bz;
                double cross_z = ax * by - ay * bx;
                
                double dot = nx * cross_x + ny * cross_y + nz * cross_z;
                
                if (dot > 0) {
                    trig.p1x = vertices[0][0]; trig.p1y = vertices[0][1]; trig.p1z = vertices[0][2];
                    trig.p2x = vertices[1][0]; trig.p2y = vertices[1][1]; trig.p2z = vertices[1][2];
                    trig.p3x = vertices[2][0]; trig.p3y = vertices[2][1]; trig.p3z = vertices[2][2];
                } else {
                    trig.p1x = vertices[2][0]; trig.p1y = vertices[2][1]; trig.p1z = vertices[2][2];
                    trig.p2x = vertices[1][0]; trig.p2y = vertices[1][1]; trig.p2z = vertices[1][2];
                    trig.p3x = vertices[0][0]; trig.p3y = vertices[0][1]; trig.p3z = vertices[0][2];
                }
                
                trig.nx = nx;
                trig.ny = ny;
                trig.nz = nz;
                
                triangles.push_back(trig);
            }
        }
        
        return triangles;
    }

    inline std::vector<triangle_data> load_stl_file_binary(const std::string& filename) {
        std::ifstream stl_file(filename, std::ios::binary);
        if (!stl_file) {
            throw std::runtime_error("Failed to open binary STL file: " + filename);
        }
        
        uint32_t length;
        stl_file.seekg(80);
        stl_file.read(reinterpret_cast<char*>(&length), sizeof(length));
        
        std::vector<triangle_data> triangles;
        triangles.resize(length);
        
        float nx, ny, nz;
        float p1x, p1y, p1z;
        float p2x, p2y, p2z;
        float p3x, p3y, p3z;
        uint16_t attribute_byte_count;
        
        for (uint32_t i = 0; i < length; ++i) {
            stl_file.read(reinterpret_cast<char*>(&nx), sizeof(nx));
            stl_file.read(reinterpret_cast<char*>(&ny), sizeof(ny));
            stl_file.read(reinterpret_cast<char*>(&nz), sizeof(nz));
            stl_file.read(reinterpret_cast<char*>(&p1x), sizeof(p1x));
            stl_file.read(reinterpret_cast<char*>(&p1y), sizeof(p1y));
            stl_file.read(reinterpret_cast<char*>(&p1z), sizeof(p1z));
            stl_file.read(reinterpret_cast<char*>(&p2x), sizeof(p2x));
            stl_file.read(reinterpret_cast<char*>(&p2y), sizeof(p2y));
            stl_file.read(reinterpret_cast<char*>(&p2z), sizeof(p2z));
            stl_file.read(reinterpret_cast<char*>(&p3x), sizeof(p3x));
            stl_file.read(reinterpret_cast<char*>(&p3y), sizeof(p3y));
            stl_file.read(reinterpret_cast<char*>(&p3z), sizeof(p3z));
            stl_file.read(reinterpret_cast<char*>(&attribute_byte_count), sizeof(attribute_byte_count));
            
            if (!stl_file.good()) {
                throw std::runtime_error("STL file might be incomplete.");
            }
            
            double ax = static_cast<double>(p2x) - static_cast<double>(p1x);
            double ay = static_cast<double>(p2y) - static_cast<double>(p1y);
            double az = static_cast<double>(p2z) - static_cast<double>(p1z);
            double bx = static_cast<double>(p3x) - static_cast<double>(p2x);
            double by = static_cast<double>(p3y) - static_cast<double>(p2y);
            double bz = static_cast<double>(p3z) - static_cast<double>(p2z);
            
            double cross_x = ay * bz - az * by;
            double cross_y = az * bx - ax * bz;
            double cross_z = ax * by - ay * bx;
            
            double dot = static_cast<double>(nx) * cross_x + 
                       static_cast<double>(ny) * cross_y + 
                       static_cast<double>(nz) * cross_z;
            
            triangle_data& trig = triangles[i];
            
            if (dot > 0) {
                trig.p1x = static_cast<double>(p1x);
                trig.p1y = static_cast<double>(p1y);
                trig.p1z = static_cast<double>(p1z);
                trig.p2x = static_cast<double>(p2x);
                trig.p2y = static_cast<double>(p2y);
                trig.p2z = static_cast<double>(p2z);
                trig.p3x = static_cast<double>(p3x);
                trig.p3y = static_cast<double>(p3y);
                trig.p3z = static_cast<double>(p3z);
            } else {
                trig.p1x = static_cast<double>(p3x);
                trig.p1y = static_cast<double>(p3y);
                trig.p1z = static_cast<double>(p3z);
                trig.p2x = static_cast<double>(p2x);
                trig.p2y = static_cast<double>(p2y);
                trig.p2z = static_cast<double>(p2z);
                trig.p3x = static_cast<double>(p1x);
                trig.p3y = static_cast<double>(p1y);
                trig.p3z = static_cast<double>(p1z);
            }
            
            trig.nx = static_cast<double>(nx);
            trig.ny = static_cast<double>(ny);
            trig.nz = static_cast<double>(nz);
        }
        
        stl_file.close();
        return triangles;
    }

    inline std::vector<triangle_data> load(const std::string& filename) {
        if (stl_file_is_ascii(filename)) {
            return load_stl_file_ascii(filename);
        } else {
            return load_stl_file_binary(filename);
        }
    }

    inline std::pair<std::array<double, 3>, std::array<double, 3>> box_of_triangles(std::vector<triangle_data>& triangles) {
        if (triangles.empty()) {
            throw std::runtime_error("No triangles in vector.");
        }
        double xmin = triangles[0].p1x, xmax = triangles[0].p1x;
        double ymin = triangles[0].p1y, ymax = triangles[0].p1y;
        double zmin = triangles[0].p1z, zmax = triangles[0].p1z;

        for (const auto& trig : triangles) {
            xmin = std::min({xmin, trig.p1x, trig.p2x, trig.p3x});
            xmax = std::max({xmax, trig.p1x, trig.p2x, trig.p3x});
            ymin = std::min({ymin, trig.p1y, trig.p2y, trig.p3y});
            ymax = std::max({ymax, trig.p1y, trig.p2y, trig.p3y});
            zmin = std::min({zmin, trig.p1z, trig.p2z, trig.p3z});
            zmax = std::max({zmax, trig.p1z, trig.p2z, trig.p3z});
        }

        return {{xmin, ymin, zmin}, {xmax, ymax, zmax}};
    }

}

}

}

#endif

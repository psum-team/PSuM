#ifndef PSUM_RUNTIME_SPECIES_HPP
#define PSUM_RUNTIME_SPECIES_HPP

#include <psum/psum.hpp>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace psum_runtime {

    using namespace psum::prelude;

    template <typename Universe, int Dim>
    struct species : psum::particle_container::particle_group<
        typename Universe::template particle<Dim>,
        psum::particle_container::pos_x_nan_is_invalid> {
        using base_t = psum::particle_container::particle_group<
            typename Universe::template particle<Dim>,
            psum::particle_container::pos_x_nan_is_invalid>;
        using base_t::base_t;

        static constexpr int dim_value = Dim;   // compile-time twin of `dim`

        int dim = Dim;
        std::optional<double> global_mass;
        std::optional<double> global_charge;
        std::optional<double> global_weight;
    };

    // metadata block layout, as doubles: [dim, has_m, m, has_c, c, has_w, w]
    template <typename Universe, int Dim>
    void save(serialization::mas_file& fp, const std::string& name, const species<Universe, Dim>& s) {
        save(fp, name + ".group", static_cast<const typename species<Universe, Dim>::base_t&>(s));
        const std::vector<double> meta{
            double(s.dim),
            s.global_mass ? 1.0 : 0.0, s.global_mass.value_or(0.0),
            s.global_charge ? 1.0 : 0.0, s.global_charge.value_or(0.0),
            s.global_weight ? 1.0 : 0.0, s.global_weight.value_or(0.0)};
        fp.template writeData<double>(name + ".meta", meta.data(), {meta.size()});
    }

    // load a species from "<name>.group" + "<name>.meta".
    template <typename Universe, int Dim>
    void load(serialization::mas_file& fp, const std::string& name,
              species<Universe, Dim>& s) {
        auto blk = fp.readData(name + ".meta");
        auto* holder = std::get_if<std::unique_ptr<double[]>>(&blk.content);
        if (!holder)
            throw std::runtime_error("species::load('" + name + "'): metadata block '" + name
                                     + ".meta' is missing or not of type double");
        const std::size_t n = blk.info.blockShape.empty() ? 0 : blk.info.blockShape[0];
        if (n != 7)
            throw std::runtime_error("species::load('" + name + "'): metadata block has "
                                     + std::to_string(n) + " entries, expected 7");

        const std::vector<double> meta(holder->get(), holder->get() + n);
        // the stored dim is the TYPE invariant: it must match the type being
        // loaded, otherwise the particles below belong to another species type
        const int stored_dim = int(meta[0]);
        if (stored_dim != Dim)
            throw std::runtime_error("species::load('" + name + "'): stored dim " + std::to_string(stored_dim)
                                     + " does not match species<Universe, " + std::to_string(Dim) + ">");

        // metadata is valid
        load(fp, name + ".group", static_cast<typename species<Universe, Dim>::base_t&>(s));

        s.dim = stored_dim;

        s.global_mass   = meta[1] ? std::optional<double>(meta[2]) : std::nullopt;
        s.global_charge = meta[3] ? std::optional<double>(meta[4]) : std::nullopt;
        s.global_weight = meta[5] ? std::optional<double>(meta[6]) : std::nullopt;
    }

} // namespace psum_runtime

#endif

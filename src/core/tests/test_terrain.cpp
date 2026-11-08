// Terrain sources, descriptors, tensor packs and the terrain correction
// (terrain_source.hpp, tensor_pack.hpp, terrain_correction.hpp).
//
// The grids are synthetic tiles laid out as GLO-30 lays its own (pixel is
// point, a tile's first row on its northern edge, its first column on its
// western edge) over a surface linear in latitude and longitude, on which
// bilinear interpolation, the slope stencil, the 1 km mean and the 1 km plane
// all have closed forms.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "resolve/tensor_pack.hpp"
#include "resolve/terrain_codec.hpp"
#include "resolve/terrain_correction.hpp"
#include "resolve/terrain_field.hpp"
#include "resolve/terrain_source.hpp"
#include "test_support.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <sstream>

using namespace resolve;
using namespace resolve::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr double kRad = std::numbers::pi / 180.0;
// The slope (degrees) of terrain_plane at latitude `lat`.
double plane_slope(double lat) {
    const double gy = 1000.0 / (kTerrainEarthRadiusM * kRad);
    const double gx = 500.0 / (kTerrainEarthRadiusM * kRad * std::cos(lat * kRad));
    return std::atan(std::sqrt(gx * gx + gy * gy)) / kRad;
}

torch::Tensor deg(const std::vector<double>& v) { return torch::tensor(v, torch::kFloat64); }

}  // namespace

TEST_CASE(".npy grids round-trip and refuse other layouts", "[terrain]") {
    TempDir dir;
    const torch::Tensor g = torch::arange(12, torch::kFloat32).reshape({3, 4}) * 1.5f;
    write_npy_grid(dir.file("g.npy"), g);
    CHECK(torch::equal(read_npy_grid(dir.file("g.npy")), g));

    std::string bytes = "\x93NUMPY\x01";
    bytes.push_back('\0');
    std::string header = "{'descr': '<f8', 'fortran_order': False, 'shape': (1, 1), }";
    header.append(64 - (10 + header.size() + 1) % 64, ' ');
    header.push_back('\n');
    bytes.push_back(static_cast<char>(header.size()));
    bytes.push_back('\0');
    dir.write("f8.npy", bytes + header + std::string(8, '\0'));
    CHECK_THROWS_AS(read_npy_grid(dir.file("f8.npy")), std::runtime_error);
    dir.write("junk.npy", "not a numpy file");
    CHECK_THROWS_AS(read_npy_grid(dir.file("junk.npy")), std::runtime_error);
}

TEST_CASE("Tensor packs hold named tensors of every dtype and read them one at a time", "[terrain]") {
    TempDir dir;
    json::Value meta = json::Value::object();
    meta.set("what", "test");
    const std::vector<std::pair<std::string, torch::Tensor>> tensors = {
        {"half", torch::linspace(-1, 1, 7).to(torch::kFloat16)},
        {"float", torch::randn({3, 5})},
        {"double", torch::randn({2}, torch::kFloat64)},
        {"short", torch::arange(-4, 4, torch::kInt16).reshape({2, 4})},
        {"int", torch::arange(5, torch::kInt32)},
        {"long", torch::arange(3, torch::kInt64)},
        {"byte", torch::arange(9, torch::kUInt8)},
        {"empty", torch::empty({0, 3})},
    };
    write_tensor_pack(dir.file("p.pack"), meta, tensors);
    const TensorPackReader pack(dir.file("p.pack"));
    CHECK(pack.meta().at("what").as_string() == "test");
    REQUIRE(pack.names().size() == tensors.size());
    for (const auto& [name, t] : tensors) {
        const torch::Tensor back = pack.read(name);
        CHECK(back.scalar_type() == t.scalar_type());
        CHECK(torch::equal(back, t));
    }
    CHECK_THROWS_AS(pack.read("missing"), std::out_of_range);
    // Several tensors at once, named out of file order, merged or read apart.
    const std::vector<std::string> some = {"long", "half", "empty", "float"};
    for (const uint64_t gap : {uint64_t{0}, uint64_t{1} << 20}) {
        const std::vector<torch::Tensor> many = pack.read_many(some, gap);
        REQUIRE(many.size() == some.size());
        for (size_t i = 0; i < some.size(); ++i) CHECK(torch::equal(many[i], pack.read(some[i])));
    }
    CHECK_THROWS_AS(pack.read_many({"float", "missing"}, 0), std::out_of_range);
    unsigned char bytes[3] = {0, 0, 0};
    pack.read_bytes("byte", 4, 3, bytes);
    CHECK((bytes[0] == 4 && bytes[1] == 5 && bytes[2] == 6));
    CHECK_THROWS_AS(pack.read_bytes("byte", 7, 3, bytes), std::out_of_range);
    CHECK_THROWS_AS(write_tensor_pack(dir.file("q.pack"), meta, {{"a", torch::ones(1)}, {"a", torch::ones(1)}}),
                    std::invalid_argument);
    dir.write("bad.pack", "RESOLVEPACK?xxxxxxxxxxxxxxxx");
    CHECK_THROWS_AS(TensorPackReader(dir.file("bad.pack")), std::runtime_error);
}

TEST_CASE("The tile index finds tiles and moves cells across tile edges", "[terrain]") {
    TempDir dir;
    write_plane_terrain(dir, {{46, 10}, {45, 10}, {46, 11}});
    const TerrainGrids grids(dir.path());
    const auto& index = grids.index();
    REQUIRE(index.size() == 3);
    CHECK(index.find(46, 10) >= 0);
    CHECK(index.find(45, 11) == -1);

    const double d = 1.0 / 60.0;
    // A point a third of a cell above 46 N is nearest the first row of the
    // tile on 45 N (its northern edge sits at 46 N).
    const TerrainCells near = index.nearest_cells(deg({46.0 + d / 3.0}), deg({10.5}));
    CHECK(near.tile.item<int64_t>() == index.find(45, 10));
    CHECK(near.row.item<int64_t>() == 0);
    // The cell east of tile (46, 10)'s last column is tile (46, 11)'s first.
    const TerrainCells moved = index.normalise(torch::tensor({index.find(46, 10)}), torch::tensor({int64_t{5}}),
                                               torch::tensor({int64_t{60}}));
    CHECK(moved.tile.item<int64_t>() == index.find(46, 11));
    CHECK(moved.col.item<int64_t>() == 0);
    // ... and the one north of the top row is in no tile.
    const TerrainCells off = index.normalise(torch::tensor({index.find(46, 10)}), torch::tensor({int64_t{-1}}),
                                             torch::tensor({int64_t{3}}));
    CHECK(off.tile.item<int64_t>() == -1);
}

TEST_CASE("Grids interpolate a plane exactly, across tile edges too, and hold nothing outside their tiles",
          "[terrain]") {
    TempDir dir;
    write_plane_terrain(dir, {{46, 10}, {45, 10}, {46, 11}, {45, 11}});
    const TerrainGrids grids(dir.path(), 2);
    auto gen = at::make_generator<at::CPUGeneratorImpl>(7);
    const torch::Tensor lat = 45.02 + 1.96 * torch::rand({500}, gen, torch::kFloat64);
    const torch::Tensor lon = 10.02 + 1.96 * torch::rand({500}, gen, torch::kFloat64);
    const torch::Tensor z = grids.elevation(lat, lon);
    double worst = 0.0;
    for (int64_t i = 0; i < 500; ++i) {
        worst = std::max(worst, std::abs(z[i].item<double>() - terrain_plane(lat[i].item<double>(), lon[i].item<double>())));
    }
    CHECK(worst < 1e-3);  // float32 cells of a plane up to ~1600 m

    const double d = 1.0 / 60.0;
    // Between the last row of the northern tiles and the first of the southern.
    const double strip = 46.0 + d / 4.0;
    CHECK_THAT(grids.elevation(deg({strip}), deg({10.7})).item<double>(), WithinAbs(terrain_plane(strip, 10.7), 1e-3));
    CHECK(std::isnan(grids.elevation(deg({50.5}), deg({10.5})).item<double>()));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    CHECK(std::isnan(grids.elevation(deg({nan}), deg({10.5})).item<double>()));
}

TEST_CASE("Terrain descriptors of a plane have their closed forms", "[terrain]") {
    TempDir dir;
    write_plane_terrain(dir, {{46, 10}, {45, 10}, {46, 11}, {45, 11}});
    const TerrainGrids grids(dir.path());
    const torch::Tensor lat = deg({46.31, 45.77, 46.0});
    const torch::Tensor lon = deg({10.42, 10.95, 11.0});
    const TerrainDescriptors t = terrain_descriptors(grids, lat, lon, 2);
    for (int64_t i = 0; i < 3; ++i) {
        const double la = lat[i].item<double>();
        const double lo = lon[i].item<double>();
        CHECK_THAT(t.elevation[i].item<double>(), WithinAbs(terrain_plane(la, lo), 1e-3));
        CHECK_THAT(t.elevation_1km[i].item<double>(), WithinAbs(terrain_plane(la, lo), 1e-3));
        // The stencil reads the nearest cell, whose latitude is within half a
        // cell of the point's.
        CHECK_THAT(t.slope[i].item<double>(), WithinAbs(plane_slope(la), 1e-3));
        CHECK_THAT(t.slope_1km[i].item<double>(), WithinAbs(plane_slope(la), 1e-4));
    }
    const TerrainDescriptors out = terrain_descriptors(grids, deg({48.5}), deg({10.5}));
    CHECK(std::isnan(out.elevation.item<double>()));
    CHECK(std::isnan(out.elevation_1km.item<double>()));
    CHECK(std::isnan(out.slope.item<double>()));
    CHECK(std::isnan(out.slope_1km.item<double>()));
}

TEST_CASE("Terrain descriptors come back in the caller's order across tiles", "[terrain]") {
    TempDir dir;
    write_plane_terrain(dir, {{46, 10}, {45, 10}, {46, 11}, {45, 11}});
    const TerrainGrids grids(dir.path(), 1);
    // Tiles interleaved, with a point outside every tile among them.
    const torch::Tensor lat = deg({45.2, 46.7, 48.5, 45.6, 46.1, 45.9, 46.4});
    const torch::Tensor lon = deg({11.3, 10.2, 10.5, 10.6, 11.8, 11.1, 10.9});
    const auto fields = [](const TerrainDescriptors& d) {
        return std::vector<torch::Tensor>{d.elevation, d.elevation_1km, d.slope, d.slope_1km};
    };
    for (const int64_t chunk : {int64_t{1}, int64_t{3}, int64_t{4096}}) {
        const auto all = fields(terrain_descriptors(grids, lat, lon, chunk));
        for (int64_t i = 0; i < lat.numel(); ++i) {
            const auto one = fields(terrain_descriptors(grids, lat.narrow(0, i, 1), lon.narrow(0, i, 1)));
            for (size_t f = 0; f < one.size(); ++f) {
                CHECK(torch::allclose(all[f].narrow(0, i, 1), one[f], 1e-12, 1e-9, /*equal_nan=*/true));
            }
        }
        CHECK(std::isnan(all[0][2].item<double>()));
        CHECK_THAT(all[0][0].item<double>(), WithinAbs(terrain_plane(45.2, 11.3), 1e-3));
    }
}

namespace {

// Plots whose elevation model is right to a few metres where the coordinate
// is precise and wrong by hundreds of metres for a share of the plots that
// grows with the recorded uncertainty, and whose prediction errs by tens of
// metres throughout, more where the members disagree.
struct TerrainCorpus {
    TerrainCorrectionInputs inputs;
    torch::Tensor truth;
};

TerrainCorpus terrain_corpus(int64_t n, uint64_t seed) {
    auto gen = at::make_generator<at::CPUGeneratorImpl>(seed);
    const torch::Tensor truth = 200.0 + 1500.0 * torch::rand({n}, gen, torch::kFloat64);
    const torch::Tensor log_u = 3.5 * torch::rand({n}, gen, torch::kFloat64);  // 1 m .. 3 km
    const torch::Tensor u = torch::pow(10.0, log_u);
    const torch::Tensor wrong = torch::rand({n}, gen, torch::kFloat64) < (0.05 + 0.15 * log_u);
    const torch::Tensor dem = truth + 2.0 * torch::randn({n}, gen, torch::kFloat64) +
                              torch::where(wrong, 300.0 * torch::randn({n}, gen, torch::kFloat64),
                                           torch::zeros({n}, torch::kFloat64));
    const torch::Tensor dem_1km = dem + 15.0 * torch::randn({n}, gen, torch::kFloat64);
    const torch::Tensor spread = 5.0 + 40.0 * torch::rand({n}, gen, torch::kFloat64);
    const torch::Tensor prediction = truth + spread * torch::randn({n}, gen, torch::kFloat64);
    TerrainCorpus c;
    c.inputs.prediction = prediction;
    c.inputs.dem = dem;
    c.inputs.dem_1km = dem_1km;
    c.inputs.uncertainty_m = u;
    c.inputs.spread = spread;
    c.inputs.recognised_share = torch::full({n}, 0.9, torch::kFloat64);
    c.inputs.n_species = torch::full({n}, 25.0, torch::kFloat64);
    c.truth = truth;
    return c;
}

double mae(const torch::Tensor& a, const torch::Tensor& b) { return (a - b).abs().mean().item<double>(); }

TerrainCorrectionConfig quick_config(int seed) {
    TerrainCorrectionConfig c;
    c.scale = 500.0f;
    // About as many optimizer steps as the default config takes over a
    // continent's plots (12 epochs of a few hundred thousand plots).
    c.epochs = 80;
    c.batch_size = 256;
    c.hidden_dim = 64;
    c.seed = seed;
    return c;
}

}  // namespace

TEST_CASE("Terrain correction learns which source to trust, over seeds", "[terrain][recovery]") {
    for (const int seed : {1, 2, 3}) {
        const TerrainCorpus train = terrain_corpus(6000, 100 + seed);
        const TerrainCorpus test = terrain_corpus(3000, 200 + seed);
        const TerrainCorrection model =
            fit_terrain_correction("altitude", TerrainDescriptor::Elevation, train.inputs, train.truth, quick_config(seed));
        const TerrainCorrectionResult r = model.apply(test.inputs);
        const double raw = mae(test.inputs.prediction, test.truth);
        const double dem = mae(test.inputs.dem, test.truth);
        const double corrected = mae(r.corrected, test.truth);
        INFO("seed " << seed << ": raw " << raw << ", dem " << dem << ", corrected " << corrected);
        CHECK(corrected < 0.5 * std::min(raw, dem));
        // Where the coordinate is precise the elevation model is right, and the
        // correction leans on it.
        const torch::Tensor precise = test.inputs.uncertainty_m < 3.0;
        REQUIRE(precise.any().item<bool>());
        CHECK(r.weights.index({precise}).select(1, 1).mean().item<double>() > 0.6);
        CHECK(r.applied.all().item<bool>());
        CHECK(model.report().mae_corrected < model.report().mae_raw);
    }
}

TEST_CASE("Terrain correction keeps the raw prediction where it cannot apply, reads point values alone, and "
          "round-trips through its file",
          "[terrain]") {
    const TerrainCorpus train = terrain_corpus(3000, 11);
    TerrainCorrectionConfig config = quick_config(5);
    config.epochs = 4;
    const TerrainCorrection model =
        fit_terrain_correction("altitude", TerrainDescriptor::Elevation, train.inputs, train.truth, config);

    TerrainCorpus test = terrain_corpus(50, 12);
    test.inputs.dem[3] = std::numeric_limits<double>::quiet_NaN();
    test.inputs.prediction[7] = std::numeric_limits<double>::quiet_NaN();
    const TerrainCorrectionResult r = model.apply(test.inputs);
    CHECK_FALSE(r.applied[3].item<bool>());
    CHECK(r.corrected[3].item<double>() == test.inputs.prediction[3].item<double>());
    CHECK(std::isnan(r.weights[3][0].item<double>()));
    CHECK_FALSE(r.applied[7].item<bool>());
    CHECK(r.applied.sum().item<int64_t>() == 48);
    CHECK_THAT(r.weights.index({r.applied}).sum(1).min().item<double>(), WithinAbs(1.0, 1e-6));

    TerrainCorrectionInputs point_only = test.inputs;
    point_only.dem_1km = torch::Tensor();
    point_only.spread = torch::Tensor();
    point_only.uncertainty_m = torch::Tensor();
    const TerrainCorrectionResult p = model.apply(point_only);
    CHECK(p.applied.sum().item<int64_t>() == 48);
    CHECK(torch::isfinite(p.corrected.index({p.applied})).all().item<bool>());

    TempDir dir;
    model.save(dir.file("altitude.terrain"));
    const TerrainCorrection back = TerrainCorrection::load(dir.file("altitude.terrain"));
    CHECK(back.target() == "altitude");
    CHECK(back.descriptor() == TerrainDescriptor::Elevation);
    CHECK(back.config().epochs == 4);
    CHECK(back.report().n_train == model.report().n_train);
    const TerrainCorrectionResult rb = back.apply(test.inputs);
    // The network runs in float32; plot 7 has no prediction, so NaN.
    CHECK(torch::allclose(rb.corrected, r.corrected, 1e-6, 1e-4, /*equal_nan=*/true));
    CHECK(torch::equal(rb.applied, r.applied));
}

TEST_CASE("Terrain correction refuses invalid configs and inputs", "[terrain]") {
    const TerrainCorpus c = terrain_corpus(20, 3);
    TerrainCorrectionConfig bad;
    bad.scale = 0.0f;
    CHECK_THROWS_AS(fit_terrain_correction("altitude", TerrainDescriptor::Elevation, c.inputs, c.truth, bad),
                    std::invalid_argument);
    TerrainCorrectionInputs short_dem = c.inputs;
    short_dem.dem = c.inputs.dem.narrow(0, 0, 10);
    CHECK_THROWS_AS(fit_terrain_correction("altitude", TerrainDescriptor::Elevation, short_dem, c.truth),
                    std::invalid_argument);
    const torch::Tensor none = torch::full({20}, std::numeric_limits<double>::quiet_NaN(), torch::kFloat64);
    CHECK_THROWS_AS(fit_terrain_correction("altitude", TerrainDescriptor::Elevation, c.inputs, none),
                    std::invalid_argument);
}

namespace {

// A smooth relief over one degree square: hills a few hundred metres high on a
// gentle tilt, sampled as a pixel-is-point tile of `cells` x `cells`.
torch::Tensor relief_grid(const TerrainTile& t) {
    const torch::Tensor r = torch::arange(t.rows, torch::kFloat64).view({-1, 1});
    const torch::Tensor c = torch::arange(t.cols, torch::kFloat64).view({1, -1});
    const torch::Tensor lat = t.lat_first - r * t.dlat;
    const torch::Tensor lon = t.lon_first + c * t.dlon;
    const double two_pi = 2.0 * std::numbers::pi;
    return (600.0 + 300.0 * torch::sin(two_pi * 2.0 * (lat - 46.0)) * torch::cos(two_pi * 3.0 * (lon - 10.0)) +
            80.0 * torch::sin(two_pi * 7.0 * (lon - 10.0) + 1.0) + 200.0 * (lon - 10.0))
        .to(torch::kFloat32);
}

TerrainTile relief_tile(int64_t cells) {
    TerrainTile t;
    t.name = "R46_10";
    t.rows = cells;
    t.cols = cells;
    t.dlat = 1.0 / static_cast<double>(cells);
    t.dlon = 1.0 / static_cast<double>(cells);
    t.lat_first = 47.0;
    t.lon_first = 10.0;
    return t;
}

TerrainFieldConfig small_field_config() {
    TerrainFieldConfig c;
    c.levels = 8;
    c.log2_table_size = 12;
    c.features = 2;
    c.coarsest_resolution = 8;
    c.hidden_dim = 32;
    c.base_cells = 8;
    c.steps = 1500;
    c.batch_size = 8192;
    c.lr = 1e-2f;
    c.gradient_weight = 0.5f;
    return c;
}

}  // namespace

TEST_CASE("A terrain tile field reproduces a smooth relief and round-trips through the field file",
          "[terrain][recovery]") {
    const TerrainTile tile = relief_tile(240);
    const torch::Tensor grid = relief_grid(tile);
    TempDir dir;
    const TerrainTileMetrics m = fit_terrain_tile(grid, tile, small_field_config(), dir.file("R46_10.tile"));
    INFO("rmse " << m.rmse << " m, max " << m.max_abs << " m, slope mae " << m.slope_mae << " deg");
    // The relief spans about 900 m; the field holds it to a few metres.
    CHECK(m.rmse < 5.0);
    CHECK(m.slope_mae < 1.0);
    CHECK(m.bytes > 0);

    assemble_terrain_field({dir.file("R46_10.tile")}, dir.file("field.bin"));
    const TerrainField field(dir.file("field.bin"));
    REQUIRE(field.index().size() == 1);
    CHECK(field.index().tiles()[0] == tile);
    CHECK(field.metrics()[0].rmse == m.rmse);

    // The field's cells against the grid: the recorded RMSE, recomputed.
    const torch::Tensor rows = torch::arange(tile.rows).repeat_interleave(tile.cols);
    const torch::Tensor cols = torch::arange(tile.cols).repeat({tile.rows});
    const torch::Tensor z = field.cell_values({torch::zeros_like(rows), rows, cols});
    const double rmse = (z - grid.reshape({-1}).to(torch::kFloat64)).pow(2).mean().sqrt().item<double>();
    CHECK_THAT(rmse, WithinRel(m.rmse, 1e-3));
    // The whole tile through the source interface: the same cells.
    const TerrainSource& source = field;
    CHECK(torch::equal(source.grid(0), z.to(torch::kFloat32).view({tile.rows, tile.cols})));
    CHECK_THROWS_AS(source.grid(1), std::out_of_range);
    CHECK(std::isnan(field.elevation(deg({40.5}), deg({10.5})).item<double>()));
}

TEST_CASE("Descriptors read from a terrain field follow those read from its grids", "[terrain][recovery]") {
    const TerrainTile tile = relief_tile(240);
    const torch::Tensor grid = relief_grid(tile);
    TempDir dir;
    write_npy_grid(dir.file("R46_10.npy"), grid);
    std::ostringstream index;
    index.precision(17);
    index << R"({"format": "resolve-terrain-grids", "version": 1, "tiles": [{"name": "R46_10", "file": "R46_10.npy", )"
          << R"("rows": 240, "cols": 240, "lat_first": 47, "lon_first": 10, "dlat": )" << tile.dlat
          << R"(, "dlon": )" << tile.dlon << "}]}";
    dir.write("tiles.json", index.str());
    const TerrainGrids grids(dir.path());
    fit_terrain_tile(grid, tile, small_field_config(), dir.file("R46_10.tile"));
    assemble_terrain_field({dir.file("R46_10.tile")}, dir.file("field.bin"));
    const TerrainField field(dir.file("field.bin"));

    auto gen = at::make_generator<at::CPUGeneratorImpl>(3);
    const torch::Tensor lat = 46.05 + 0.9 * torch::rand({200}, gen, torch::kFloat64);
    const torch::Tensor lon = 10.05 + 0.9 * torch::rand({200}, gen, torch::kFloat64);
    const TerrainDescriptors g = terrain_descriptors(grids, lat, lon);
    const TerrainDescriptors f = terrain_descriptors(field, lat, lon);
    CHECK((f.elevation - g.elevation).abs().mean().item<double>() < 5.0);
    CHECK((f.elevation_1km - g.elevation_1km).abs().mean().item<double>() < 5.0);
    CHECK((f.slope - g.slope).abs().mean().item<double>() < 1.0);
    CHECK((f.slope_1km - g.slope_1km).abs().mean().item<double>() < 0.5);
}

TEST_CASE("Assembling a terrain field refuses mixed configs and repeated tiles", "[terrain]") {
    const TerrainTile tile = relief_tile(64);
    const torch::Tensor grid = relief_grid(tile);
    TempDir dir;
    TerrainFieldConfig a = small_field_config();
    a.steps = 20;
    TerrainFieldConfig b = a;
    b.hidden_dim = 16;
    fit_terrain_tile(grid, tile, a, dir.file("a.tile"));
    fit_terrain_tile(grid, tile, b, dir.file("b.tile"));
    CHECK_THROWS_AS(assemble_terrain_field({dir.file("a.tile"), dir.file("b.tile")}, dir.file("f.bin")),
                    std::runtime_error);
    CHECK_THROWS_AS(assemble_terrain_field({dir.file("a.tile"), dir.file("a.tile")}, dir.file("f.bin")),
                    std::runtime_error);
    TerrainFieldConfig bad = a;
    bad.levels = 0;
    CHECK_THROWS_AS(fit_terrain_tile(grid, tile, bad, dir.file("c.tile")), std::invalid_argument);
}

TEST_CASE("The terrain codec keeps every cell within its error bound, cliffs and flat ground included",
          "[terrain]") {
    const TerrainTile tile = relief_tile(130);  // not a multiple of the block: edge blocks
    torch::Tensor grid = relief_grid(tile);
    // A cliff of 3 km (escape codes) and a flat lake (run mode).
    grid.narrow(0, 20, 30).narrow(1, 40, 30).add_(3000.0f);
    grid.narrow(0, 80, 40).narrow(1, 0, 60).fill_(412.0f);
    for (const double max_error : {0.5, 2.0}) {
        const TerrainTileCode code = encode_terrain_tile(grid, 2.0 * max_error, 32);
        const torch::Tensor back = decode_terrain_tile(code, tile.rows, tile.cols, 2.0 * max_error, 32);
        CHECK((back - grid).abs().max().item<double>() <= max_error + 1e-3);
        CHECK(code.offsets.numel() == 5 * 5 + 1);
    }
    const TerrainTileCode coarse = encode_terrain_tile(grid, 4.0, 32);
    const TerrainTileCode fine = encode_terrain_tile(grid, 1.0, 32);
    CHECK(coarse.stream.numel() < fine.stream.numel());
    CHECK_THROWS_AS(encode_terrain_tile(grid, 0.0, 32), std::invalid_argument);
    TerrainTileCode broken = coarse;
    broken.offsets = broken.offsets.narrow(0, 0, 3);
    CHECK_THROWS_AS(decode_terrain_tile(broken, tile.rows, tile.cols, 4.0, 32), std::runtime_error);
}

TEST_CASE("A coded terrain field reads like its grids and opens through the format-detecting opener", "[terrain]") {
    TempDir dir;
    write_plane_terrain(dir, {{46, 9}, {46, 10}, {45, 9}, {45, 10}}, 60);
    const TerrainGrids grids(dir.path());
    TerrainCodecConfig config;
    config.max_error = 1.0f;
    config.block = 16;
    const auto metrics = encode_terrain_grids(grids, config, dir.file("coded.bin"));
    REQUIRE(metrics.size() == 4);
    for (const auto& [name, m] : metrics) {
        INFO(name);
        CHECK(m.max_abs <= 1.0 + 1e-3);
        CHECK(m.bytes > 0);
    }
    const std::shared_ptr<TerrainSource> coded = open_terrain_field(dir.file("coded.bin"));
    REQUIRE(dynamic_cast<const TerrainCodedField*>(coded.get()) != nullptr);
    CHECK(coded->index().size() == 4);
    const torch::Tensor lat = deg({46.2, 45.9, 46.0});
    const torch::Tensor lon = deg({9.3, 10.05, 10.0});
    const TerrainDescriptors c = terrain_descriptors(*coded, lat, lon);
    const TerrainDescriptors g = terrain_descriptors(grids, lat, lon);
    CHECK((c.elevation - g.elevation).abs().max().item<double>() <= 1.0 + 1e-3);
    CHECK((c.elevation_1km - g.elevation_1km).abs().max().item<double>() <= 1.0 + 1e-3);
    CHECK_THROWS_AS(encode_terrain_grids(grids, config, dir.file("none.bin"), {"no_such_tile"}),
                    std::invalid_argument);
    write_tensor_pack(dir.file("other.pack"), json::parse(R"({"format": "something-else"})"), {});
    CHECK_THROWS_AS(open_terrain_field(dir.file("other.pack")), std::runtime_error);
}

TEST_CASE("A coded terrain field's cells read block by block equal its decoded tiles", "[terrain]") {
    TempDir dir;
    write_plane_terrain(dir, {{46, 9}, {46, 10}}, 70);
    const TerrainGrids grids(dir.path());
    TerrainCodecConfig config;
    config.max_error = 0.5f;
    config.block = 16;
    (void)encode_terrain_grids(grids, config, dir.file("coded.bin"));
    const TerrainCodedField roomy(dir.file("coded.bin"));
    // One block kept between queries: every query reads and decodes again.
    const TerrainCodedField tight(dir.file("coded.bin"), 1);
    for (int64_t t = 0; t < 2; ++t) {
        const torch::Tensor whole = roomy.grid(t);
        const int64_t rows = whole.size(0);
        const int64_t cols = whole.size(1);
        // Cells in every block (the last row and column of blocks are partial:
        // 70 is not a multiple of 16), alternating tiles and blocks.
        std::vector<int64_t> ts;
        std::vector<int64_t> rs;
        std::vector<int64_t> cs;
        for (int64_t r = 0; r < rows; r += 7) {
            for (int64_t c = 0; c < cols; c += 5) {
                ts.push_back(t);
                rs.push_back(r);
                cs.push_back(c);
                ts.push_back(1 - t);
                rs.push_back(rows - 1 - r);
                cs.push_back(cols - 1 - c);
            }
        }
        const TerrainCells cells{torch::tensor(ts, torch::kInt64), torch::tensor(rs, torch::kInt64),
                                 torch::tensor(cs, torch::kInt64)};
        const torch::Tensor other = roomy.grid(1 - t);
        const torch::Tensor expected =
            torch::where(cells.tile == t, whole.index({cells.row, cells.col}), other.index({cells.row, cells.col}))
                .to(torch::kFloat64);
        CHECK(torch::equal(roomy.cell_values(cells), expected));
        CHECK(torch::equal(tight.cell_values(cells), expected));
        const int64_t last = ((rows + 15) / 16) * ((cols + 15) / 16) - 1;
        const torch::Tensor corner = tight.block(t, last);
        CHECK(corner.size(0) == rows - 16 * ((rows - 1) / 16));
        CHECK(torch::equal(corner, whole.index({torch::indexing::Slice(rows - corner.size(0)),
                                                 torch::indexing::Slice(cols - corner.size(1))})));
        CHECK_THROWS_AS(tight.block(t, last + 1), std::out_of_range);
    }
    CHECK_THROWS_AS(TerrainCodedField(dir.file("coded.bin"), 0), std::invalid_argument);
    const TerrainCells outside{torch::tensor({int64_t{0}}), torch::tensor({int64_t{70}}), torch::tensor({int64_t{3}})};
    CHECK_THROWS_AS(roomy.cell_values(outside), std::out_of_range);
}

// Tests for name resolution (name_resolution.hpp): the routes, their order,
// the spelling step's ties and genus bucket, the name table file, and the
// loader reading resolved names.

#include <catch2/catch_test_macros.hpp>

#include "resolve/dataset.hpp"
#include "resolve/enum_names.hpp"
#include "resolve/name_resolution.hpp"
#include "resolve/sha256.hpp"
#include "test_support.hpp"

#include <memory>
#include <string>
#include <vector>

using namespace resolve;
using resolve::test::Roles;

// A failing route check prints the route's name.
template <>
struct Catch::StringMaker<NameRoute> {
    static std::string convert(NameRoute route) { return enum_to_name(route); }
};

using resolve::test::TempFile;

namespace {

// A checkpoint-ordered vocabulary: [0] is the reserved unknown code.
std::vector<std::string> vocabulary(std::vector<std::string> tokens) {
    tokens.insert(tokens.begin(), "<UNK>");
    return tokens;
}

std::shared_ptr<const NameTable> table(const std::vector<std::string>& names,
                                       const std::vector<std::string>& tokens,
                                       const std::vector<std::string>& sources) {
    return std::make_shared<const NameTable>(NameTable::from_columns(names, tokens, sources));
}

NameResolverOptions edits(int k) {
    NameResolverOptions options;
    options.max_edit_distance = k;
    return options;
}

}  // namespace

TEST_CASE("each name takes the first route that reaches a token", "[names]") {
    const auto vocab = vocabulary({"Anemone altaica", "Carex curvicollis", "Quercus robur"});
    const auto names = table({"Anemone nemorosa", "Anemonoides nemorosa", "Carex viridula"},
                             {"Anemone altaica", "Anemone altaica", "Carex curvicollis"},
                             {"eva", "wfo", "eva"});
    const NameResolver resolver(vocab, names);

    const ResolvedName exact = resolver.resolve("Quercus robur");
    CHECK(exact.route == NameRoute::Exact);
    CHECK(exact.token == "Quercus robur");

    const ResolvedName alias = resolver.resolve("Anemonoides nemorosa");
    CHECK(alias.route == NameRoute::Alias);
    CHECK(alias.token == "Anemone altaica");
    CHECK(alias.source == "wfo");

    const ResolvedName typo = resolver.resolve("Quercus robru");
    CHECK(typo.route == NameRoute::Spelling);
    CHECK(typo.token == "Quercus robur");
    CHECK(typo.via == "Quercus robur");
    CHECK(typo.source.empty());
    CHECK(typo.distance == 1);

    // A misspelled table name resolves through the table name's token.
    const ResolvedName typo_alias = resolver.resolve("Carex viridul");
    CHECK(typo_alias.route == NameRoute::Spelling);
    CHECK(typo_alias.token == "Carex curvicollis");
    CHECK(typo_alias.via == "Carex viridula");
    CHECK(typo_alias.source == "eva");

    const ResolvedName unseen = resolver.resolve("Tragopogon dubius");
    CHECK(unseen.route == NameRoute::Unresolved);
    CHECK(unseen.token == "Tragopogon dubius");

    CHECK(resolver.resolve("").route == NameRoute::Unresolved);
}

TEST_CASE("exact wins over the table, and letter case alone is a spelling at distance 0", "[names]") {
    const auto vocab = vocabulary({"Festuca rubra", "Festuca ovina"});
    const NameResolver resolver(vocab, table({"Festuca rubra"}, {"Festuca ovina"}, {"wfo"}));
    CHECK(resolver.resolve("Festuca rubra").route == NameRoute::Exact);

    const ResolvedName cased = resolver.resolve("festuca rubra");
    CHECK(cased.route == NameRoute::Spelling);
    CHECK(cased.token == "Festuca rubra");
    CHECK(cased.distance == 0);
}

TEST_CASE("an alias whose token the vocabulary lacks is no route", "[names]") {
    const auto vocab = vocabulary({"Carex nigra"});
    const NameResolver resolver(vocab, table({"Carex goodenowii", "Carex viridula"},
                                             {"Carex nigra", "Carex curvicollis"}, {"wfo", "eva"}));
    CHECK(resolver.resolve("Carex goodenowii").route == NameRoute::Alias);
    // Neither the alias nor a spelling of it reaches a token this vocabulary holds.
    CHECK(resolver.resolve("Carex viridula").route == NameRoute::Unresolved);
    CHECK(resolver.resolve("Carex viridulla").route == NameRoute::Unresolved);
}

TEST_CASE("the spelling step needs every nearest entry to give one token", "[names]") {
    const auto vocab = vocabulary({"Rosa canina", "Rosa canino", "Salix alba"});
    const NameResolver resolver(vocab, table({"Salix albus"}, {"Salix alba"}, {"wfo"}));
    // One edit from two tokens: left as written.
    CHECK(resolver.resolve("Rosa caninu").route == NameRoute::Unresolved);
    // One edit from a token and from a table name of the same token: resolved,
    // via the first entry (tokens come first).
    const ResolvedName salix = resolver.resolve("Salix albu");
    CHECK(salix.route == NameRoute::Spelling);
    CHECK(salix.token == "Salix alba");
    CHECK(salix.via == "Salix alba");
}

TEST_CASE("ties past the first nearest entries are all seen", "[names]") {
    // Ten table names one edit from the needle, more than the first query
    // keeps, all of one token: resolved.
    std::vector<std::string> same, same_tokens(10, "Carex nigra"), same_sources(10, "wfo");
    for (int i = 0; i < 10; ++i) same.push_back("Carex x" + std::to_string(i));
    const auto vocab = vocabulary({"Carex nigra", "Carex flava"});
    const ResolvedName one = NameResolver(vocab, table(same, same_tokens, same_sources)).resolve("Carex x");
    CHECK(one.route == NameRoute::Spelling);
    CHECK(one.token == "Carex nigra");

    // The last of eleven ties names another token: no longer one token.
    same.push_back("Carex xz");
    same_tokens.push_back("Carex flava");
    same_sources.push_back("wfo");
    CHECK(NameResolver(vocab, table(same, same_tokens, same_sources)).resolve("Carex x").route ==
          NameRoute::Unresolved);
}

TEST_CASE("the genus is searched first, every entry only when it holds none", "[names]") {
    const auto vocab = vocabulary({"Poa annua", "Poa alpina", "Pox annuaxx"});
    const NameResolver resolver(vocab, nullptr, edits(2));
    // Two edits within the genus Poa are taken over one edit in another genus.
    const ResolvedName poa = resolver.resolve("Poa annuaxx");
    CHECK(poa.route == NameRoute::Spelling);
    CHECK(poa.token == "Poa annua");
    // A misspelled genus has no bucket: the whole index is searched.
    const ResolvedName pob = resolver.resolve("Pob alpina");
    CHECK(pob.route == NameRoute::Spelling);
    CHECK(pob.token == "Poa alpina");
}

TEST_CASE("max_edit_distance bounds the spelling step and 0 turns it off", "[names]") {
    const auto vocab = vocabulary({"Quercus robur"});
    CHECK(NameResolver(vocab, nullptr, edits(0)).resolve("Quercus robru").route ==
          NameRoute::Unresolved);
    CHECK(NameResolver(vocab, nullptr, edits(1)).resolve("Qurecus robru").route ==
          NameRoute::Unresolved);
    const ResolvedName two = NameResolver(vocab, nullptr, edits(2)).resolve("Qurecus robru");
    CHECK(two.route == NameRoute::Spelling);
    CHECK(two.distance == 2);
    CHECK_THROWS(NameResolver(vocab, nullptr, edits(-1)));
    CHECK_THROWS(NameResolver(vocab, nullptr, edits(kMaxNameEditDistance + 1)));
}

TEST_CASE("route names round-trip", "[names]") {
    for (const NameRoute r : {NameRoute::Exact, NameRoute::Alias, NameRoute::Spelling,
                              NameRoute::Unresolved}) {
        CHECK(enum_from_name<NameRoute>(enum_to_name(r)) == r);
    }
}

TEST_CASE("the name table file is read with its digest and checked", "[names]") {
    const std::string text =
        R"({"source": "test", "aliases": {"Anemone nemorosa": ["Anemone altaica", "eva"],)"
        R"( "Carex viridula": ["Carex curvicollis", "eva"]}})";
    const TempFile file(text, ".json", resolve::test::WriteMode::Binary);
    const NameTable t = NameTable::read(file.path());
    CHECK(t.size() == 2);
    CHECK(t.name(0) == "Anemone nemorosa");
    CHECK(t.find("Carex viridula")->token == "Carex curvicollis");
    CHECK(t.find("Carex viridula")->source == "eva");
    CHECK(t.find("Carex") == nullptr);
    CHECK(t.description() == "test");
    CHECK(t.sha256() == sha256_hex(text));

    const TempFile bad(R"({"aliases": {"Anemone nemorosa": ["Anemone altaica"]}})", ".json",
                       resolve::test::WriteMode::Binary);
    CHECK_THROWS(NameTable::read(bad.path()));
    const TempFile none(R"({"source": "x"})", ".json", resolve::test::WriteMode::Binary);
    CHECK_THROWS(NameTable::read(none.path()));
    CHECK_THROWS(NameTable::from_columns({"a", "a"}, {"b", "b"}, {"", ""}));
    CHECK_THROWS(NameTable::from_columns({"a"}, {""}, {""}));
    CHECK_THROWS(NameTable::from_columns({"a"}, {"b", "c"}, {""}));
}

TEST_CASE("a load with names reads each record as its token and reports the names", "[names][dataset]") {
    const DatasetConfig cfg = resolve::test::vocab_dataset_config(SpeciesEncodingMode::RankPool, true);
    const RoleMapping roles = Roles();
    const TempFile train("plot_id,sp,cover\n"
                         "p1,Anemone altaica,10\np1,Quercus robur,5\n"
                         "p2,Carex curvicollis,20\np2,Quercus robur,1\n");
    const ResolveDataset fitted = ResolveDataset::load(std::nullopt, TableSource::csv(train.path()),
                                                       roles, {}, cfg);
    const ExternalVocabs vocabs = fitted.external_vocabs();

    // The same plots written with the tokens, and with an entered name, a
    // misspelling and a species the vocabulary lacks. In q1 the entered name
    // and its token are both recorded and merge into one entry.
    const TempFile as_tokens("plot_id,sp,cover\n"
                             "q1,Anemone altaica,12\nq1,Quercus robur,5\n"
                             "q2,Carex curvicollis,20\nq2,Tragopogon dubius,3\n");
    const TempFile as_written("plot_id,sp,cover\n"
                              "q1,Anemone nemorosa,10\nq1,Anemone altaica,2\nq1,Quercus robru,5\n"
                              "q2,Carex curvicollis,20\nq2,Tragopogon dubius,3\n");
    SpeciesNames names;
    names.table = table({"Anemone nemorosa"}, {"Anemone altaica"}, {"eva"});

    const ResolveDataset expected = ResolveDataset::load(
        std::nullopt, TableSource::csv(as_tokens.path()), roles, {}, cfg, &vocabs);
    const ResolveDataset resolved = ResolveDataset::load(
        std::nullopt, TableSource::csv(as_written.path()), roles, {}, cfg, &vocabs, &names);

    CHECK(torch::equal(resolved.species_ids(), expected.species_ids()));
    CHECK(torch::equal(resolved.pool_weights(), expected.pool_weights()));
    CHECK(torch::equal(resolved.unknown_count(), expected.unknown_count()));
    CHECK(torch::equal(resolved.species_recognition().n_recognised,
                       expected.species_recognition().n_recognised));
    CHECK(expected.species_names().size() == 0);

    const SpeciesNameReport& report = resolved.species_names();
    REQUIRE(report.size() == 3);
    CHECK(report.name == std::vector<std::string>{"Anemone nemorosa", "Quercus robru", "Tragopogon dubius"});
    CHECK(report.token == std::vector<std::string>{"Anemone altaica", "Quercus robur", "Tragopogon dubius"});
    std::vector<std::string> routes;
    for (const NameRoute r : report.route) routes.emplace_back(enum_to_name(r));
    CHECK(routes == std::vector<std::string>{"alias", "spelling", "unresolved"});
    CHECK(report.source == std::vector<std::string>{"eva", "", ""});
    CHECK(report.distance == std::vector<int32_t>{0, 1, 0});
    CHECK(report.n_records == std::vector<int64_t>{1, 1, 1});

    // Without names the table is read as written.
    const ResolveDataset plain = ResolveDataset::load(
        std::nullopt, TableSource::csv(as_written.path()), roles, {}, cfg, &vocabs);
    CHECK(plain.species_names().size() == 0);
    CHECK_FALSE(torch::equal(plain.species_ids(), expected.species_ids()));

    // Names resolve against a trained vocabulary only.
    CHECK_THROWS(ResolveDataset::load(std::nullopt, TableSource::csv(as_written.path()), roles, {},
                                      cfg, nullptr, &names));
}

#include "fishfilter.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "external/chess.hpp"
#include "external/gzip/gzstream.h"
#include "external/parallel_hashmap/phmap.h"
#include "external/threadpool.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

using namespace chess;

using fen_map_t = phmap::parallel_flat_hash_map<
    std::string, int, std::hash<std::string>, std::equal_to<std::string>,
    std::allocator<std::pair<const std::string, int>>, 8, std::mutex>;

// map to hold move counters that cutechess-cli changed from original FENs
using map_fens = std::unordered_map<std::string, std::pair<int, int>>;

fen_map_t fen_map;

// map to collect metadata for tests
using map_meta = std::unordered_map<std::string, TestMetaData>;

std::atomic<std::size_t> total_files = 0;
std::atomic<std::size_t> total_games = 0;

namespace analysis {

class Analyze : public pgn::Visitor {
public:
  Analyze(std::string_view file, const std::string &regex_engine,
          const map_fens &fixfen_map, int abs_eval_min, int abs_eval_max,
          int min_depth, const unsigned int count_stop_early,
          std::mutex &progress_output)
      : file(file), regex_engine(regex_engine), fixfen_map(fixfen_map),
        abs_eval_min(abs_eval_min), abs_eval_max(abs_eval_max),
        min_depth(min_depth), count_stop_early(count_stop_early),
        progress_output(progress_output) {}

  virtual ~Analyze() {}

  void startPgn() override {}

  void header(std::string_view key, std::string_view value) override {

    if (key == "FEN") {
      std::regex p("^(.+) 0 1$");
      std::smatch match;
      std::string value_str(value);

      // revert changes by cutechess-cli to move counters
      if (!fixfen_map.empty() && std::regex_search(value_str, match, p) &&
          match.size() > 1) {
        std::string fen = match[1];
        auto it = fixfen_map.find(fen);

        if (it == fixfen_map.end()) {
          std::cerr << "While parsing " << file << " could not find FEN " << fen
                    << " in fixFENsource." << std::endl;
          board.setFen(value);
        } else {
          const auto &fix = it->second;
          std::string fixed_value = fen + " " + std::to_string(fix.first) +
                                    " " + std::to_string(fix.second);
          board.setFen(fixed_value);
        }
      } else
        board.setFen(value);

      fen = value;
    }

    if (key == "Variant" && value == "fischerandom") {
      board.set960(true);
    }

    if (key == "White") {
      white = value;
    }

    if (key == "Black") {
      black = value;
    }

    if (key == "WhiteElo") {
      whiteElo = std::atoi(value.data());
    }
    if (key == "BlackElo") {
      blackElo = std::atoi(value.data());
    }
  }

  void startMoves() override {
    do_filter = !regex_engine.empty();

    if (do_filter) {
      if (white.empty() || black.empty()) {
        this->skipPgn(true);
        return;
      }

      std::regex regex(regex_engine);

      if (std::regex_match(white, regex)) {
        filter_side = Color::WHITE;
      }

      if (std::regex_match(black, regex)) {
        if (filter_side == Color::NONE) {
          filter_side = Color::BLACK;
        } else {
          do_filter = false;
        }
      }
    }
    total_games++;
  }

  void move(std::string_view move, std::string_view comment) override {

    // fishtest uses Nf3 {+0.57/17 2.313s}
    const size_t slash_pos = comment.find("/");
    const size_t space_pos = comment.find(" ");

    if (slash_pos != std::string::npos && comment != "book") {
      const auto match_eval = comment.substr(0, slash_pos);
      int eval;
      if (match_eval[1] == 'M')
        eval = match_eval[0] == '+' ? 30001 : -30001;
      else
        eval =
            std::clamp(int(100 * fast_stof(match_eval.data())), -30000, 30000);
      const auto match_depth = comment.substr(slash_pos + 1, space_pos);
      int depth = std::max(1, std::stoi(match_depth.data()));
      const auto abseval = std::abs(eval);
      if (depth > min_depth && abseval >= abs_eval_min &&
          abseval <= abs_eval_max) {
        const auto knights = board.pieces(PieceType::KNIGHT).count();
        const auto bishops = board.pieces(PieceType::BISHOP).count();
        const auto rooks = board.pieces(PieceType::ROOK).count();
        const auto queens = board.pieces(PieceType::QUEEN).count();
        const auto pawns = board.pieces(PieceType::PAWN).count();

        const auto material =
            9 * queens + 5 * rooks + 3 * bishops + 3 * knights + pawns;
        // endgame filter
        if (material - pawns < 20) {
          auto key = board.getFen();
          bool is_new_entry = fen_map.lazy_emplace_l(
              std::move(key),
              [&](fen_map_t::value_type &p) { p.second = eval; },
              [&](const fen_map_t::constructor &ctor) {
                ctor(std::move(key), eval);
              });

          if (is_new_entry)
            new_entry_count++;

          if (count_stop_early == new_entry_count) {
            this->skipPgn(true);
            return;
          }
        }
      }
    }

    try {
      Move m = uci::parseSan(board, move, moves);

      // chess-lib may call move() with empty strings for move
      if (m == Move::NO_MOVE) {
        this->skipPgn(true);
        return;
      }

      board.makeMove<true>(m);
    } catch (const uci::AmbiguousMoveError &e) {
      std::cerr << "While parsing " << file << " encountered: " << e.what()
                << '\n';
      this->skipPgn(true);
    }
  }

  void endPgn() override {
    fen = constants::STARTPOS;
    board.set960(false);
    board.setFen(constants::STARTPOS);

    filter_side = Color::NONE;

    white.clear();
    black.clear();

    whiteElo = blackElo = 0;
    new_entry_count = 0;
  }

private:
  std::string_view file;
  const std::string &regex_engine;
  const map_fens &fixfen_map;
  int abs_eval_min, abs_eval_max;
  int min_depth;
  const unsigned int count_stop_early;
  std::mutex &progress_output;

  Board board;
  Movelist moves;

  std::string fen = constants::STARTPOS;
  bool skip = false;

  bool do_filter = false;
  Color filter_side = Color::NONE;

  std::string white;
  std::string black;

  int whiteElo = 0, blackElo = 0;
  unsigned int new_entry_count = 0;
};

void ana_files(const std::vector<std::string> &files,
               const std::string &regex_engine, const map_fens &fixfen_map,
               int abs_eval_min, int abs_eval_max, int min_depth,
               const unsigned int count_stop_early,
               std::mutex &progress_output) {

  for (const auto &file : files) {
    std::string move_counter;
    const auto pgn_iterator = [&](std::istream &iss) {
      auto vis = std::make_unique<Analyze>(
          file, regex_engine, fixfen_map, abs_eval_min, abs_eval_max, min_depth,
          count_stop_early, progress_output);

      pgn::StreamParser parser(iss);

      try {
        parser.readGames(*vis);
      } catch (const std::exception &e) {
        std::cout << "Error when parsing: " << file << std::endl;
        std::cerr << e.what() << '\n';
      }
    };

    if (file.size() >= 3 && file.substr(file.size() - 3) == ".gz") {
      igzstream input(file.c_str());
      pgn_iterator(input);
    } else {
      std::ifstream pgn_stream(file);
      pgn_iterator(pgn_stream);
      pgn_stream.close();
    }

    ++total_files;

    // Limit the scope of the lock
    {
      const std::lock_guard<std::mutex> lock(progress_output);
      std::cout << "\rProcessed " << total_files << " files" << std::flush;
    }
  }
}

} // namespace analysis

[[nodiscard]] map_fens get_fixfen(std::string file) {
  map_fens fixfen_map;
  if (file.empty()) {
    return fixfen_map;
  }

  const auto fen_iterator = [&](std::istream &iss) {
    std::string line;
    while (std::getline(iss, line)) {
      std::istringstream iss(line);
      std::string f1, f2, f3, ep;
      int halfmove, fullmove = 0;

      iss >> f1 >> f2 >> f3 >> ep >> halfmove >> fullmove;

      if (!fullmove)
        continue;

      auto key = f1 + ' ' + f2 + ' ' + f3 + ' ' + ep;
      auto fixfen_data = std::pair<int, int>(halfmove, fullmove);

      if (fixfen_map.find(key) != fixfen_map.end()) {
        // for duplicate FENs, prefer the one with lower full move counter
        if (fullmove < fixfen_map[key].second) {
          fixfen_map[key] = fixfen_data;
        }
      } else {
        fixfen_map[key] = fixfen_data;
      }
    }
  };

  if (file.size() >= 3 && file.substr(file.size() - 3) == ".gz") {
    igzstream input(file.c_str());
    fen_iterator(input);
  } else {
    std::ifstream input(file);
    fen_iterator(input);
  }

  return fixfen_map;
}

[[nodiscard]] map_meta get_metadata(const std::vector<std::string> &file_list,
                                    bool allow_duplicates) {
  map_meta meta_map;
  // map to check for duplicate tests
  std::unordered_map<std::string, std::string> test_map;
  std::set<std::string> test_warned;

  for (const auto &pathname : file_list) {
    fs::path path(pathname);
    std::string filename = path.filename().string();
    std::string test_id = filename.substr(0, filename.find_first_of("-."));
    std::string test_filename = (path.parent_path() / test_id).string();

    if (test_map.find(test_id) == test_map.end()) {
      test_map[test_id] = test_filename;
    } else if (test_map[test_id] != test_filename) {
      if (test_warned.find(test_filename) == test_warned.end()) {
        std::cout << (allow_duplicates ? "Warning" : "Error")
                  << ": Detected a duplicate of test " << test_id
                  << " in directory " << path.parent_path().string()
                  << std::endl;
        test_warned.insert(test_filename);

        if (!allow_duplicates) {
          std::cout << "Use --allowDuplicates to continue nonetheless."
                    << std::endl;
          std::exit(1);
        }
      }
    }

    // load the JSON data from disk, only once for each test
    if (meta_map.find(test_filename) == meta_map.end()) {
      std::ifstream json_file(test_filename + ".json");

      if (!json_file.is_open())
        continue;

      json metadata = json::parse(json_file);

      meta_map[test_filename] = metadata.get<TestMetaData>();
    }
  }

  return meta_map;
}

template <typename STRATEGY>
void filter_files(std::vector<std::string> &file_list, const map_meta &meta_map,
                  const STRATEGY &strategy) {
  const auto applier = [&](const std::string &pathname) {
    fs::path path(pathname);
    std::string filename = path.filename().string();
    std::string test_id = filename.substr(0, filename.find_first_of("-."));
    std::string test_filename = (path.parent_path() / test_id).string();
    return strategy.apply(test_filename, meta_map);
  };
  const auto it = std::remove_if(file_list.begin(), file_list.end(), applier);
  file_list.erase(it, file_list.end());
}

class BookFilterStrategy {
  std::regex regex_book;
  bool invert;

public:
  BookFilterStrategy(const std::regex &rb, bool inv)
      : regex_book(rb), invert(inv) {}

  bool apply(const std::string &filename, const map_meta &meta_map) const {
    // check if metadata and "book" entry exist
    if (meta_map.find(filename) != meta_map.end() &&
        meta_map.at(filename).book.has_value()) {
      bool match =
          std::regex_match(meta_map.at(filename).book.value(), regex_book);
      return invert ? match : !match;
    }

    // missing metadata or "book" entry can never match
    return true;
  }
};

class RevFilterStrategy {
  std::regex regex_rev;

public:
  RevFilterStrategy(const std::regex &rb) : regex_rev(rb) {}

  bool apply(const std::string &filename, const map_meta &meta_map) const {
    if (meta_map.find(filename) == meta_map.end()) {
      return true;
    }

    if (meta_map.at(filename).resolved_base.has_value() &&
        std::regex_match(meta_map.at(filename).resolved_base.value(),
                         regex_rev)) {
      return false;
    }

    if (meta_map.at(filename).resolved_new.has_value() &&
        std::regex_match(meta_map.at(filename).resolved_new.value(),
                         regex_rev)) {
      return false;
    }

    return true;
  }
};

class TcFilterStrategy {
  std::regex regex_tc;

public:
  TcFilterStrategy(const std::regex &rb) : regex_tc(rb) {}

  bool apply(const std::string &filename, const map_meta &meta_map) const {
    if (meta_map.find(filename) == meta_map.end()) {
      return true;
    }

    if (meta_map.at(filename).new_tc.has_value() &&
        meta_map.at(filename).tc.has_value()) {
      if (meta_map.at(filename).new_tc.value() !=
          meta_map.at(filename).tc.value()) {
        return true;
      }

      if (std::regex_match(meta_map.at(filename).tc.value(), regex_tc)) {
        return false;
      }
    }

    return true;
  }
};

class ThreadsFilterStrategy {
  int threads;

public:
  ThreadsFilterStrategy(int t) : threads(t) {}

  bool apply(const std::string &filename, const map_meta &meta_map) const {
    if (meta_map.find(filename) == meta_map.end()) {
      return true;
    }

    if (meta_map.at(filename).threads.has_value() &&
        meta_map.at(filename).threads.value() == threads) {
      return false;
    }

    return true;
  }
};

void process(const std::vector<std::string> &files_pgn,
             const std::string &regex_engine, const map_fens &fixfen_map,
             int abs_eval_min, int abs_eval_max, int min_depth,
             const unsigned int count_stop_early, int concurrency) {
  // Create more chunks than threads to prevent threads from idling.
  int target_chunks = 4 * concurrency;

  auto files_chunked = split_chunks(files_pgn, target_chunks);

  std::cout << "Found " << files_pgn.size() << " .pgn(.gz) files, creating "
            << files_chunked.size() << " chunks for processing." << std::endl;

  // Mutex for progress output
  std::mutex progress_output;

  // Create a thread pool
  ThreadPool pool(concurrency);

  for (const auto &files : files_chunked) {

    pool.enqueue([&files, &regex_engine, &fixfen_map, &abs_eval_min,
                  &abs_eval_max, &min_depth, &count_stop_early,
                  &progress_output, &files_chunked]() {
      analysis::ana_files(files, regex_engine, fixfen_map, abs_eval_min,
                          abs_eval_max, min_depth, count_stop_early,
                          progress_output);
    });
  }

  // Wait for all threads to finish
  pool.wait();
}

void print_usage(char const *program_name) {
  std::stringstream ss;

  // clang-format off
    ss << "Usage: " << program_name << " [options]" << "\n";
    ss << "Options:" << "\n";
    ss << "  --file <path>         Path to .pgn(.gz) file" << "\n";
    ss << "  --dir <path>          Path to directory containing .pgn(.gz) files (default: pgns)" << "\n";
    ss << "  -r                    Search for .pgn(.gz) files recursively in subdirectories" << "\n";
    ss << "  --allowDuplicates     Allow duplicate directories for test pgns" << "\n";
    ss << "  --concurrency <N>     Number of concurrent threads to use (default: maximum)" << "\n";
    ss << "  --matchRev <regex>    Filter data based on revision SHA in metadata" << "\n";
    ss << "  --matchEngine <regex> Filter data based on engine name in pgns, defaults to matchRev if given" << "\n";
    ss << "  --matchTC <regex>     Filter data based on time control in metadata" << "\n";
    ss << "  --matchThreads <N>    Filter data based on used threads in metadata" << "\n";
    ss << "  --matchBook <regex>   Filter data based on book name" << "\n";
    ss << "  --matchBookInvert     Invert the filter" << "\n";
    ss << "  -o <path>             Path to output epd file (default: fishfilter.epd)" << "\n";
    ss << "  --fixFENsource        Patch move counters lost by cutechess-cli based on FENs in this file" << "\n";
    ss << "  --absEvalMin <cp>     Minimum absolute evaluation in cp (default: 85)" << "\n";
    ss << "  --absEvalMax <cp>     Maximum absolute evaluation in cp (default: 115)" << "\n";
    ss << "  --minDepth            Minimal depth needed to accept an eval (default: 0)" << "\n";
    ss << "  --stopEarly           Stop analysing the game as soon as countStopEarly positions are reached (default false) for the analysing thread." << "\n";
    ss << "  --countStopEarly <N>  Number of new positions encountered before stopping with stopEarly (default 1)" << "\n";
    ss << "  --help                Print this help message" << "\n";
  // clang-format on

  std::cout << ss.str();
}

int main(int argc, char const *argv[]) {
  CommandLine cmd(argc, argv);

  std::vector<std::string> files_pgn;
  std::string default_path = "./pgns";
  std::string regex_engine, regex_book, filename = "fishfilter.epd";
  map_fens fixfen_map;
  int abs_eval_min = 85, abs_eval_max = 115, min_depth = 0;
  int concurrency = std::max(1, int(std::thread::hardware_concurrency()));
  unsigned int count_stop_early = 1;

  if (cmd.has_argument("--help", true)) {
    print_usage(argv[0]);
    return 0;
  }

  if (cmd.has_argument("--concurrency")) {
    concurrency = std::stoi(cmd.get_argument("--concurrency"));
  }

  if (cmd.has_argument("--file")) {
    files_pgn = {cmd.get_argument("--file")};
    if (!fs::exists(files_pgn[0])) {
      std::cout << "Error: File not found: " << files_pgn[0] << std::endl;
      std::exit(1);
    }
  } else {
    auto path = cmd.get_argument("--dir", default_path);

    bool recursive = cmd.has_argument("-r", true);
    std::cout << "Looking " << (recursive ? "(recursively) " : "")
              << "for pgn files in " << path << std::endl;

    files_pgn = get_files(path, recursive);

    // sort to easily check for "duplicate" files, i.e. "foo.pgn.gz" and
    // "foo.pgn"
    std::sort(files_pgn.begin(), files_pgn.end());

    for (size_t i = 1; i < files_pgn.size(); ++i) {
      if (files_pgn[i].find(files_pgn[i - 1]) == 0) {
        std::cout << "Error: \"Duplicate\" files: " << files_pgn[i - 1]
                  << " and " << files_pgn[i] << std::endl;
        std::exit(1);
      }
    }
  }

  std::cout << "Found " << files_pgn.size() << " .pgn(.gz) files in total."
            << std::endl;

  auto meta_map =
      get_metadata(files_pgn, cmd.has_argument("--allowDuplicates", true));

  if (cmd.has_argument("--matchBook")) {
    auto regex_book = cmd.get_argument("--matchBook");

    if (!regex_book.empty()) {
      bool invert = cmd.has_argument("--matchBookInvert", true);
      std::cout << "Filtering pgn files " << (invert ? "not " : "")
                << "matching the book name " << regex_book << std::endl;
      filter_files(files_pgn, meta_map,
                   BookFilterStrategy(std::regex(regex_book), invert));
    }
  }

  if (cmd.has_argument("--matchRev")) {
    auto regex_rev = cmd.get_argument("--matchRev");

    if (!regex_rev.empty()) {
      std::cout << "Filtering pgn files matching revision SHA " << regex_rev
                << std::endl;
      filter_files(files_pgn, meta_map,
                   RevFilterStrategy(std::regex(regex_rev)));
    }

    regex_engine = regex_rev;
  }

  if (cmd.has_argument("--fixFENsource"))
    fixfen_map = get_fixfen(cmd.get_argument("--fixFENsource"));

  if (cmd.has_argument("--matchEngine")) {
    regex_engine = cmd.get_argument("--matchEngine");
  }

  if (cmd.has_argument("--matchTC")) {
    auto regex_tc = cmd.get_argument("--matchTC");

    if (!regex_tc.empty()) {
      std::cout << "Filtering pgn files matching TC " << regex_tc << std::endl;
      filter_files(files_pgn, meta_map, TcFilterStrategy(std::regex(regex_tc)));
    }
  }

  if (cmd.has_argument("--matchThreads")) {
    int threads = std::stoi(cmd.get_argument("--matchThreads"));

    std::cout << "Filtering pgn files using threads = " << threads << std::endl;
    filter_files(files_pgn, meta_map, ThreadsFilterStrategy(threads));
  }

  if (cmd.has_argument("--absEvalMin")) {
    abs_eval_min = std::stoi(cmd.get_argument("--absEvalMin"));
  }
  if (cmd.has_argument("--absEvalMax")) {
    abs_eval_max = std::stoi(cmd.get_argument("--absEvalMax"));
  }
  if (cmd.has_argument("--minDepth")) {
    min_depth = std::stoi(cmd.get_argument("--minDepth"));
  }
  std::cout << "Filtering positions with absEval in [" << abs_eval_min << ","
            << abs_eval_max << "] and depth >= " << min_depth << "."
            << std::endl;

  bool stop_early = cmd.has_argument("--stopEarly", true);
  if (cmd.has_argument("--countStopEarly")) {
    count_stop_early = std::stoi(cmd.get_argument("--countStopEarly"));
  }
  if (stop_early)
    std::cout << "Filtering at most " << count_stop_early
              << " position(s) per game." << std::endl;
  else
    count_stop_early = std::numeric_limits<decltype(count_stop_early)>::max();

  if (cmd.has_argument("-o")) {
    filename = cmd.get_argument("-o");
  }

  std::ofstream out_file(filename);

  const auto t0 = std::chrono::high_resolution_clock::now();

  process(files_pgn, regex_engine, fixfen_map, abs_eval_min, abs_eval_max,
          min_depth, count_stop_early, concurrency);

  for (const auto &pair : fen_map) {
    std::string fen = pair.first;
    auto board = Board(fen);
    out_file << fen << " c0 \"eval: " << pair.second << "\";\n";
  }
  out_file.close();

  const auto t1 = std::chrono::high_resolution_clock::now();

  std::cout << "\nSaved " << fen_map.size() << " unique positions from "
            << total_games << " games to " << filename << "."
            << "\nTotal time for processing: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                       .count() /
                   1000.0
            << " s" << std::endl;

  return 0;
}

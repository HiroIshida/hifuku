#include <optional>
#include <limits>

extern "C" { 
  void compute_coverage_and_fp_jit(
    double* biases, // numpy's [N_path]
    double* realss, // numpy's [N_mc, N_path]
    double* estss, // numpy's [N_mc, N_path]
    double threshold,
    int N_path, int N_mc,
    double* optimal_coverage,
    double* optimal_fp);
}

void compute_coverage_and_fp_jit(
    double* biases,
    double* realss,
    double* estss,
    double threshold,
    int N_path, int N_mc,
    double* optimal_coverage,
    double* optimal_fp) {
  size_t n_coverage = 0;
  size_t n_fp = 0;

  for(size_t j = 0; j < N_mc; j++) {
    std::optional<size_t> best_path_idx = std::nullopt;
    double value_min = std::numeric_limits<double>::max();
    for(size_t i = 0; i < N_path; i++) {
      auto value = estss[j * N_path + i] + biases[i];
      if(value < value_min) {
        value_min = std::move(value);
        best_path_idx = i;
      }
    }
    bool is_est_ok = estss[j * N_path +  *best_path_idx] + biases[*best_path_idx] < threshold;
    if(is_est_ok) {
      n_coverage++;
      bool is_real_ok = realss[j * N_path +  *best_path_idx] < threshold;
      if (!is_real_ok) {
        n_fp++;
      }
    }
  }
  *optimal_coverage = static_cast<double>(n_coverage) / N_mc;
  if(n_coverage > 0) { 
    double fp = static_cast<double>(n_fp) / n_coverage;
    *optimal_fp = fp;
  } else {
    *optimal_fp = std::numeric_limits<double>::infinity();
  }
}

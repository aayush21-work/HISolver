#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <string>
#include <cmath>

static double read_r(const std::string& fname)
{
    std::ifstream f(fname);
    if (!f.is_open()) throw std::runtime_error("Cannot open " + fname);
    std::string line;
    while (std::getline(f, line))
    {
        std::istringstream ss(line);
        std::string tok;
        ss >> tok;
        if (tok == "r")
        {
            auto eq = line.find('=');
            if (eq != std::string::npos)
                return std::stod(line.substr(eq + 1));
        }
    }
    throw std::runtime_error("r not found in " + fname);
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <input.dat> [output.dat]\n";
        return 1;
    }
    const std::string infile = argv[1];

    std::ifstream in(infile);
    if (!in.is_open()) throw std::runtime_error("Cannot open file!");

    struct Row { double k; double p; };
    std::string line;
    std::vector<Row> data;
    Row row;

    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        if (!(iss >> row.k >> row.p)) continue;
        if (!std::isfinite(row.k) || !std::isfinite(row.p)) continue;
        data.push_back(row);
    }
    if (data.empty()) throw std::runtime_error("No valid data rows!");

    
    Row pivot = data[0];
    double best = std::abs(data[0].k - 1.0);
    for (const auto& i : data)
    {
        double d = std::abs(i.k - 1.0);
        if (d < best) { best = d; pivot = i; }
    }
    if (!(pivot.p > 0.0))
        throw std::runtime_error("Pivot P is non-positive; cannot normalize.");

    const double As      = 2.18e-9;   // scalar amplitude at pivot
    const double k_pivot = 0.05;      // pivot in Mpc^-1

    
    double target = As;
    if (infile.find("tensor") != std::string::npos)
    {
        double r = read_r("spectra_summary.dat");
        target = r * As;
        std::cout << "tensor file: r=" << r << ", target A_t=" << target << "\n";
    }

    const double factor_p = target / pivot.p;

    std::string outfile = (argc >= 3)
        ? argv[2]
        : infile.substr(0, infile.rfind('.')) + "_rescaled"
          + infile.substr(infile.rfind('.'));

    std::ofstream out(outfile);
    out.precision(12);
    out << "# k [Mpc^-1]  P   (pivot " << k_pivot << " Mpc^-1, target "
        << target << ")\n";
    for (auto i : data)
        out << i.k * k_pivot << " " << i.p * factor_p << '\n';

    std::cout << "pivot k/k_star=" << pivot.k << "  P=" << pivot.p
              << "  factor_p=" << factor_p
              << "  -> " << outfile << " (" << data.size() << " rows)\n";
    return 0;
}

// perturbations.cpp
// =================
// Mukhanov-Sasaki solver for scalar and tensor modes.
// Reads background.dat, solves MS equation for each k mode,
// outputs primordial power spectra P_R(k) and P_T(k).
//

#include "model_potential.hpp"
#include "model_generated.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

// =============================================================================
// Background trajectory
// =============================================================================

struct BGPoint {
    double N, phi, dphi, H, logH, eps;
    double QT_val, cT2_val, Qs_val, cs2_val;
};

static std::vector<BGPoint> load_background(const std::string &fname) {
    std::ifstream in(fname);
    if (!in) throw std::runtime_error("Cannot open " + fname);
    std::vector<BGPoint> tr;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0]=='#') continue;
        BGPoint p;
        int n = sscanf(line.c_str(),
            "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
            &p.N,&p.phi,&p.dphi,&p.H,&p.logH,&p.eps,
            &p.QT_val,&p.cT2_val,&p.Qs_val,&p.cs2_val);
        if (n >= 9) {
            if (n < 10) p.cs2_val = 1.0;   // fallback if cs2 absent
            tr.push_back(p);
        }
    }
    if (tr.empty()) throw std::runtime_error("Empty background file");
    std::cout << "Loaded " << tr.size() << " background points, "
              << "N = " << tr.front().N << " to " << tr.back().N << "\n";
    return tr;
}

// =============================================================================
// Linear interpolation
// =============================================================================

static double interp(const std::vector<BGPoint> &tr, double N,
                     double BGPoint::*f) {
    int lo=0, hi=(int)tr.size()-1;
    while (hi-lo>1) {
        int m=(lo+hi)/2;
        (tr[m].N<N ? lo : hi) = m;
    }
    double t=(N-tr[lo].N)/(tr[hi].N-tr[lo].N);
    return tr[lo].*f*(1-t) + tr[hi].*f*t;
}

#define BG(FIELD) interp(tr,N,&BGPoint::FIELD)

static double dX_dN(const std::vector<BGPoint> &tr, double N,
                    double BGPoint::*f) {
    double dN=1e-3;
    double N0=tr.front().N, N1=tr.back().N;
    if (N-dN<N0) dN=(N-N0)*0.5;
    if (N+dN>N1) dN=(N1-N)*0.5;
    if (dN<1e-8) return 0.0;
    return (interp(tr,N+dN,f)-interp(tr,N-dN,f))/(2*dN);
}


struct Mode
{
    double Re, Im, ReP, ImP ;

 };

static Mode scalar_rhs(const Mode &y, double N, double k,
                        const std::vector<BGPoint> &tr) {
    double H_   = BG(H);
    double eps_ = BG(eps);
    double Qs_  = BG(Qs_val);
    double cs2_ = BG(cs2_val);
    double QsN_ = dX_dN(tr, N, &BGPoint::Qs_val);
    double x    = (k / H_) * std::exp(-N);
    double fric = 3.0 + eps_ + QsN_/Qs_;
    double om2  = cs2_ * x * x;

    return { y.ReP, y.ImP,
             -fric*y.ReP - om2*y.Re,
             -fric*y.ImP - om2*y.Im };
}

static Mode tensor_rhs(const Mode &y, double N, double k,
                        const std::vector<BGPoint> &tr) {
    double H_   = BG(H);
    double eps_ = BG(eps);
    double QT_  = BG(QT_val);
    double cT2_ = BG(cT2_val);
    double QTN_ = dX_dN(tr, N, &BGPoint::QT_val);

    double x    = (k / H_) * std::exp(-N);
    double fric = 3.0 - eps_ + QTN_/QT_;
    double om2  = cT2_ * x * x;

    return { y.ReP, y.ImP,
             -fric*y.ReP - om2*y.Re,
             -fric*y.ImP - om2*y.Im };
}

static Mode rk4_MS(const Mode &y, double N, double dN, double k,
                   const std::vector<BGPoint> &tr, bool scalar) {
    auto f = scalar ? scalar_rhs : tensor_rhs;
    auto a = f(y, N, k, tr);
    auto b = f({y.Re+0.5*dN*a.Re, y.Im+0.5*dN*a.Im,
                y.ReP+0.5*dN*a.ReP, y.ImP+0.5*dN*a.ImP}, N+0.5*dN, k, tr);
    auto c = f({y.Re+0.5*dN*b.Re, y.Im+0.5*dN*b.Im,
                y.ReP+0.5*dN*b.ReP, y.ImP+0.5*dN*b.ImP}, N+0.5*dN, k, tr);
    auto d = f({y.Re+dN*c.Re, y.Im+dN*c.Im,
                y.ReP+dN*c.ReP, y.ImP+dN*c.ImP}, N+dN, k, tr);
    return { y.Re +(dN/6)*(a.Re +2*b.Re +2*c.Re +d.Re ),
             y.Im +(dN/6)*(a.Im +2*b.Im +2*c.Im +d.Im ),
             y.ReP+(dN/6)*(a.ReP+2*b.ReP+2*c.ReP+d.ReP),
             y.ImP+(dN/6)*(a.ImP+2*b.ImP+2*c.ImP+d.ImP) };
}



static double solve_mode(double k, bool scalar,
                          const std::vector<BGPoint> &tr) {

    const double x_init   = 50.0;   // k/(aH) at IC  (subhorizon)
    const double x_freeze = 0.05;   // k/(aH) at superhorizon freeze-out
    const double dN_past  = 4.0;    // e-folds past horizon exit to evaluate at
                                    // (clamped to end of background)


    double target_init = k / x_init;
    double N_init = tr.front().N;
    bool found = false;
    for (size_t i = 0; i+1 < tr.size(); ++i) {
        double v0 = tr[i].H   * std::exp(tr[i].N);
        double v1 = tr[i+1].H * std::exp(tr[i+1].N);
        if (v0 <= target_init && v1 >= target_init) {
            double t = (target_init-v0)/(v1-v0);
            N_init = tr[i].N + t*(tr[i+1].N-tr[i].N);
            found = true; break;
        }
    }
    if (!found) {
        // No clean Bunch-Davies start: the mode never crosses k/(aH)=x_init
        // inside the stored trajectory (it begins already superhorizon, or
        // would enter only after the background ends). 
        return -1.0;
    }

    double H0   = interp(tr, N_init, &BGPoint::H);
    double Qs0  = interp(tr, N_init, &BGPoint::Qs_val);
    double QT0  = interp(tr, N_init, &BGPoint::QT_val);
    double cs2_0= interp(tr, N_init, &BGPoint::cs2_val);
    double cT2_0= interp(tr, N_init, &BGPoint::cT2_val);
    double a0   = std::exp(N_init);

    double cs0  = scalar ? std::sqrt(std::max(cs2_0, 1e-10))
                         : std::sqrt(std::max(cT2_0, 1e-10));
    double Q0   = scalar ? Qs0 : QT0/4.0;

    // Amplitude of R_k in WKB
    double amp  = 1.0 / (a0 * std::sqrt(2.0*Q0) * std::sqrt(2.0*cs0*k));

    // WKB phase at IC: integral of cs*k/(aH) dN ≈ cs*k/(aH) = cs*x_init
    double phase = cs0 * x_init;

    Mode y;
    y.Re  =  amp * std::cos(phase);
    y.Im  =  amp * std::sin(phase);
    // d(R)/dN = -i*(cs*k/(aH)) * R  =>  d(Re)/dN = (cs*k/(aH))*Im
    double freq = cs0 * k / H0 * std::exp(-N_init);   // cs*k/(aH) at IC
    y.ReP =  freq * y.Im;
    y.ImP = -freq * y.Re;

    // Integrate until the mode has frozen well outside the horizon 
    
    const double dN = 5e-4;
    double N = N_init;
    double N_exit = -1e30;   // N at which x crosses 1 (horizon exit)
    double x_prev = (k / interp(tr, N, &BGPoint::H)) * std::exp(-N);

    const double N_last = tr.back().N;
    while (N < N_last - dN) {
        double H_ = interp(tr, N, &BGPoint::H);
        double x  = (k / H_) * std::exp(-N);

        // record horizon exit (x passes through 1 from above)
        if (x_prev >= 1.0 && x < 1.0) N_exit = N;
        x_prev = x;

        // freeze condition: superhorizon AND enough e-folds since exit
        if (x < x_freeze && (N - N_exit) >= dN_past) break;

        double dN_step = std::min(dN, N_last - N - dN);
        if (dN_step <= 0) break;
        y = rk4_MS(y, N, dN_step, k, tr, scalar);
        N += dN_step;
    }

    return y.Re*y.Re + y.Im*y.Im;   // |R_k|^2
}

// =============================================================================
// Main
// =============================================================================

int main() {
    auto tr = load_background("background.dat");

    const int    N_k    = 400;   // wider k-range -> more samples

    // k_star: exits horizon 55 e-folds before (change accordingly)
    double N_end  = tr.back().N;
    double N_star = N_end - 55.0;
    double k_star = tr.front().H * std::exp(tr.front().N);
    for (size_t i = 0; i+1 < tr.size(); ++i) {
        if (tr[i].N <= N_star && tr[i+1].N >= N_star) {
            double t  = (N_star - tr[i].N)/(tr[i+1].N - tr[i].N);
            double Hs = tr[i].H*(1-t) + tr[i+1].H*t;
            k_star = Hs * std::exp(N_star);
            break;
        }
    }
    const double x_init_g   = 50.0;   // must match solve_mode x_init
    const double x_freeze_g = 0.05;   // must match solve_mode x_freeze
    const double margin     = 2.0;
    double aH_first = tr.front().H * std::exp(tr.front().N);
    double aH_last  = tr.back().H  * std::exp(tr.back().N);
    double k_min = aH_first * x_init_g   * std::exp(+margin);
    double k_max = aH_last  * x_freeze_g * std::exp(-margin);
    if (k_max <= k_min) {           // too short rejected
        k_min = k_star * 1e-2;
        k_max = k_star * 1e2;
    }
    std::cout << "k_star=" << k_star
              << " k_min=" << k_min << " k_max=" << k_max
              << "  (decades=" << std::log10(k_max/k_min) << ")\n";

    std::vector<double> k_grid(N_k);
    double lk0 = std::log(k_min), lk1 = std::log(k_max);
    for (int i=0; i<N_k; ++i)
        k_grid[i] = std::exp(lk0 + i*(lk1-lk0)/(N_k-1));

    std::ofstream out_s("scalar_spectrum.dat");
    std::ofstream out_t("tensor_spectrum.dat");
    out_s << std::setprecision(15);
    out_t << std::setprecision(15);
    out_s << "# k  Rk2  Delta2_s\n";
    out_t << "# k  hk2  Delta2_t\n";

    std::vector<double> P_s(N_k), P_t(N_k);
    double P_s_star=0, P_t_star=0;
    double k_star_dist=1e99;
    int i_star=0;

    std::cout << "Solving " << N_k << " modes...\n";

    for (int i=0; i<N_k; ++i) {
        double k = k_grid[i];
        double Rk2 = solve_mode(k, true,  tr);
        double hk2 = solve_mode(k, false, tr);

        // skip modes without a clean Bunch-Davies start / valid solve
        if (Rk2 < 0.0 || hk2 < 0.0 ||
            !std::isfinite(Rk2) || !std::isfinite(hk2)) {
            P_s[i] = -1.0; P_t[i] = -1.0;
            continue;
        }

        double pref = k*k*k / (2.0*M_PI*M_PI);
        P_s[i] = pref * Rk2;
        P_t[i] = pref * hk2 * 4.0;

        // out_s << k << " " << Rk2 << " " << P_s[i] << "\n";
        out_s << k/k_star <<  " " << P_s[i] << "\n";
        //out_t << k << " " << hk2 << " " << P_t[i] << "\n";
        out_t << k/k_star <<  " " << P_t[i] << "\n";


        if (std::fabs(k-k_star) < k_star_dist) {
            k_star_dist = std::fabs(k-k_star);
            i_star = i;
            P_s_star = P_s[i];
            P_t_star = P_t[i];
        }

        //if (i%10==0)
            //std::cout << "  k=" << k
                      //<< "  D2_s=" << P_s[i]
                      //<< "  D2_t=" << P_t[i] << "\n";
    }
    

    // Spectral indices 
    int i0=std::max(0,i_star-3), i1=std::min(N_k-1,i_star+3);
    // nudge i0/i1 off any skipped (-1) samples
    while (i0 < i_star && P_s[i0] <= 0.0) ++i0;
    while (i1 > i_star && P_s[i1] <= 0.0) --i1;
    double ns=1.0, nT=0.0;
    if (i1>i0 && P_s[i0]>0 && P_s[i1]>0)
        ns = 1.0 + (std::log(P_s[i1])-std::log(P_s[i0]))
                  /(std::log(k_grid[i1])-std::log(k_grid[i0]));
    if (P_t[i0]>0 && P_t[i1]>0)
        nT = (std::log(P_t[i1])-std::log(P_t[i0]))
            /(std::log(k_grid[i1])-std::log(k_grid[i0]));

    double r_eff = (P_s_star>0) ? P_t_star/P_s_star : 0.0;

    std::ofstream sum("spectra_summary.dat");
    sum << std::setprecision(15);
    sum << "# Primordial power spectrum summary\n";
    sum << "# k_star = " << k_star << "\n";
    sum << "A_s = " << P_s_star << "\n";
    sum << "n_s = " << ns       << "\n";
    sum << "r   = " << r_eff    << "\n";
    sum << "A_T = " << P_t_star << "\n";
    sum << "n_T = " << nT       << "\n";

    std::cout << "Results at k_star \n"
              << "  A_s = " << P_s_star << "\n"
              << "  n_s = " << ns       << "\n"
              << "  r   = " << r_eff    << "\n"
              << "  n_T = " << nT       << "\n";

    return 0;
}

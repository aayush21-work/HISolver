// background.cpp — with cs2 output
#include "model_potential.hpp"
#include "model_generated.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

static double H_of(double phi, double dphi) {
    double rho = 0.5*dphi*dphi + V_phi(phi);
    if (rho <= 0.0) throw std::runtime_error("H_of: rho<=0");
    double H = std::sqrt(rho / 3.0);
    for (int i = 0; i < 50; ++i) {
        double f  = 3.0*H*H - rho_phi(phi, dphi, H);
        double dh = 1e-7*H;
        double fp = (3.0*(H+dh)*(H+dh) - rho_phi(phi,dphi,H+dh)
                   - 3.0*(H-dh)*(H-dh) + rho_phi(phi,dphi,H-dh)) / (2.0*dh);
        double step = -f/fp;
        H += step;
        if (std::fabs(step) < 1e-12*H) break;
    }
    return H;
}

static void rhs(double phi, double dphi, double &phi_N, double &dphi_N) {
    double H   = H_of(phi, dphi);
    double eps = epsilon_H(phi, dphi, H);
    double dH  = -eps * H * H;
    double dd  = ddphi_rhs(phi, dphi, H, dH);
    phi_N  = dphi / H;
    dphi_N = dd   / H;
}

static double slow_roll_dphi(double phi) {
    double H  = std::sqrt(V_phi(phi)/3.0);
    double dp = -dV_phi(phi)/(3.0*H);
    for (int i = 0; i < 5; ++i) { H = H_of(phi,dp); dp = -dV_phi(phi)/(3.0*H); }
    return dp;
}

int main() {
    double phi  = 4.5;
    double dphi = slow_roll_dphi(phi);
    double N    = 0.0;
    const double dN=1e-4; const int we=10; const double Nmax=500;

    std::ofstream out("background.dat");
    out << std::setprecision(15);
    out << "# N  phi  dphi  H  logH  epsilon  QT  cT2  Qs  cs2\n";

    auto write_row = [&](double N_, double phi_, double dphi_) {
        double H_   = H_of(phi_, dphi_);
        double eps_ = epsilon_H(phi_, dphi_, H_);
        double dH_  = -eps_ * H_ * H_;
        double dd_  = ddphi_rhs(phi_, dphi_, H_, dH_);
        double cs2_ = cs2(phi_, dphi_, H_, dH_, dd_);
        out << N_   << " " << phi_  << " " << dphi_ << " "
            << H_   << " " << std::log(H_) << " " << eps_ << " "
            << QT (phi_,dphi_,H_) << " "
            << cT2(phi_,dphi_,H_) << " "
            << Qs (phi_,dphi_,H_) << " "
            << cs2_ << "\n";
    };

    write_row(N, phi, dphi);
    int step = 0;
    while (true) {
        double H_   = H_of(phi, dphi);
        double eps_ = epsilon_H(phi, dphi, H_);
        if (eps_ >= 1.0 || N >= Nmax) break;
        double k1p,k1d,k2p,k2d,k3p,k3d,k4p,k4d;
        rhs(phi,              dphi,              k1p,k1d);
        rhs(phi+0.5*dN*k1p,  dphi+0.5*dN*k1d,  k2p,k2d);
        rhs(phi+0.5*dN*k2p,  dphi+0.5*dN*k2d,  k3p,k3d);
        rhs(phi+    dN*k3p,  dphi+    dN*k3d,  k4p,k4d);
        phi  += (dN/6)*(k1p+2*k2p+2*k3p+k4p);
        dphi += (dN/6)*(k1d+2*k2d+2*k3d+k4d);
        N += dN; ++step;
        if (step%we==0) write_row(N,phi,dphi);
    }
    write_row(N,phi,dphi);

    double H_end = H_of(phi,dphi);
    double e_end = epsilon_H(phi,dphi,H_end);
    std::cout<<"Inflation ended:\n"
             <<"  N_tot="<<N<<"\n  phi="<<phi<<"\n  H="<<H_end
             <<"\n  epsilon="<<e_end
             <<"\n  QT="<<QT(phi,dphi,H_end)
             <<"\n  cT2="<<cT2(phi,dphi,H_end)
             <<"\n  Qs="<<Qs(phi,dphi,H_end)<<"\n";
    return 0;
}

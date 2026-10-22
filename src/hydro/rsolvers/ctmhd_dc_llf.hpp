#ifndef RSOLVERS_CTMHD_DC_LLF_HPP_
#define RSOLVERS_CTMHD_DC_LLF_HPP_

#include "../../eos/adiabatic_ctmhd.hpp"
#include "../../main.hpp"
#include "interface/variable_pack.hpp"
#include "rsolvers.hpp"

using parthenon::Real;

struct CTMHDCons1D {
  Real d, mx, my, mz, e, by, bz;
};

constexpr int NCTMHD = 8;

template <>
struct Riemann<Fluid::ctmhd, RiemannSolver::llf> {
  static KOKKOS_INLINE_FUNCTION void Solve(const AdiabaticCTMHDEOS &eos, const int k,
                                           const int j, const int i, const int ivx,
                                           const VariablePack<Real> &prim,
                                           VariableFluxPack<Real> &cons, const Real) {
    const int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    const int ivz = IV1 + ((ivx - IV1) + 2) % 3;
    const int iBx = ivx - 1 + NHYDRO;
    const int iBy = ivy - 1 + NHYDRO;
    const int iBz = ivz - 1 + NHYDRO;
    const Real igm1 = 1.0 / (eos.GetGamma() - 1.0);
    const int il = (ivx == 1) ? i - 1 : i;
    const int jl = (ivx == 2) ? j - 1 : j;
    const int kl = (ivx == 3) ? k - 1 : k;

    Real wl[NCTMHD], wr[NCTMHD];
    wl[IDN] = prim(IDN, kl, jl, il);
    wl[IV1] = prim(ivx, kl, jl, il);
    wl[IV2] = prim(ivy, kl, jl, il);
    wl[IV3] = prim(ivz, kl, jl, il);
    wl[IPR] = prim(IPR, kl, jl, il);
    wl[IB1] = prim(iBx, kl, jl, il);
    wl[IB2] = prim(iBy, kl, jl, il);
    wl[IB3] = prim(iBz, kl, jl, il);
    wr[IDN] = prim(IDN, k, j, i);
    wr[IV1] = prim(ivx, k, j, i);
    wr[IV2] = prim(ivy, k, j, i);
    wr[IV3] = prim(ivz, k, j, i);
    wr[IPR] = prim(IPR, k, j, i);
    wr[IB1] = prim(iBx, k, j, i);
    wr[IB2] = prim(iBy, k, j, i);
    wr[IB3] = prim(iBz, k, j, i);

    const Real bx = 0.5 * (wl[IB1] + wr[IB1]);
    const Real ql = 0.5 * (SQR(wl[IB2]) + SQR(wl[IB3]) - SQR(bx));
    const Real qr = 0.5 * (SQR(wr[IB2]) + SQR(wr[IB3]) - SQR(bx));
    const Real el = wl[IPR] * igm1 + 0.5 * wl[IDN] *
        (SQR(wl[IV1]) + SQR(wl[IV2]) + SQR(wl[IV3])) + ql + SQR(bx);
    const Real er = wr[IPR] * igm1 + 0.5 * wr[IDN] *
        (SQR(wr[IV1]) + SQR(wr[IV2]) + SQR(wr[IV3])) + qr + SQR(bx);

    CTMHDCons1D f;
    f.d = wl[IDN] * wl[IV1] + wr[IDN] * wr[IV1];
    f.mx = wl[IDN] * SQR(wl[IV1]) + wr[IDN] * SQR(wr[IV1]) + ql + qr +
           wl[IPR] + wr[IPR];
    f.my = wl[IDN] * wl[IV1] * wl[IV2] + wr[IDN] * wr[IV1] * wr[IV2] -
           bx * (wl[IB2] + wr[IB2]);
    f.mz = wl[IDN] * wl[IV1] * wl[IV3] + wr[IDN] * wr[IV1] * wr[IV3] -
           bx * (wl[IB3] + wr[IB3]);
    f.by = wl[IB2] * wl[IV1] + wr[IB2] * wr[IV1] - bx * (wl[IV2] + wr[IV2]);
    f.bz = wl[IB3] * wl[IV1] + wr[IB3] * wr[IV1] - bx * (wl[IV3] + wr[IV3]);
    f.e = (el + wl[IPR] + ql) * wl[IV1] + (er + wr[IPR] + qr) * wr[IV1] -
          bx * (wl[IB2] * wl[IV2] + wl[IB3] * wl[IV3] +
               wr[IB2] * wr[IV2] + wr[IB3] * wr[IV3]);

    const Real a = fmax(fabs(wl[IV1]) + eos.FastMagnetosonicSpeed(
        wl[IDN], wl[IPR], wl[IB1], wl[IB2], wl[IB3]),
        fabs(wr[IV1]) + eos.FastMagnetosonicSpeed(
        wr[IDN], wr[IPR], wr[IB1], wr[IB2], wr[IB3]));
    cons.flux(ivx, IDN, k, j, i) = 0.5 * (f.d - a * (wr[IDN] - wl[IDN]));
    cons.flux(ivx, ivx, k, j, i) = 0.5 * (f.mx - a *
        (wr[IDN] * wr[IV1] - wl[IDN] * wl[IV1]));
    cons.flux(ivx, ivy, k, j, i) = 0.5 * (f.my - a *
        (wr[IDN] * wr[IV2] - wl[IDN] * wl[IV2]));
    cons.flux(ivx, ivz, k, j, i) = 0.5 * (f.mz - a *
        (wr[IDN] * wr[IV3] - wl[IDN] * wl[IV3]));
    cons.flux(ivx, IEN, k, j, i) = 0.5 * (f.e - a * (er - el));
    cons.flux(ivx, iBx, k, j, i) = 0.0;
    cons.flux(ivx, iBy, k, j, i) = 0.5 * (f.by - a * (wr[IB2] - wl[IB2]));
    cons.flux(ivx, iBz, k, j, i) = 0.5 * (f.bz - a * (wr[IB3] - wl[IB3]));

    for (auto n = Hydro::GetNVars<Fluid::ctmhd>(); n < cons.GetDim(4); ++n) {
      const auto source = (cons.flux(ivx, IDN, k, j, i) >= 0.0) ?
          prim(n, kl, jl, il) : prim(n, k, j, i);
      cons.flux(ivx, n, k, j, i) = cons.flux(ivx, IDN, k, j, i) * source;
    }
  }
};

#endif // RSOLVERS_CTMHD_DC_LLF_HPP_

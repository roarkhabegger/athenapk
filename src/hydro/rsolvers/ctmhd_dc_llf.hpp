#ifndef RSOLVERS_CTMHD_DC_LLF_HPP_
#define RSOLVERS_CTMHD_DC_LLF_HPP_

#include "../../eos/adiabatic_ctmhd.hpp"
#include "../../main.hpp"
#include "interface/variable_pack.hpp"
#include "rsolvers.hpp"

using parthenon::Real;

// LLF Riemann solver for CTMHD with dc reconstruction.
//
// Besides the conserved fluxes, this solver stores the upwind induction fluxes of the
// transverse field components in the magnetic entries of cons.flux:
//   cons.flux(n, B_{n+1}) = F_n(B_{n+1}) = v_n B_{n+1} - B_n v_{n+1}
//   cons.flux(n, B_{n+2}) = F_n(B_{n+2}) = v_n B_{n+2} - B_n v_{n+2}
// with LLF dissipation. The normal component flux is zero by construction.
//
// These fluxes define the EMFs used by constrained transport, E = -v x B:
//   E_{n+2} = -F_n(B_{n+1}),  E_{n+1} = +F_n(B_{n+2}).
// The energy flux's magnetic (Poynting) part is built from the same numbers, so the
// CT face update and the total energy update see one consistent upwind state.

struct CTMHDCons1D {
  Real d, mx, my, mz, e, by, bz;
};

constexpr int NCTMHD = 8;

// Upwind weight for the Gardiner-Stone corner EMF average, as in Athena++
// (Hydro::GetWeightForCT). It is 1 for strong mass flux in +x, 0 for strong flux in -x,
// and 0.5 when the flux is negligible on the timescale dt across a cell of width dx.
KOKKOS_INLINE_FUNCTION Real GetWeightForCT(const Real dflx, const Real rhol,
                                           const Real rhor, const Real dx,
                                           const Real dt) {
  const Real v_over_c = 1024.0 * dt * dflx / (dx * (rhol + rhor));
  const Real tmp_min = fmin(0.5, v_over_c);
  return 0.5 + fmax(-0.5, tmp_min);
}

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
    f.e = (el + wl[IPR] + ql) * wl[IV1] +
          (er + wr[IPR] + qr) * wr[IV1] -
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

    for (auto n = Hydro::GetNVars<Fluid::ctmhd>(); n < cons.GetDim(4); ++n) {
      const auto source = (cons.flux(ivx, IDN, k, j, i) >= 0.0) ?
          prim(n, kl, jl, il) : prim(n, k, j, i);
      cons.flux(ivx, n, k, j, i) = cons.flux(ivx, IDN, k, j, i) * source;
    }
  }
};

// Induction EMFs and Gardiner-Stone weight for the face normal to ivx. These use the same
// reconstructed states and LLF dissipation as the ctmhd flux, following Athena++
// llf_mhd.cpp: the induction flux of B_y gives ey = -flx(IBY) and that of B_z gives
// ez = +flx(IBZ). emf_y and emf_z are the two transverse EMF components on this face,
// and wct is the upwind weight from the LLF mass flux through the face.
KOKKOS_INLINE_FUNCTION void ComputeCTMHDFaceEMF(const AdiabaticCTMHDEOS &eos, const int k,
                                                const int j, const int i, const int ivx,
                                                const VariablePack<Real> &prim,
                                                const Real dx, const Real dt,
                                                Real &emf_y, Real &emf_z, Real &wct) {
  const int ivy = IV1 + ((ivx - IV1) + 1) % 3;
  const int ivz = IV1 + ((ivx - IV1) + 2) % 3;
  const int iBx = ivx - 1 + NHYDRO;
  const int iBy = ivy - 1 + NHYDRO;
  const int iBz = ivz - 1 + NHYDRO;
  const int il = (ivx == 1) ? i - 1 : i;
  const int jl = (ivx == 2) ? j - 1 : j;
  const int kl = (ivx == 3) ? k - 1 : k;

  const Real rhol = prim(IDN, kl, jl, il);
  const Real rhor = prim(IDN, k, j, i);
  const Real vxl = prim(ivx, kl, jl, il);
  const Real vxr = prim(ivx, k, j, i);
  const Real vyl = prim(ivy, kl, jl, il);
  const Real vyr = prim(ivy, k, j, i);
  const Real vzl = prim(ivz, kl, jl, il);
  const Real vzr = prim(ivz, k, j, i);
  const Real byl = prim(iBy, kl, jl, il);
  const Real byr = prim(iBy, k, j, i);
  const Real bzl = prim(iBz, kl, jl, il);
  const Real bzr = prim(iBz, k, j, i);
  const Real bxl = prim(iBx, kl, jl, il);
  const Real bxr = prim(iBx, k, j, i);
  const Real bx = 0.5 * (bxl + bxr);

  // Induction fluxes F_x(B_y), F_x(B_z), built as in the flux above.
  const Real fy_l = byl * vxl - bx * vyl;
  const Real fy_r = byr * vxr - bx * vyr;
  const Real fz_l = bzl * vxl - bx * vzl;
  const Real fz_r = bzr * vxr - bx * vzr;

  const Real wl = eos.FastMagnetosonicSpeed(rhol, prim(IPR, kl, jl, il), bxl, byl, bzl);
  const Real wr = eos.FastMagnetosonicSpeed(rhor, prim(IPR, k, j, i), bxr, byr, bzr);
  const Real a = fmax(fabs(vxl) + wl, fabs(vxr) + wr);

  const Real flx_y = 0.5 * (fy_l + fy_r) - 0.5 * a * (byr - byl);
  const Real flx_z = 0.5 * (fz_l + fz_r) - 0.5 * a * (bzr - bzl);
  emf_y = -flx_y;
  emf_z = flx_z;

  // Weight from the mass flux through this face, as in Athena++ llf_mhd.cpp.
  const Real mass_flux = 0.5 * (rhol * vxl + rhor * vxr) - 0.5 * a * (rhor - rhol);
  wct = GetWeightForCT(mass_flux, rhol, rhor, dx, dt);
}

#endif // RSOLVERS_CTMHD_DC_LLF_HPP_

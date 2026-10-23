#ifndef EOS_ADIABATIC_CTMHD_HPP_
#define EOS_ADIABATIC_CTMHD_HPP_

#include "mesh/mesh.hpp"

#include "../main.hpp"
#include "eos.hpp"

using parthenon::MeshData;
using parthenon::Real;

class AdiabaticCTMHDEOS : public EquationOfState {
 public:
  AdiabaticCTMHDEOS(Real pressure_floor, Real density_floor, Real internal_e_floor,
                    Real velocity_ceiling, Real internal_e_ceiling, Real gamma)
      : EquationOfState(pressure_floor, density_floor, internal_e_floor, velocity_ceiling,
                        internal_e_ceiling),
        gamma_{gamma} {}

  void ConservedToPrimitive(MeshData<Real> *md) const override;

  KOKKOS_INLINE_FUNCTION
  Real GetGamma() const { return gamma_; }

  KOKKOS_INLINE_FUNCTION
  Real FastMagnetosonicSpeed(const Real d, const Real p, const Real bx, const Real by,
                             const Real bz) const {
    const Real asq = gamma_ * p;
    const Real ct2 = by * by + bz * bz;
    const Real qsq = bx * bx + ct2 + asq;
    const Real tmp = bx * bx + ct2 - asq;
    return std::sqrt(0.5 * (qsq + std::sqrt(tmp * tmp + 4.0 * asq * ct2)) / d);
  }

  template <typename View4D>
  KOKKOS_INLINE_FUNCTION void ConsToPrim(View4D cons, View4D prim, const int &nhydro,
                                         const int &nscalars, const int &k, const int &j,
                                         const int &i) const {
    const Real gm1 = gamma_ - 1.0;
    const Real density_floor = GetDensityFloor();
    const Real pressure_floor = GetPressureFloor();
    const Real e_floor = GetInternalEFloor();
    const Real velocity_ceiling = GetVelocityCeiling();
    const Real e_ceiling = GetInternalECeiling();

    Real &u_d = cons(IDN, k, j, i);
    Real &u_m1 = cons(IM1, k, j, i);
    Real &u_m2 = cons(IM2, k, j, i);
    Real &u_m3 = cons(IM3, k, j, i);
    Real &u_e = cons(IEN, k, j, i);
    Real &w_d = prim(IDN, k, j, i);
    Real &w_vx = prim(IV1, k, j, i);
    Real &w_vy = prim(IV2, k, j, i);
    Real &w_vz = prim(IV3, k, j, i);
    Real &w_p = prim(IPR, k, j, i);
    // cons(IB*) is synchronized from face-centered ct_b before EOS conversion
    // and remains the single authoritative cell-center B representation
    const Real b1 = cons(IB1, k, j, i);
    const Real b2 = cons(IB2, k, j, i);
    const Real b3 = cons(IB3, k, j, i);
    Real &w_Bx = prim(IB1, k, j, i);
    Real &w_By = prim(IB2, k, j, i);
    Real &w_Bz = prim(IB3, k, j, i);

    PARTHENON_REQUIRE(u_d > 0.0 || density_floor > 0.0,
                      "Got negative density in CT MHD.");
    u_d = (u_d > density_floor) ? u_d : density_floor;
    w_d = u_d;

    const Real di = 1.0 / u_d;
    w_vx = u_m1 * di;
    w_vy = u_m2 * di;
    w_vz = u_m3 * di;
    w_Bx = b1;
    w_By = b2;
    w_Bz = b3;

    Real e_k = 0.5 * di * (SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
    const Real e_B = 0.5 * (SQR(b1) + SQR(b2) + SQR(b3));
    w_p = gm1 * (u_e - e_k - e_B);

    const Real w_v2 = SQR(w_vx) + SQR(w_vy) + SQR(w_vz);
    if (w_v2 > SQR(velocity_ceiling)) {
      const Real w_v = sqrt(w_v2);
      const Real scale = velocity_ceiling / w_v;
      w_vx *= scale;
      w_vy *= scale;
      w_vz *= scale;
      u_m1 *= scale;
      u_m2 *= scale;
      u_m3 *= scale;
      const Real e_k_new = 0.5 * u_d * SQR(velocity_ceiling);
      u_e -= e_k - e_k_new;
      e_k = e_k_new;
    }

    PARTHENON_REQUIRE(w_p > 0.0 || pressure_floor > 0.0 || e_floor > 0.0,
                      "Got negative pressure in CT MHD.");
    if (pressure_floor > 0.0 && w_p < pressure_floor) {
      u_e = pressure_floor / gm1 + e_k + e_B;
      w_p = pressure_floor;
    }
    const Real effective_pressure_floor = gm1 * u_d * e_floor;
    if (w_p < effective_pressure_floor) {
      u_e = u_d * e_floor + e_k + e_B;
      w_p = effective_pressure_floor;
    }
    const Real effective_pressure_ceiling = gm1 * u_d * e_ceiling;
    if (w_p > effective_pressure_ceiling) {
      u_e = u_d * e_ceiling + e_k + e_B;
      w_p = effective_pressure_ceiling;
    }

    for (auto n = nhydro; n < nhydro + nscalars; ++n) {
      prim(n, k, j, i) = cons(n, k, j, i) * di;
    }
  }

 private:
  Real gamma_;
};

#endif // EOS_ADIABATIC_CTMHD_HPP_

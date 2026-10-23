#ifndef HYDRO_CTMHD_CT_UPDATE_HPP_
#define HYDRO_CTMHD_CT_UPDATE_HPP_

#include "interface/variable_pack.hpp"
#include "kokkos_abstraction.hpp"

#include "../hydro.hpp"

namespace Hydro {

inline TaskStatus InitializeCTFaceField(MeshData<Real> *md) {
  using TE = parthenon::TopologicalElement;
  auto cons = md->PackVariables(std::vector<std::string>{"cons"});
  auto face_b = md->PackVariables(std::vector<std::string>{"ct_b"});
  const auto ib = md->GetBoundsI(parthenon::CellLevel::same, IndexDomain::entire);
  const auto jb = md->GetBoundsJ(parthenon::CellLevel::same, IndexDomain::entire);
  const auto kb = md->GetBoundsK(parthenon::CellLevel::same, IndexDomain::entire);

  const auto initialize_face = [&](const TE te, const int component) {
    const auto face_ib = md->GetBoundsI(parthenon::CellLevel::same, IndexDomain::entire, te);
    const auto face_jb = md->GetBoundsJ(parthenon::CellLevel::same, IndexDomain::entire, te);
    const auto face_kb = md->GetBoundsK(parthenon::CellLevel::same, IndexDomain::entire, te);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Initialize CT face field", parthenon::DevExecSpace(), 0,
        face_b.GetDim(5) - 1, face_kb.s, face_kb.e, face_jb.s, face_jb.e, face_ib.s,
        face_ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          int kl = k;
          int kr = k;
          int jl = j;
          int jr = j;
          int il = i;
          int ir = i;
          if (te == TE::F1) {
            il = i - 1;
            if (il < ib.s) il = ib.s;
            if (ir > ib.e) ir = ib.e;
          } else if (te == TE::F2) {
            jl = j - 1;
            if (jl < jb.s) jl = jb.s;
            if (jr > jb.e) jr = jb.e;
          } else {
            kl = k - 1;
            if (kl < kb.s) kl = kb.s;
            if (kr > kb.e) kr = kb.e;
          }
          face_b(b, te, 0, k, j, i) =
              0.5 * (cons(b, IB1 + component, kl, jl, il) +
                     cons(b, IB1 + component, kr, jr, ir));
        });
  };

  initialize_face(TE::F1, 0);
  initialize_face(TE::F2, 1);
  initialize_face(TE::F3, 2);
  return TaskStatus::complete;
}

inline void InitializeCTFields(MeshData<Real> *md) {
  InitializeCTFaceField(md);
}

inline TaskStatus CalculateCTEMFs(MeshData<Real> *md) {
  using TE = parthenon::TopologicalElement;
  auto prim = md->PackVariables(std::vector<std::string>{"prim"});
  auto face_b = md->PackVariablesAndFluxes(
      std::vector<parthenon::MetadataFlag>{Metadata::Independent, Metadata::Face});
  const int ndim = md->GetMeshPointer()->ndim;

  auto edge_emf = KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                                const int edge) {
    Real emf = 0.0;
    const int k0 = ((edge == 1 || edge == 2) && ndim > 2) ? k - 1 : k;
    const int j0 = ((edge == 1 || edge == 3) && ndim > 1) ? j - 1 : j;
    const int i0 = (edge == 2 || edge == 3) ? i - 1 : i;
    const int nk = ((edge == 1 || edge == 2) && ndim > 2) ? 2 : 1;
    const int nj = ((edge == 1 || edge == 3) && ndim > 1) ? 2 : 1;
    const int ni = (edge == 2 || edge == 3) ? 2 : 1;
    const int ns = nk * nj * ni;
    for (int dk = 0; dk < nk; ++dk) {
      for (int dj = 0; dj < nj; ++dj) {
        for (int di = 0; di < ni; ++di) {
          const int kk = k0 + dk;
          const int jj = j0 + dj;
          const int ii = i0 + di;
          const Real vx = prim(b, IV1, kk, jj, ii);
          const Real vy = prim(b, IV2, kk, jj, ii);
          const Real vz = prim(b, IV3, kk, jj, ii);
          const Real bx = prim(b, IB1, kk, jj, ii);
          const Real by = prim(b, IB2, kk, jj, ii);
          const Real bz = prim(b, IB3, kk, jj, ii);
          emf += (edge == 1) ? -(vy * bz - vz * by)
                 : (edge == 2) ? -(vz * bx - vx * bz)
                               : -(vx * by - vy * bx);
        }
      }
    }
    return emf / static_cast<Real>(ns);
  };

    const auto ib_e1 = md->GetBoundsI(parthenon::CellLevel::same, IndexDomain::interior, TE::E1);
    const auto jb_e1 = md->GetBoundsJ(parthenon::CellLevel::same, IndexDomain::interior, TE::E1);
    const auto kb_e1 = md->GetBoundsK(parthenon::CellLevel::same, IndexDomain::interior, TE::E1);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CT edge EMF x1", parthenon::DevExecSpace(), 0,
      face_b.GetDim(5) - 1, kb_e1.s, kb_e1.e, jb_e1.s, jb_e1.e, ib_e1.s, ib_e1.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        face_b(b).template flux<parthenon::TopologicalType::Edge>(1, 0, k, j, i) =
          edge_emf(b, k, j, i, 1);
      });
  const auto ib_e2 = md->GetBoundsI(parthenon::CellLevel::same, IndexDomain::interior, TE::E2);
  const auto jb_e2 = md->GetBoundsJ(parthenon::CellLevel::same, IndexDomain::interior, TE::E2);
  const auto kb_e2 = md->GetBoundsK(parthenon::CellLevel::same, IndexDomain::interior, TE::E2);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CT edge EMF x2", parthenon::DevExecSpace(), 0,
      face_b.GetDim(5) - 1, kb_e2.s, kb_e2.e, jb_e2.s, jb_e2.e, ib_e2.s, ib_e2.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        face_b(b).template flux<parthenon::TopologicalType::Edge>(2, 0, k, j, i) =
            edge_emf(b, k, j, i, 2);
      });
  const auto ib_e3 = md->GetBoundsI(parthenon::CellLevel::same, IndexDomain::interior, TE::E3);
  const auto jb_e3 = md->GetBoundsJ(parthenon::CellLevel::same, IndexDomain::interior, TE::E3);
  const auto kb_e3 = md->GetBoundsK(parthenon::CellLevel::same, IndexDomain::interior, TE::E3);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CT edge EMF x3", parthenon::DevExecSpace(), 0,
      face_b.GetDim(5) - 1, kb_e3.s, kb_e3.e, jb_e3.s, jb_e3.e, ib_e3.s, ib_e3.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        face_b(b).template flux<parthenon::TopologicalType::Edge>(3, 0, k, j, i) =
            edge_emf(b, k, j, i, 3);
      });
  return TaskStatus::complete;
}

template <typename PackT0, typename PackT1>
inline void CTWeightedSumFace(const Real gam0, const Real gam1,
                              const parthenon::TopologicalElement te,
                              PackT0 &u0, const PackT1 &u1,
                              const MeshData<Real> *data) {
  const auto ib = data->GetBoundsI(parthenon::CellLevel::same, IndexDomain::interior, te);
  const auto jb = data->GetBoundsJ(parthenon::CellLevel::same, IndexDomain::interior, te);
  const auto kb = data->GetBoundsK(parthenon::CellLevel::same, IndexDomain::interior, te);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CT face weighted sum", parthenon::DevExecSpace(), 0,
      u0.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        for (int l = 0; l < u0.GetDim(4); ++l)
          u0(b, te, l, k, j, i) = gam0 * u0(b, te, l, k, j, i) +
                                   gam1 * u1(b, te, l, k, j, i);
      });
}

template <typename PackT>
inline void CTStokesComponent(const Real fac, const parthenon::CellLevel cl,
                              const parthenon::TopologicalElement te_var,
                              const parthenon::TopologicalElement te_flux,
                              const int ndim, const PackT &flux_pack, PackT &out_pack,
                              const MeshData<Real> *data) {
  const int ioff = TopologicalOffsetI(te_flux) - TopologicalOffsetI(te_var);
  const int joff = TopologicalOffsetJ(te_flux) - TopologicalOffsetJ(te_var);
  const int koff = TopologicalOffsetK(te_flux) - TopologicalOffsetK(te_var);
  PARTHENON_REQUIRE(ioff == 0 || ioff == 1, "Bad CT topology in i direction");
  PARTHENON_REQUIRE(joff == 0 || joff == 1, "Bad CT topology in j direction");
  PARTHENON_REQUIRE(koff == 0 || koff == 1, "Bad CT topology in k direction");
  PARTHENON_REQUIRE(ioff + joff + koff == 1, "Bad CT topology combination");
  const int active_joff = ndim > 1 ? joff : 0;
  const int active_koff = ndim > 2 ? koff : 0;
  const auto ib = data->GetBoundsI(cl, IndexDomain::interior, te_var);
  const auto jb = data->GetBoundsJ(cl, IndexDomain::interior, te_var);
  const auto kb = data->GetBoundsK(cl, IndexDomain::interior, te_var);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CT generalized Stokes", parthenon::DevExecSpace(), 0,
      out_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = out_pack.GetCoords(b);
        for (int l = 0; l < out_pack.GetDim(4); ++l) {
          out_pack(b, te_var, l, k, j, i) +=
              fac * (coords.Volume(cl, te_flux, k, j, i) *
                         flux_pack(b).template flux<parthenon::TopologicalType::Edge>(
                           static_cast<int>(te_flux) - 5, l, k, j, i) -
                     coords.Volume(cl, te_flux, k + active_koff, j + active_joff,
                                   i + ioff) *
                         flux_pack(b).template flux<parthenon::TopologicalType::Edge>(
                           static_cast<int>(te_flux) - 5, l, k + active_koff,
                           j + active_joff, i + ioff)) /
              coords.Volume(cl, te_var, k, j, i);
        }
      });
}

inline TaskStatus UpdateCTFaceField(MeshData<Real> *u0_data, MeshData<Real> *u1_data,
                                    const Real gam0, const Real gam1,
                                    const Real beta_dt) {
  using TT = parthenon::TopologicalType;
  using TE = parthenon::TopologicalElement;
  auto u0 = u0_data->PackVariablesAndFluxes(
      std::vector<parthenon::MetadataFlag>{Metadata::Independent, Metadata::Face});
  const auto u1 = u1_data->PackVariables(
      std::vector<parthenon::MetadataFlag>{Metadata::Independent, Metadata::Face});
  const int ndim = u0_data->GetMeshPointer()->ndim;

  for (const auto te : parthenon::GetTopologicalElements(TT::Face))
    CTWeightedSumFace(gam0, gam1, te, u0, u1, u0_data);

  for (const auto te_var : parthenon::GetTopologicalElements(TT::Face)) {
    for (const auto te_flux : parthenon::GetTopologicalElements(TT::Edge)) {
      if (!parthenon::IsSubmanifold(te_flux, te_var)) continue;
      const int indicator = (static_cast<int>(te_flux) -
                             static_cast<int>(te_var) + 3) % 3;
      const Real fac = (indicator == 2) ? -beta_dt : beta_dt;
      CTStokesComponent(fac, parthenon::CellLevel::same, te_var, te_flux, ndim, u0, u0,
            u0_data);
    }
  }
  return TaskStatus::complete;
}

} // namespace Hydro

#endif // HYDRO_CTMHD_CT_UPDATE_HPP_

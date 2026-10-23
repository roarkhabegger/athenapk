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
  const auto ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  const auto initialize_face = [&](const TE te, const int component) {
    const auto face_ib = md->GetBoundsI(parthenon::CellLevel::same, IndexDomain::interior, te);
    const auto face_jb = md->GetBoundsJ(parthenon::CellLevel::same, IndexDomain::interior, te);
    const auto face_kb = md->GetBoundsK(parthenon::CellLevel::same, IndexDomain::interior, te);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Initialize CT face field", parthenon::DevExecSpace(), 0,
        face_b.GetDim(5) - 1, face_kb.s, face_kb.e, face_jb.s, face_jb.e, face_ib.s,
        face_ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const int il = i - 1;
          const int ir = i;
          const int jl = (te == TE::F2) ? j - 1 : j;
          const int jr = (te == TE::F2) ? j : j;
          const int kl = (te == TE::F3) ? k - 1 : k;
          const int kr = (te == TE::F3) ? k : k;
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

inline TaskStatus SyncCTConservedMagneticField(MeshData<Real> *md) {
  using TE = parthenon::TopologicalElement;
  auto cons = md->PackVariables(std::vector<std::string>{"cons"});
  auto face_b = md->PackVariables(std::vector<std::string>{"ct_b"});
  const auto ib = md->GetBoundsI(parthenon::CellLevel::same, IndexDomain::interior);
  const auto jb = md->GetBoundsJ(parthenon::CellLevel::same, IndexDomain::interior);
  const auto kb = md->GetBoundsK(parthenon::CellLevel::same, IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sync CT magnetic field to conserved state",
      parthenon::DevExecSpace(), 0, cons.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        cons(b, IB1, k, j, i) =
            0.5 * (face_b(b, TE::F1, 0, k, j, i) +
                   face_b(b, TE::F1, 0, k, j, i + 1));
        cons(b, IB2, k, j, i) =
            0.5 * (face_b(b, TE::F2, 0, k, j, i) +
                   face_b(b, TE::F2, 0, k, j + 1, i));
        cons(b, IB3, k, j, i) =
            0.5 * (face_b(b, TE::F3, 0, k, j, i) +
                   face_b(b, TE::F3, 0, k + 1, j, i));
      });
  return TaskStatus::complete;
}

// Gardiner-Stone 2007 (GS07) corner EMF combination
// Combines face-centered EMFs computed by the Riemann solver using weights.
// 
// Scheme:
// - Riemann solver outputs face-centered EMFs (E1, E2, E3) and weights (w_E1, w_E2, w_E3)
// - Stored in ct_emf_weights field (6 components: [E1, E2, E3, w_E1, w_E2, w_E3])
// - For each edge: weighted average of four surrounding face EMFs
// - E_edge = (w1*E1 + w2*E2 + w3*E3 + w4*E4) / (w1 + w2 + w3 + w4)
// 
// Expects Riemann solver to populate:
//   ct_emf_weights[0] = E1 face EMF (perpendicular to X1 face, in Y-Z plane)
//   ct_emf_weights[1] = E2 face EMF (perpendicular to X2 face, in X-Z plane)
//   ct_emf_weights[2] = E3 face EMF (perpendicular to X3 face, in X-Y plane)
//   ct_emf_weights[3] = weight for E1 (from X1 mass flux)
//   ct_emf_weights[4] = weight for E2 (from X2 mass flux)
//   ct_emf_weights[5] = weight for E3 (from X3 mass flux)
inline TaskStatus ComputeGardinerStoneEMFs(MeshData<Real> *md) {
  using TE = parthenon::TopologicalElement;
  auto emf_wts = md->PackVariables(std::vector<std::string>{"ct_emf_weights"});
  auto face_b = md->PackVariablesAndFluxes(
      std::vector<parthenon::MetadataFlag>{Metadata::Independent, Metadata::Face});
  const int ndim = md->GetMeshPointer()->ndim;

  // E1 edge (Y-Z plane): weighted average of four surrounding face EMFs
  // Combines EMF from X2 faces (j,j+1) and X3 faces (k,k+1)
  const auto ib_e1 = md->GetBoundsI(parthenon::CellLevel::same, IndexDomain::interior, TE::E1);
  const auto jb_e1 = md->GetBoundsJ(parthenon::CellLevel::same, IndexDomain::interior, TE::E1);
  const auto kb_e1 = md->GetBoundsK(parthenon::CellLevel::same, IndexDomain::interior, TE::E1);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GS07 edge E1", parthenon::DevExecSpace(), 0,
      emf_wts.GetDim(5) - 1, kb_e1.s, kb_e1.e, jb_e1.s, jb_e1.e, ib_e1.s, ib_e1.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        // Four surrounding faces: (j,k), (j+1,k), (j,k+1), (j+1,k+1)
        // E1 comes from X2 faces (j,j+1) and X3 faces (k,k+1)
        const Real e1_jk = emf_wts(b, 0, k, j, i);      // E1 at (j,k)
        const Real e1_jkp = emf_wts(b, 0, k + 1, j, i);  // E1 at (j,k+1)
        const Real e1_jp = emf_wts(b, 0, k, j + 1, i);   // E1 at (j+1,k)
        const Real e1_jkpp = emf_wts(b, 0, k + 1, j + 1, i); // E1 at (j+1,k+1)
        
        const Real w_jk = emf_wts(b, 3, k, j, i);      // weight at (j,k)
        const Real w_jkp = emf_wts(b, 3, k + 1, j, i);
        const Real w_jp = emf_wts(b, 3, k, j + 1, i);
        const Real w_jkpp = emf_wts(b, 3, k + 1, j + 1, i);
        
        const Real w_sum = w_jk + w_jkp + w_jp + w_jkpp;
        const Real e1_edge = (w_sum > 0.0) ? 
            (w_jk * e1_jk + w_jkp * e1_jkp + w_jp * e1_jp + w_jkpp * e1_jkpp) / w_sum :
            0.25 * (e1_jk + e1_jkp + e1_jp + e1_jkpp);
        
        face_b(b).template flux<parthenon::TopologicalType::Edge>(1, 0, k, j, i) = e1_edge;
      });

  // E2 edge (X-Z plane): weighted average using E2 values
  const auto ib_e2 = md->GetBoundsI(parthenon::CellLevel::same, IndexDomain::interior, TE::E2);
  const auto jb_e2 = md->GetBoundsJ(parthenon::CellLevel::same, IndexDomain::interior, TE::E2);
  const auto kb_e2 = md->GetBoundsK(parthenon::CellLevel::same, IndexDomain::interior, TE::E2);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GS07 edge E2", parthenon::DevExecSpace(), 0,
      emf_wts.GetDim(5) - 1, kb_e2.s, kb_e2.e, jb_e2.s, jb_e2.e, ib_e2.s, ib_e2.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real e2_ik = emf_wts(b, 1, k, j, i);
        const Real e2_ikp = emf_wts(b, 1, k + 1, j, i);
        const Real e2_ip = emf_wts(b, 1, k, j, i + 1);
        const Real e2_ikpp = emf_wts(b, 1, k + 1, j, i + 1);
        
        const Real w_ik = emf_wts(b, 4, k, j, i);
        const Real w_ikp = emf_wts(b, 4, k + 1, j, i);
        const Real w_ip = emf_wts(b, 4, k, j, i + 1);
        const Real w_ikpp = emf_wts(b, 4, k + 1, j, i + 1);
        
        const Real w_sum = w_ik + w_ikp + w_ip + w_ikpp;
        const Real e2_edge = (w_sum > 0.0) ?
            (w_ik * e2_ik + w_ikp * e2_ikp + w_ip * e2_ip + w_ikpp * e2_ikpp) / w_sum :
            0.25 * (e2_ik + e2_ikp + e2_ip + e2_ikpp);
        
        face_b(b).template flux<parthenon::TopologicalType::Edge>(2, 0, k, j, i) = e2_edge;
      });

  // E3 edge (X-Y plane): weighted average using E3 values
  const auto ib_e3 = md->GetBoundsI(parthenon::CellLevel::same, IndexDomain::interior, TE::E3);
  const auto jb_e3 = md->GetBoundsJ(parthenon::CellLevel::same, IndexDomain::interior, TE::E3);
  const auto kb_e3 = md->GetBoundsK(parthenon::CellLevel::same, IndexDomain::interior, TE::E3);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GS07 edge E3", parthenon::DevExecSpace(), 0,
      emf_wts.GetDim(5) - 1, kb_e3.s, kb_e3.e, jb_e3.s, jb_e3.e, ib_e3.s, ib_e3.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real e3_ij = emf_wts(b, 2, k, j, i);
        const Real e3_ijp = emf_wts(b, 2, k, j + 1, i);
        const Real e3_ip = emf_wts(b, 2, k, j, i + 1);
        const Real e3_ijpp = emf_wts(b, 2, k, j + 1, i + 1);
        
        const Real w_ij = emf_wts(b, 5, k, j, i);
        const Real w_ijp = emf_wts(b, 5, k, j + 1, i);
        const Real w_ip = emf_wts(b, 5, k, j, i + 1);
        const Real w_ijpp = emf_wts(b, 5, k, j + 1, i + 1);
        
        const Real w_sum = w_ij + w_ijp + w_ip + w_ijpp;
        const Real e3_edge = (w_sum > 0.0) ?
            (w_ij * e3_ij + w_ijp * e3_ijp + w_ip * e3_ip + w_ijpp * e3_ijpp) / w_sum :
            0.25 * (e3_ij + e3_ijp + e3_ip + e3_ijpp);
        
        face_b(b).template flux<parthenon::TopologicalType::Edge>(3, 0, k, j, i) = e3_edge;
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

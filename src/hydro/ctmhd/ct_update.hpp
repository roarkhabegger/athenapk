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

// Cell-centered EMF E = -(v x B) for component comp (0: E1, 1: E2, 2: E3), with B from
// cons(IB*), which plays the role of Athena++ bcc. Same sign convention as
// calculate_corner_e.cpp (Newtonian branch).
template <typename P>
KOKKOS_INLINE_FUNCTION Real CellCenteredEMF(const int comp, const P &prim, const P &cons,
                                            const int b, const int k, const int j,
                                            const int i) {
  const Real vx = prim(b, IV1, k, j, i);
  const Real vy = prim(b, IV2, k, j, i);
  const Real vz = prim(b, IV3, k, j, i);
  const Real bx = cons(b, IB1, k, j, i);
  const Real by = cons(b, IB2, k, j, i);
  const Real bz = cons(b, IB3, k, j, i);
  if (comp == 0) return vz * by - vy * bz;
  if (comp == 1) return vx * bz - vz * bx;
  return vy * bx - vx * by;
}

// One weighted correction term from calculate_corner_e.cpp:
//   (1 - w) * (face_a - cc_a) + w * (face_b - cc_b)
KOKKOS_INLINE_FUNCTION Real WeightedCorrection(const Real w, const Real face_a,
                                               const Real cc_a, const Real face_b,
                                               const Real cc_b) {
  return (1.0 - w) * (face_a - cc_a) + w * (face_b - cc_b);
}

// Gardiner-Stone (2007) corner EMF, 3D form of Athena++ calculate_corner_e.cpp.
//
// Each edge is 0.25 times the sum of four corrected terms: two face-EMF terms from each
// perpendicular face, each corrected toward the cell-centered EMF with the upwind weights.
// Inputs in ct_emf_weights (9 components per face, see the registration in hydro.cpp):
//   x1 face: [E2 (0), E3 (1), w (2)]   x2 face: [E1 (3), E3 (4), w (5)]
//   x3 face: [E1 (6), E2 (7), w (8)]
// Output: flux<Edge>(dir) on ct_b, dir 1, 2, 3 for E1, E2, E3.
//
// Dimensions follow Athena++ calculate_corner_e.cpp:
//   1D: every edge is copied from the face EMF that lies on it (no averaging).
//   2D: E3 uses the corner average over x1 and x2 faces; E1 and E2 are copied from the
//       x2 and x1 face EMFs, as the k-direction has no extent.
//   3D: full four-correction stencil for E1, E2, E3.
inline TaskStatus ComputeGardinerStoneEMFs(MeshData<Real> *md) {
  using TE = parthenon::TopologicalElement;
  const int ndim = md->GetMeshPointer()->ndim;
  auto wt = md->PackVariables(std::vector<std::string>{"ct_emf_weights"});
  auto prim = md->PackVariables(std::vector<std::string>{"prim"});
  auto cons = md->PackVariables(std::vector<std::string>{"cons"});
  auto face_b = md->PackVariablesAndFluxes(
      std::vector<parthenon::MetadataFlag>{Metadata::Independent, Metadata::Face});

  if (ndim == 1) {
    // E2 and E3 on the x1 edges are the x1-face EMFs (slots 0 and 1), copied through.
    const auto ib = md->GetBoundsI(parthenon::CellLevel::same, IndexDomain::interior, TE::E2);
    const auto jb = md->GetBoundsJ(parthenon::CellLevel::same, IndexDomain::interior, TE::E2);
    const auto kb = md->GetBoundsK(parthenon::CellLevel::same, IndexDomain::interior, TE::E2);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "GS07 1D copy E2/E3", parthenon::DevExecSpace(), 0,
        wt.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          face_b(b).template flux<parthenon::TopologicalType::Edge>(2, 0, k, j, i) =
              wt(b, 0, k, j, i);
          face_b(b).template flux<parthenon::TopologicalType::Edge>(3, 0, k, j, i) =
              wt(b, 1, k, j, i);
        });
    return TaskStatus::complete;
  }

  if (ndim == 2) {
    // E1 and E2 do not vary in k, so they are copied from the face EMFs: E1 from the x2
    // face (slot 3) and E2 from the x1 face (slot 0). E3 uses the x1/x2 corner average.
    const auto ib1 = md->GetBoundsI(parthenon::CellLevel::same, IndexDomain::interior, TE::E1);
    const auto jb1 = md->GetBoundsJ(parthenon::CellLevel::same, IndexDomain::interior, TE::E1);
    const auto kb1 = md->GetBoundsK(parthenon::CellLevel::same, IndexDomain::interior, TE::E1);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "GS07 2D copy E1/E2", parthenon::DevExecSpace(), 0,
        wt.GetDim(5) - 1, kb1.s, kb1.e, jb1.s, jb1.e, ib1.s, ib1.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          face_b(b).template flux<parthenon::TopologicalType::Edge>(1, 0, k, j, i) =
              wt(b, 3, k, j, i);
          face_b(b).template flux<parthenon::TopologicalType::Edge>(2, 0, k, j, i) =
              wt(b, 0, k, j, i);
        });

    const auto ib3 = md->GetBoundsI(parthenon::CellLevel::same, IndexDomain::interior, TE::E3);
    const auto jb3 = md->GetBoundsJ(parthenon::CellLevel::same, IndexDomain::interior, TE::E3);
    const auto kb3 = md->GetBoundsK(parthenon::CellLevel::same, IndexDomain::interior, TE::E3);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "GS07 2D corner E3", parthenon::DevExecSpace(), 0,
        wt.GetDim(5) - 1, kb3.s, kb3.e, jb3.s, jb3.e, ib3.s, ib3.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const Real e3x1_j = wt(b, 1, k, j, i), e3x1_jm = wt(b, 1, k, j - 1, i);
          const Real e3x2_i = wt(b, 4, k, j, i), e3x2_im = wt(b, 4, k, j, i - 1);
          const Real w_x1_j = wt(b, 2, k, j, i), w_x1_jm = wt(b, 2, k, j - 1, i);
          const Real w_x2_i = wt(b, 5, k, j, i), w_x2_im = wt(b, 5, k, j, i - 1);
          const Real cc_j = CellCenteredEMF(2, prim, cons, b, k, j, i);
          const Real cc_jm = CellCenteredEMF(2, prim, cons, b, k, j - 1, i);
          const Real cc_i = CellCenteredEMF(2, prim, cons, b, k, j, i - 1);
          const Real cc_jmi = CellCenteredEMF(2, prim, cons, b, k, j - 1, i - 1);
          const Real de3_l2 = WeightedCorrection(w_x1_jm, e3x2_i, cc_jm, e3x2_im, cc_jmi);
          const Real de3_r2 = WeightedCorrection(w_x1_j, e3x2_i, cc_j, e3x2_im, cc_i);
          const Real de3_l1 = WeightedCorrection(w_x2_im, e3x1_j, cc_i, e3x1_jm, cc_jmi);
          const Real de3_r1 = WeightedCorrection(w_x2_i, e3x1_j, cc_j, e3x1_jm, cc_jm);
          face_b(b).template flux<parthenon::TopologicalType::Edge>(3, 0, k, j, i) =
              0.25 * (de3_l1 + de3_r1 + de3_l2 + de3_r2 + e3x2_im + e3x2_i + e3x1_jm +
                      e3x1_j);
        });
    return TaskStatus::complete;
  }

  // 3D: full four-correction stencil.
  const auto ib = md->GetBoundsI(parthenon::CellLevel::same, IndexDomain::interior, TE::E1);
  const auto jb = md->GetBoundsJ(parthenon::CellLevel::same, IndexDomain::interior, TE::E1);
  const auto kb = md->GetBoundsK(parthenon::CellLevel::same, IndexDomain::interior, TE::E1);

  // E1 (x1 edge). Face terms: x2 face E1 (slot 3, w slot 5), x3 face E1 (slot 6, w slot 8).
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GS07 corner E1", parthenon::DevExecSpace(), 0,
      wt.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real e1x2_k = wt(b, 3, k, j, i), e1x2_km = wt(b, 3, k - 1, j, i);
        const Real e1x3_j = wt(b, 6, k, j, i), e1x3_jm = wt(b, 6, k, j - 1, i);
        const Real w_x2_k = wt(b, 5, k, j, i), w_x2_km = wt(b, 5, k - 1, j, i);
        const Real w_x3_j = wt(b, 8, k, j, i), w_x3_jm = wt(b, 8, k, j - 1, i);
        const Real cc_k = CellCenteredEMF(0, prim, cons, b, k, j, i);
        const Real cc_km = CellCenteredEMF(0, prim, cons, b, k - 1, j, i);
        const Real cc_j = CellCenteredEMF(0, prim, cons, b, k, j - 1, i);
        const Real cc_kmj = CellCenteredEMF(0, prim, cons, b, k - 1, j - 1, i);
        const Real de1_l3 = WeightedCorrection(w_x2_km, e1x3_j, cc_km, e1x3_jm, cc_kmj);
        const Real de1_r3 = WeightedCorrection(w_x2_k, e1x3_j, cc_k, e1x3_jm, cc_j);
        const Real de1_l2 = WeightedCorrection(w_x3_jm, e1x2_k, cc_j, e1x2_km, cc_kmj);
        const Real de1_r2 = WeightedCorrection(w_x3_j, e1x2_k, cc_k, e1x2_km, cc_km);
        face_b(b).template flux<parthenon::TopologicalType::Edge>(1, 0, k, j, i) =
            0.25 * (de1_l3 + de1_r3 + de1_l2 + de1_r2 + e1x2_km + e1x2_k + e1x3_jm +
                    e1x3_j);
      });

  // E2 (x2 edge). Face terms: x1 face E2 (slot 0, w slot 2), x3 face E2 (slot 7, w slot 8).
  const auto ib2 = md->GetBoundsI(parthenon::CellLevel::same, IndexDomain::interior, TE::E2);
  const auto jb2 = md->GetBoundsJ(parthenon::CellLevel::same, IndexDomain::interior, TE::E2);
  const auto kb2 = md->GetBoundsK(parthenon::CellLevel::same, IndexDomain::interior, TE::E2);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GS07 corner E2", parthenon::DevExecSpace(), 0,
      wt.GetDim(5) - 1, kb2.s, kb2.e, jb2.s, jb2.e, ib2.s, ib2.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real e2x1_k = wt(b, 0, k, j, i), e2x1_km = wt(b, 0, k - 1, j, i);
        const Real e2x3_i = wt(b, 7, k, j, i), e2x3_im = wt(b, 7, k, j, i - 1);
        const Real w_x1_k = wt(b, 2, k, j, i), w_x1_km = wt(b, 2, k - 1, j, i);
        const Real w_x3_i = wt(b, 8, k, j, i), w_x3_im = wt(b, 8, k, j, i - 1);
        const Real cc_k = CellCenteredEMF(1, prim, cons, b, k, j, i);
        const Real cc_km = CellCenteredEMF(1, prim, cons, b, k - 1, j, i);
        const Real cc_i = CellCenteredEMF(1, prim, cons, b, k, j, i - 1);
        const Real cc_kmi = CellCenteredEMF(1, prim, cons, b, k - 1, j, i - 1);
        const Real de2_l3 = WeightedCorrection(w_x1_km, e2x3_i, cc_km, e2x3_im, cc_kmi);
        const Real de2_r3 = WeightedCorrection(w_x1_k, e2x3_i, cc_k, e2x3_im, cc_i);
        const Real de2_l1 = WeightedCorrection(w_x3_im, e2x1_k, cc_i, e2x1_km, cc_kmi);
        const Real de2_r1 = WeightedCorrection(w_x3_i, e2x1_k, cc_k, e2x1_km, cc_km);
        face_b(b).template flux<parthenon::TopologicalType::Edge>(2, 0, k, j, i) =
            0.25 * (de2_l3 + de2_r3 + de2_l1 + de2_r1 + e2x3_im + e2x3_i + e2x1_km +
                    e2x1_k);
      });

  // E3 (x3 edge). Face terms: x1 face E3 (slot 1, w slot 2), x2 face E3 (slot 4, w slot 5).
  const auto ib3 = md->GetBoundsI(parthenon::CellLevel::same, IndexDomain::interior, TE::E3);
  const auto jb3 = md->GetBoundsJ(parthenon::CellLevel::same, IndexDomain::interior, TE::E3);
  const auto kb3 = md->GetBoundsK(parthenon::CellLevel::same, IndexDomain::interior, TE::E3);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GS07 corner E3", parthenon::DevExecSpace(), 0,
      wt.GetDim(5) - 1, kb3.s, kb3.e, jb3.s, jb3.e, ib3.s, ib3.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real e3x1_j = wt(b, 1, k, j, i), e3x1_jm = wt(b, 1, k, j - 1, i);
        const Real e3x2_i = wt(b, 4, k, j, i), e3x2_im = wt(b, 4, k, j, i - 1);
        const Real w_x1_j = wt(b, 2, k, j, i), w_x1_jm = wt(b, 2, k, j - 1, i);
        const Real w_x2_i = wt(b, 5, k, j, i), w_x2_im = wt(b, 5, k, j, i - 1);
        const Real cc_j = CellCenteredEMF(2, prim, cons, b, k, j, i);
        const Real cc_jm = CellCenteredEMF(2, prim, cons, b, k, j - 1, i);
        const Real cc_i = CellCenteredEMF(2, prim, cons, b, k, j, i - 1);
        const Real cc_jmi = CellCenteredEMF(2, prim, cons, b, k, j - 1, i - 1);
        const Real de3_l2 = WeightedCorrection(w_x1_jm, e3x2_i, cc_jm, e3x2_im, cc_jmi);
        const Real de3_r2 = WeightedCorrection(w_x1_j, e3x2_i, cc_j, e3x2_im, cc_i);
        const Real de3_l1 = WeightedCorrection(w_x2_im, e3x1_j, cc_i, e3x1_jm, cc_jmi);
        const Real de3_r1 = WeightedCorrection(w_x2_i, e3x1_j, cc_j, e3x1_jm, cc_jm);
        face_b(b).template flux<parthenon::TopologicalType::Edge>(3, 0, k, j, i) =
            0.25 * (de3_l1 + de3_r1 + de3_l2 + de3_r2 + e3x2_im + e3x2_i + e3x1_jm +
                    e3x1_j);
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

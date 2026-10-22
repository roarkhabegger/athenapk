#include "../eos/adiabatic_ctmhd.hpp"

#include "config.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"

using parthenon::IndexDomain;

void AdiabaticCTMHDEOS::ConservedToPrimitive(MeshData<Real> *md) const {
  const auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  const auto jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  const auto kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto nhydro = pkg->Param<int>("nhydro");
  const auto nscalars = pkg->Param<int>("nscalars");
  const auto eos = (*this);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConservedToPrimitiveCTMHD", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        eos.ConsToPrim(cons_pack(b), prim_pack(b), nhydro, nscalars, k, j, i);
      });
}

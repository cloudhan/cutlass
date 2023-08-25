#include "thrust/device_vector.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
using namespace cute;
using namespace cutlass;
using namespace cutlass::gemm;

template <typename Element, typename Layout, int ThreadCount, int ShapeM, int ShapeK>
struct DefaultGemm_Simt_OperandA;

///////////////////////////////////////////////////////////////////////////////

template <typename Element>
struct DefaultGemm_Simt_OperandA<Element, layout::ColumnMajor, 256, 128, 8>
{
  using SmemLayoutAtom = Layout<Shape <_128,  _8>,
                                Stride<  _1,_128>>;

  using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<Element>, Element>{},
                    Layout<Shape <_32, _8>,
                           Stride< _1,_32>>{},
                    Layout<Shape<_1,_1>>{}));
};

template <typename Element>
struct DefaultGemm_Simt_OperandA<Element, layout::RowMajor, 256, 128, 8>
{
  using SmemLayoutAtom = Layout<Shape <_128,          _8>,
                                Stride<  _1,Int<128 + 4>>>;   // Padded

  using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<Element>, Element>{},
                    Layout<Shape <_32, _8>,
                           Stride< _8, _1>>{},
                    Layout<Shape<_1,_1>>{}));

};

template <typename Element, typename Layout, int ThreadCount, int ShapeN, int ShapeK>
struct DefaultGemm_Simt_OperandB;

template <typename Element, int ThreadCount, int ShapeN, int ShapeK>
struct DefaultGemm_Simt_OperandB<Element, layout::ColumnMajor, ThreadCount, ShapeN, ShapeK>
     : DefaultGemm_Simt_OperandA<Element, layout::RowMajor,    ThreadCount, ShapeN, ShapeK> {};

template <typename Element, int ThreadCount, int ShapeN, int ShapeK>
struct DefaultGemm_Simt_OperandB<Element, layout::RowMajor,    ThreadCount, ShapeN, ShapeK>
     : DefaultGemm_Simt_OperandA<Element, layout::ColumnMajor, ThreadCount, ShapeN, ShapeK> {};

void cutlass_sm70_gemm(
  const thrust::device_vector<float>& dev_a,
  const thrust::device_vector<float>& dev_b,
  thrust::device_vector<float>& dev_c
) {
  using ElementAccumulator = float;
  using ElementA = float;
  using ElementB = float;
  using ElementC = float;

  using TileShape = Shape<_128, _128, _8>;
  constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopSm70TwoStage;
  using TiledMma = TiledMMA<
      MMA_Atom<UniversalFMA<ElementAccumulator, ElementA, ElementB, ElementC>>,
      Layout<Shape<_16, _16, _1>>>;

  // A
  using LayoutA = layout::ColumnMajor;
  constexpr int kAlignmentA = 1;
  using DefaultOperandA = DefaultGemm_Simt_OperandA<ElementA, LayoutA, ThreadCount, 128, 8>;
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
  using SmemCopyAtomA   = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA  = typename DefaultOperandA::GmemTiledCopy;

  // B
  using LayoutB = layout::RowMajor;
  constexpr int kAlignmentB = 1;
  using DefaultOperandB = DefaultGemm_Simt_OperandB<ElementB, LayoutB, ThreadCount, 128, 8>;
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
  using SmemCopyAtomB   = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB  = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<LayoutA>,
    ElementB, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  using LayoutC = layout::RowMajor;
  using ElementC = float;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::EpilogueDefault>;



  using Gemm = gemm::kernel::GemmUniversal<cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

  using DeviceGemm = gemm::device::GemmUniversalAdapter<Gemm>;
  DeviceGemm cutlass_sgemm{};

  using StrideA = typename DeviceGemm::GemmKernel::StrideA;
  using StrideB = typename DeviceGemm::GemmKernel::StrideB;
  using StrideC = typename DeviceGemm::GemmKernel::StrideC;
  using StrideD = typename DeviceGemm::GemmKernel::StrideD;

  StrideA stride_a = make_cute_packed_stride(StrideA{}, cute::make_shape(4096, 4096, Int<1>{}));
  StrideB stride_b = make_cute_packed_stride(StrideB{}, cute::make_shape(4096, 4096, Int<1>{}));
  StrideC stride_c = make_cute_packed_stride(StrideC{}, cute::make_shape(4096, 4096, Int<1>{}));


  cutlass_sgemm({
    cutlass::gemm::GemmUniversalMode::kGemm,
    make_shape(4096, 4096, 4096),
    { dev_a.data().get(), stride_a, dev_b.data().get(), stride_b },
    { {1.0f, 0.0f}, dev_c.data().get(), stride_c, dev_c.data().get(), stride_c }

  });
}

int main() {
  auto dev_a = thrust::device_vector<float>(4096 * 4096);
  auto dev_b = thrust::device_vector<float>(4096 * 4096);
  auto dev_c = thrust::device_vector<float>(4096 * 4096);

  cutlass_sm70_gemm(dev_a, dev_b, dev_c);
}

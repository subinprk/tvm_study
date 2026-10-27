from tvm.script import tir as T

@T.prim_func
def matmul(
    A: T.Buffer((4, 4), "float32"),
    B: T.Buffer((4, 4), "float32"),
    C: T.Buffer((4, 4), "float32"),
):
    for i, j, k in T.grid(4, 4, 4):
    # for: 언제 반복하는가?
        with T.block("matmul"):
        # block: 무엇을 계산하는가?
        # 블록의 계산: C[i, j] += A[i, k]B[k, j] (Matmul 이니까)
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            # Block Iterators: 반복 축은 어떤 역할인가?
            # Spatial-S-i , Spatial-S-j, Reduction-R-k
            with T.init():
                C[vi, vj] = T.float32(0)

            C[vi, vj] = (
                C[vi, vj]
                + A[vi, vk] * B[vk, vj]
            )
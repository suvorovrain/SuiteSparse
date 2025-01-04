{
    const int64_t m = C->vlen;
    const int64_t *restrict Bp = B->p;
    const int64_t *restrict Bh = B->h;
    const int64_t *restrict Bi = B->i;
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *)A->x;
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *)B->x;
    size_t vl = __riscv_vsetvl_e64m8(m);
    GB_C_TYPE *restrict Cx = (GB_C_TYPE *)C->x;

#pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
    for (int tid = 0; tid < ntasks; tid++)
    {
        const int64_t jB_start = B_slice[tid];
        const int64_t jB_end = B_slice[tid + 1];

        for (int64_t jB = jB_start; jB < jB_end; jB++)
        {
            const int64_t j = GBH_B(Bh, jB);
            GB_C_TYPE *restrict Cxj = Cx + (j * m);
            const int64_t pB_start = Bp[jB];
            const int64_t pB_end = Bp[jB + 1];
            for (int64_t i = 0; i < m && (m - i) >= vl; i += vl)
            {
                vfloat64m8_t vc = __riscv_vle64_v_f64m8(Cxj + i, vl);

                for (int64_t pB = pB_start; pB < pB_end; pB++)
                {
                    const int64_t k = Bi[pB];
                    const GB_B_TYPE bkj = Bx[pB];
                    vfloat64m8_t va = __riscv_vle64_v_f64m8(Ax + i + k * m, vl);
                    vc = __riscv_vfmacc_vf_f64m8(vc, bkj, va, vl);
                }

                __riscv_vse64_v_f64m8(Cxj + i, vc, vl);
            }
            int64_t remaining = m % vl;
            if (remaining > 0)
            {
                int64_t i = m - remaining;
                vfloat64m8_t vc = __riscv_vle64_v_f64m8(Cxj + i, remaining);

                for (int64_t pB = pB_start; pB < pB_end; pB++)
                {
                    const int64_t k = Bi[pB];
                    const GB_B_TYPE bkj = Bx[pB];
                    vfloat64m8_t va = __riscv_vle64_v_f64m8(Ax + i + k * m, remaining);
                    vc = __riscv_vfmacc_vf_f64m8(vc, bkj, va, remaining);
                }

                __riscv_vse64_v_f64m8(Cxj + i, vc, remaining);
            }
        }
    }
}

import os
import json
import csv
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# === 输入路径 ===
json_file_path = '/nfs/home/xutongqiao/vector/xs-env/NEMU/cluster-0-0.json'
csv_dir = '/nfs/home/xutongqiao/vector/xs-env/NEMU_vector_explore/test04271037'

# === 类别映射表 ===
category_map = {
    'vmvxs': 'cross_domain_move',
    'vpopc': 'config', # 1的数量
    'vfirst': 'config', # 第1个为1的元素
    'vmvsx': 'cross_domain_move',
    'vzextvf8': 'int_extension',
    'vsextvf8': 'int_extension',
    'vzextvf4': 'int_extension',
    'vsextvf4': 'int_extension',
    'vzextvf2': 'int_extension',
    'vsextvf2': 'int_extension',
    'vbrev_v': 'logical',
    'vbrev8_v': 'logical',
    'vrev8_v': 'logical',
    'vclz_v': 'config',  # 前导0个数
    'vcpop_v': 'config', # 人口计数（1的数量）
    'vctz_v': 'config', # 尾部0个数
    'vmsbf': 'mask',
    'vmsof': 'mask',
    'vmsif': 'mask',
    'viota': 'mask',
    'vid': 'mask',
    'vfmvfs': 'cross_domain_move',
    'vfmvsf': 'cross_domain_move',
    'vfcvt_xufv': 'fp_convert',
    'vfcvt_xfv': 'fp_convert',
    'vfcvt_fxuv': 'fp_convert',
    'vfcvt_fxv': 'fp_convert',
    'vfcvt_rtz_xufv': 'fp_convert',
    'vfcvt_rtz_xfv': 'fp_convert',
    'vfwcvt_xufv': 'fp_convert',
    'vfwcvt_xfv': 'fp_convert',
    'vfwcvt_fxuv': 'fp_convert',
    'vfwcvt_fxv': 'fp_convert',
    'vfwcvt_ffv': 'fp_convert',
    'vfwcvt_rtz_xufv': 'fp_convert',
    'vfwcvt_rtz_xfv': 'fp_convert',
    'vfncvt_xufw': 'fp_convert',
    'vfncvt_xfw': 'fp_convert',
    'vfncvt_fxuw': 'fp_convert',
    'vfncvt_fxw': 'fp_convert',
    'vfncvt_ffw': 'fp_convert',
    'vfncvt_rod_ffw': 'fp_convert',
    'vfncvt_rtz_xufw': 'fp_convert',
    'vfncvt_rtz_xfw': 'fp_convert',
    'vfsqrt_v': 'fp_sqrt',
    'vfrsqrt7_v': 'fp_sqrt',
    'vfrec7_v': 'fp_div',
    'vfclass_v': 'classify',
    'vadd': 'int_arith',
    'vsub': 'int_arith',
    'vrsub': 'int_arith',
    'vminu': 'int_arith',
    'vmin': 'int_arith',
    'vmaxu': 'int_arith',
    'vmax': 'int_arith',
    'vand': 'logical',
    'vor': 'logical',
    'vxor': 'logical',
    'vrgather': 'gather',
    'vrgatherei16': 'gather',
    'vadc': 'int_arith',
    'vmadc': 'int_arith',
    'vsbc': 'int_arith',
    'vmsbc': 'int_arith',
    'vmerge': 'merge',
    'vmseq': 'int_arith',
    'vmsne': 'int_arith',
    'vmsltu': 'int_arith',
    'vmslt': 'int_arith',
    'vmsleu': 'int_arith',
    'vmsle': 'int_arith',
    'vmsgtu': 'int_arith',
    'vmsgt': 'int_arith',
    'vsaddu': 'int_arith',
    'vsadd': 'int_arith',
    'vssubu': 'int_arith',
    'vssub': 'int_arith',
    'vsll': 'shift',
    'vmvnr': 'cross_domain_move',
    'vsmul': 'int_mul',
    'vsrl': 'shift',
    'vsra': 'shift',
    'vssra': 'shift',
    'vnsrl': 'shift',
    'vnsra': 'shift',
    'vnclipu': 'narrow',
    'vnclip': 'narrow',
    'vssrl': 'shift',
    'vwredsumu': 'reduction',
    'vwredsum': 'reduction',
    'vdotu': 'dot',
    'vdot': 'dot',
    'vwsmaccu': 'int_madd',
    'vwsmacc': 'int_madd',
    'vwsmaccsu': 'int_madd',
    'vwsmaccus': 'int_madd',
    'vandn': 'logical',
    'vrol': 'shift',
    'vror': 'shift',
    'vwsll': 'shift',
    'vslideup': 'slide',
    'vslidedown': 'slide',
    'vredsum': 'reduction',
    'vredand': 'reduction',
    'vredor': 'reduction',
    'vredxor': 'reduction',
    'vredminu': 'reduction',
    'vredmin': 'reduction',
    'vredmaxu': 'reduction',
    'vredmax': 'reduction',
    'vaaddu': 'int_arith',
    'vaadd': 'int_arith',
    'vasubu': 'int_arith',
    'vasub': 'int_arith',
    'vcompress': 'compress',
    'vmandnot': 'mask',
    'vmand': 'mask',
    'vmor': 'mask',
    'vmxor': 'mask',
    'vmornot': 'mask',
    'vmnand': 'mask',
    'vmnor': 'mask',
    'vmxnor': 'mask',
    'vdivu': 'int_div',
    'vdiv': 'int_div',
    'vremu': 'int_div',
    'vrem': 'int_div',
    'vmulhu': 'int_mul',
    'vmul': 'int_mul',
    'vmulhsu': 'int_mul',
    'vmulh': 'int_mul',
    'vmadd': 'int_madd',
    'vnmsub': 'int_madd',
    'vmacc': 'int_madd',
    'vnmsac': 'int_madd',
    'vwaddu': 'int_arith',
    'vwadd': 'int_arith',
    'vwsubu': 'int_arith',
    'vwsub': 'int_arith',
    'vwaddu_w': 'int_arith',
    'vwadd_w': 'int_arith',
    'vwsubu_w': 'int_arith',
    'vwsub_w': 'int_arith',
    'vwmulu': 'int_mul',
    'vwmulsu': 'int_mul',
    'vwmul': 'int_mul',
    'vwmaccu': 'int_madd',
    'vwmacc': 'int_madd',
    'vwmaccus': 'int_madd',
    'vwmaccsu': 'int_madd',
    'vslide1up': 'slide1',
    'vslide1down': 'slide1',
    'vfadd': 'fp_arith',
    'vfredusum': 'reduction',
    'vfsub': 'fp_arith',
    'vfredosum': 'reduction',
    'vfmin': 'fp_arith',
    'vfredmin': 'reduction',
    'vfmax': 'fp_arith',
    'vfredmax': 'reduction',
    'vfsgnj': 'sign_injection',
    'vfsgnjn': 'sign_injection',
    'vfsgnjx': 'sign_injection',
    'vmfeq': 'fp_arith',
    'vmfle': 'fp_arith',
    'vmflt': 'fp_arith',
    'vmfne': 'fp_arith',
    'vfdiv': 'fp_div',
    'vfmul': 'fp_mul',
    'vfmadd': 'fp_madd',
    'vfnmadd': 'fp_madd',
    'vfmsub': 'fp_madd',
    'vfnmsub': 'fp_madd',
    'vfmacc': 'fp_madd',
    'vfnmacc': 'fp_madd',
    'vfmsac': 'fp_madd',
    'vfnmsac': 'fp_madd',
    'vfwadd': 'fp_arith',
    'vfwredusum': 'reduction',
    'vfwsub': 'fp_arith',
    'vfwredosum': 'reduction',
    'vfwadd_w': 'fp_arith',
    'vfwsub_w': 'fp_arith',
    'vfwmul': 'fp_mul',
    'vfwmacc': 'fp_madd',
    'vfwnmacc': 'fp_madd',
    'vfwmsac': 'fp_madd',
    'vfwnmsac': 'fp_madd',
    'vfslide1up': 'slide1',
    'vfslide1down': 'slide1',
    'vfmerge': 'merge',
    'vmfgt': 'fp_arith',
    'vmfge': 'fp_arith',
    'vfrdiv': 'fp_div',
    'vfrsub': 'fp_arith',
    'vsetvli': 'vset',
    'vsetivli': 'vset',
    'vsetvl': 'vset',
    'vwxunary0_dispatch': 'category',
    'vmunary0_dispatch': 'category',
    'vrxunary0_dispatch': 'category',
    'vwfunary0_dispatch': 'category',
    'vrfunary0_dispatch': 'category',
    'vopivv': 'category',
    'vopfvv': 'category',
    'vopmvv': 'category',
    'vopivi': 'category',
    'vopivx': 'category',
    'vopfvf': 'category',
    'vopmvx': 'category',
    'vsetvl_dispatch': 'category', 

    'vle': 'vload-unit-stride',   
    'vleff': 'vload-unit-stride',
    'vlr': 'vload-whole',
    'vlm': 'vload-mask',
    'vlxe': 'vload-index',
    'vlse': 'vload-strided',
    'vse': 'vstore-unit-stride',
    'vsr': 'vstore-whole',
    'vsm': 'vstore-mask',
    'vsxe': 'vstore-index',
    'vsse': 'vstore-strided',
    
    'vle_mmu': 'vload-unit-stride',
    'vleff_mmu': 'vload-unit-stride',
    'vlr_mmu': 'vload-whole',
    'vlm_mmu': 'vload-mask',
    'vlxe_mmu': 'vload-index',
    'vlse_mmu': 'vload-strided',
    'vse_mmu': 'vstore-unit-stride',
    'vsr_mmu': 'vstore-whole',
    'vsm_mmu': 'vstore-mask',
    'vsxe_mmu': 'vstore-index',
    'vsse_mmu': 'vstore-strided',
}

# === 按指令分类的汇总函数 ===
def summarize_by_category(instr_counts, category_map, category_lists):
    category_totals = []
    for i in range(len(category_lists)):
        category_totals.append(np.zeros((8, 8, 9), dtype=float))
    for (instr, vsew, vlmul, segment), count in instr_counts.items():
        # category = category_map.get(instr, 'other')
        category_totals[category_lists.index(instr)][int(vsew)][int(vlmul)][int(segment)+1] += count
        
    return category_totals

category_lists = []
for key, value in category_map.items():
    if key not in category_lists:
        category_lists.append(key)

# === 主处理 ===
coverage_list = defaultdict(float)
total_list = []
with open(json_file_path, 'r') as f:
    json_data = json.load(f)
benchmark_lists = []
for benchmark_name, data in json_data.items():
    benchmark_lists.append(benchmark_name)
    coverage = 0
    if "points" not in data:
        continue
    point_ids = data["points"].keys()

    instr_counts = defaultdict(float)

    for point_id in point_ids:
        expected_filename = f'count_{benchmark_name}_{point_id}.csv'
        full_path = os.path.join(csv_dir, expected_filename)
        if os.path.exists(full_path):
            current_coverage = float(data["points"][point_id])
            coverage += current_coverage
            with open(full_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) >= 2:
                        instr = row[0].strip()
                        # category_key = (row[1].strip(), row[2].strip(), row[3].strip())  # 使用 row[1], row[2], row[3] 作为新的分类标准
                        try:
                            count = int(row[4].strip())
                            instr_counts[(instr, row[1].strip(), row[2].strip(), row[3].strip())] += count * current_coverage
                        except ValueError:
                            continue
    
    coverage_list[benchmark_name] = coverage
    # 分类统计
    category_totals = summarize_by_category(instr_counts, category_map, category_lists)
    total_list.append(category_totals)

a=np.array(total_list)
a = a.reshape(55, -1)
inst_2d_list = []
inst_sew_list = []
inst_sew_value = [8,16,32,64,-1,-2,-3,-4]
inst_lmul_list = []
inst_lmul_value = [1,2,4,8,-1,1/8,1/4,1/2]
inst_seg_list = []
inst_seg_value = [-1,0,1,2,3,4,5,6,7,8]
for cate in category_lists:
    for i in range(8):
        for j in range(8):
            for k in range(9):
                inst_2d_list.append(f"{cate}")
                inst_sew_list.append(inst_sew_value[i])
                inst_lmul_list.append(inst_lmul_value[j])
                inst_seg_list.append(inst_seg_value[k])
index = pd.MultiIndex.from_product(
    [benchmark_lists, inst_2d_list],
    names=["benchmark", "inst_cate"]
)

df = pd.DataFrame(a, columns=[inst_2d_list, inst_sew_list, inst_lmul_list, inst_seg_list], index=benchmark_lists)
columns = pd.MultiIndex.from_arrays([inst_2d_list, inst_sew_list, inst_lmul_list, inst_seg_list], names=('cate', 'sew', 'lmul', 'seg'))

df.columns = columns
df = df.loc[:, ~(df == 0).all(axis=0)]
df_sum = df.sum(level='cate', axis=1)
print(df)
df = df.transpose()
df_sum = df_sum.transpose()
df.to_excel("rvv_instruction_ratios.xlsx", index=True)
df_sum.to_excel("rvv_instruction_ratios_sum.xlsx", index=True)
print(coverage_list)
# plt.figure(figsize=(12, 8))  # 可调整图形大小
# sns.heatmap(df_sum, annot=True, cmap='coolwarm', cbar=True)
# plt.title('Heatmap of DataFrame')
# plt.show()
# plt.savefig("./rvv_instruction_ratios.png", bbox_inches='tight')
# plt.close()

# # === 输出表格 ===
# all_categories = sorted(benchmark_category_table.keys())
# all_benchmarks = sorted(json_data.keys())
# print("Category," + ",".join(all_benchmarks)+",")
# print("Coverage," + ",".join(f"{coverage_list[b]:.2f}" for b in all_benchmarks) + ",")
# for category in all_categories:
#     if category=="category":
#         continue
#     row = [category]
#     for benchmark in all_benchmarks:
#         row.append(f"{benchmark_category_table[category].get(benchmark, 0):.2f}")
#     print(",".join(row)+",")

# # === 添加总计行 ===
# total_per_benchmark = []
# for benchmark in all_benchmarks:
#     total = sum(benchmark_category_table[cat].get(benchmark, 0) for cat in all_categories if cat != "category")
#     total_per_benchmark.append(f"{total:.2f}")
# print("Total," + ",".join(total_per_benchmark) + ",")
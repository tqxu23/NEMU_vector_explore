import os
import json
import csv
from collections import defaultdict

# === 输入路径 ===
json_file_path = '/nfs/home/xutongqiao/vector/xs-env/NEMU/cluster-0-0.json'
csv_dir = '/nfs/home/xutongqiao/vector/xs-env/NEMU/vec_count_results'

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
    'vslide1up': 'slide',
    'vslide1down': 'slide',
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
    'vfslide1up': 'slide',
    'vfslide1down': 'slide',
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

    'vle': 'vload',   
    'vleff': 'vload',
    'vlr': 'vload',
    'vlr': 'vload',
    'vlr': 'vload',
    'vlr': 'vload',
    'vlm': 'vload',
    'vlxe': 'vload',
    'vlse': 'vload',
    'vlxe': 'vload',
    'vse': 'vstore',
    'vsr': 'vstore',
    'vsm': 'vstore',
    'vsxe': 'vstore',
    'vsse': 'vstore',
    'vle_mmu': 'vload',
    'vleff_mmu': 'vload',
    'vle_mmu': 'vload',
    'vlr_mmu': 'vload',
    'vlm_mmu': 'vload',
    'vlxe_mmu': 'vload',
    'vlse_mmu': 'vload',
    'vse_mmu': 'vstore',
    'vsr_mmu': 'vstore',
    'vsm_mmu': 'vstore',
    'vsxe_mmu': 'vstore',
    'vsse_mmu': 'vstore',
}

# === 按指令分类的汇总函数 ===
def summarize_by_category(instr_counts, category_map):
    category_totals = {'cross_domain_move': 0,
                    'config': 0,
                    'int_extension': 0,
                    'logical': 0,
                    'config': 0,
                    'mask': 0,
                    'fp_convert': 0,
                    'fp_sqrt': 0,
                    'fp_div': 0,
                    'classify': 0,
                    'int_arith': 0,
                    'gather': 0,
                    'merge': 0,
                    'int_mul': 0,
                    'int_madd': 0,
                    'compress': 0,
                    'shift': 0,
                    'narrow': 0,
                    'reduction': 0,
                    'dot': 0,
                    'slide': 0,
                    'fp_arith': 0,
                    'fp_mul': 0,
                    'fp_madd': 0,
                    'sign_injection': 0,
                    'int_sqrt': 0,
                    'int_div': 0,
                    'vset': 0,
                    'category': 0,
                    'vload': 0,
                    'vstore': 0,
                    }

    for instr, count in instr_counts.items():
        category = category_map.get(instr, 'other')
        category_totals[category] += count

    return category_totals

# === 主处理 ===
benchmark_category_table = defaultdict(lambda: defaultdict(float))
coverage_list = defaultdict(float)

with open(json_file_path, 'r') as f:
    json_data = json.load(f)

for benchmark_name, data in json_data.items():
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
                        try:
                            count = int(row[1].strip())
                            instr_counts[instr] += count * current_coverage
                        except ValueError:
                            continue
    
    coverage_list[benchmark_name] = coverage
    # 分类统计
    category_totals = summarize_by_category(instr_counts, category_map)
    for category, value in category_totals.items():
        benchmark_category_table[category][benchmark_name] = value

# === 输出表格 ===
all_categories = sorted(benchmark_category_table.keys())
all_benchmarks = sorted(json_data.keys())
print("Category," + ",".join(all_benchmarks)+",")
print("Coverage," + ",".join(f"{coverage_list[b]:.2f}" for b in all_benchmarks) + ",")
for category in all_categories:
    if category=="category":
        continue
    row = [category]
    for benchmark in all_benchmarks:
        row.append(f"{benchmark_category_table[category].get(benchmark, 0):.2f}")
    print(",".join(row)+",")

# === 添加总计行 ===
total_per_benchmark = []
for benchmark in all_benchmarks:
    total = sum(benchmark_category_table[cat].get(benchmark, 0) for cat in all_categories if cat != "category")
    total_per_benchmark.append(f"{total:.2f}")
print("Total," + ",".join(total_per_benchmark) + ",")
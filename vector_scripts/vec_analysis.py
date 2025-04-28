import os
import subprocess
import shutil
from datetime import datetime

# 常量路径
CHECKPOINT_DIR = "/nfs/home/jiaxiaoyu/checkpoint/spec06_gcc15.x.0_rv64gcbv_base_intFppOff_2025_0307_elf_NEMU_archgroup_2025-03-28-14-26/checkpoint-0-0-0"
NEMU_PATH = "/nfs/home/xutongqiao/vector/xs-env/NEMU_vector_explore/build/riscv64-nemu-interpreter"
COUNT_CSV = "/nfs/home/xutongqiao/vector/xs-env/NEMU/count.csv"
DEST_DIR = "/nfs/home/xutongqiao/vector/xs-env/NEMU_vector_explore/test04271037"
TIMEOUT_LOG = "/nfs/home/xutongqiao/vector/xs-env/NEMU_vector_explore/test04271037/timeout.log"
TIMEOUT_SECONDS = 60

def log_timeout(task_name):
    with open(TIMEOUT_LOG, "a") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] TIMEOUT: {task_name}\n")

# 创建输出目录（如果不存在）
os.makedirs(DEST_DIR, exist_ok=True)

# 遍历 .zstd 文件
for root, _, files in os.walk(CHECKPOINT_DIR):
    for file in files:
        if file.endswith(".zstd"):
            zstd_path = os.path.join(root, file)

            # 提取 benchmark 名和子路径，如 bwaves/10068
            rel_path = os.path.relpath(zstd_path, CHECKPOINT_DIR)
            parts = rel_path.split(os.sep)
            if len(parts) < 3:
                print(f"跳过路径（结构不足）: {zstd_path}")
                continue
            benchmark = parts[0]
            subdir = parts[1]
            task_name = f"{benchmark}/{subdir}"

            output_filename = f"count_{benchmark}_{subdir}.csv"
            dest_file = os.path.join(DEST_DIR, output_filename)
            if os.path.exists(dest_file):
                print(f"已存在 {output_filename}，跳过 {benchmark}/{subdir}")
                continue


            # 构造命令
            cmd = [
                NEMU_PATH,
                zstd_path,
                "--max-instr=40000000",
                "-b"
            ]
            print(f"运行：{' '.join(cmd)}")

            try:
                subprocess.run(cmd, timeout=TIMEOUT_SECONDS, check=True)
            except subprocess.TimeoutExpired:
                print(f"任务超时，跳过：{task_name}")
                log_timeout(task_name)
                continue
            except subprocess.CalledProcessError as e:
                print(f"命令执行失败: {e}")
                continue

            # 拷贝 count.csv 并重命名
            dest_file = os.path.join(DEST_DIR, f"count_{benchmark}_{subdir}.csv")
            try:
                shutil.copy(COUNT_CSV, dest_file)
                print(f"结果已保存至: {dest_file}")
            except FileNotFoundError:
                print("错误：count.csv 未找到")
            except Exception as e:
                print(f"拷贝失败: {e}")


            try:
                os.remove(COUNT_CSV)
                print(f"文件 '{COUNT_CSV}' 已删除。")
            except FileNotFoundError:
                print(f"文件 '{COUNT_CSV}' 不存在。")
            except PermissionError:
                print(f"没有权限删除文件 '{COUNT_CSV}'。")
            except Exception as e:
                print(f"删除文件时发生错误: {e}")
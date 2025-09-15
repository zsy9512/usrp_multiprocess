import subprocess
import time
import argparse
import os
import signal
import sys
import multiprocessing
import threading
import socket

class USRPMultiprocessCoordinator:
    """USRP多进程协调器：管理发射、接收、处理三个程序"""

    def __init__(self, args):
        self.args = args
        self.processes = {}
        self.ipc_file = args.ipc_file
        # 使用Queue进行进程间通信
        self.data_queue = multiprocessing.Queue(maxsize=1000)
        self.use_udp = args.use_udp.lower() == 'true'

        # UDP相关（如果使用UDP模式）
        if self.use_udp:
            self.udp_port = args.udp_port or self._find_free_port()
            self.udp_host = '127.0.0.1'
        else:
            self.udp_port = None
            self.udp_host = None

    def _find_free_port(self):
        """查找一个空闲的UDP端口"""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def start_processes(self):
        """启动所有进程"""
        print("启动USRP多进程系统...")

        # 获取当前目录
        current_dir = os.getcwd()

        try:
            # 分配UDP端口用于进程间通信
            print(f"UDP通信端口: {self.udp_host}:{self.udp_port}")

            # 启动发射程序
            print("启动发射程序...")
            tx_cmd = [
                sys.executable, os.path.join(current_dir, 'tx_program.py'),
                '--tx_freq', str(self.args.tx_freq),
                '--rate', str(self.args.rate),
                '--tx_gain', str(self.args.tx_gain),
                '--repeat_count', str(self.args.repeat_count),
                '--bit_generator', self.args.bit_generator
            ]
            if self.args.args:
                tx_cmd.extend(['--args', self.args.args])

            self.processes['tx'] = subprocess.Popen(tx_cmd)
            print(f"发射程序PID: {self.processes['tx'].pid}")

            # 等待一下让发射程序初始化
            time.sleep(2)

            # 启动接收程序
            print("启动接收程序...")
            rx_cmd = [
                sys.executable, os.path.join(current_dir, 'rx_program.py'),
                '--rx_freq', str(self.args.rx_freq),
                '--rate', str(self.args.rate),
                '--rx_gain', str(self.args.rx_gain),
                '--buffer_size', str(self.args.buffer_size),
                '--udp_host', self.udp_host,
                '--udp_port', str(self.udp_port)
            ]
            if self.args.args:
                rx_cmd.extend(['--args', self.args.args])

            self.processes['rx'] = subprocess.Popen(rx_cmd)
            print(f"接收程序PID: {self.processes['rx'].pid}")

            # 等待一下让接收程序初始化
            time.sleep(2)

            # 启动处理程序
            print("启动处理程序...")
            proc_cmd = [
                sys.executable, os.path.join(current_dir, 'processing_program.py'),
                '--rate', str(self.args.rate),
                '--output_file', self.args.output_file,
                '--udp_host', self.udp_host,
                '--udp_port', str(self.udp_port)
            ]

            self.processes['proc'] = subprocess.Popen(proc_cmd)
            print(f"处理程序PID: {self.processes['proc'].pid}")

            print("所有进程已启动")
            print(f"数据流: 发射程序 → USRP发送 | USRP接收 → 接收程序 → UDP({self.udp_host}:{self.udp_port}) → 处理程序")
            print("按Ctrl+C停止所有进程...")

        except Exception as e:
            print(f"启动进程失败: {str(e)}")
            self.stop_all_processes()
            raise

    def monitor_processes(self):
        """监控进程状态"""
        while True:
            try:
                time.sleep(5)  # 每5秒检查一次

                # 检查进程是否还在运行
                for name, process in self.processes.items():
                    if process.poll() is not None:
                        print(f"进程 {name} 已退出，退出码: {process.returncode}")

                        # 如果关键进程退出，停止所有进程
                        if name in ['tx', 'rx']:
                            print(f"关键进程 {name} 退出，停止所有进程")
                            self.stop_all_processes()
                            return

                # 打印系统状态
                print("系统状态: 所有进程正常运行")

            except KeyboardInterrupt:
                print("\n收到停止信号...")
                break
            except Exception as e:
                print(f"监控错误: {str(e)}")

    def stop_all_processes(self):
        """停止所有进程"""
        print("停止所有进程...")

        for name, process in self.processes.items():
            try:
                if process.poll() is None:  # 进程还在运行
                    print(f"终止进程 {name} (PID: {process.pid})")
                    process.terminate()

                    # 等待进程结束
                    try:
                        process.wait(timeout=5)
                        print(f"进程 {name} 已正常终止")
                    except subprocess.TimeoutExpired:
                        print(f"进程 {name} 没有响应，强制终止")
                        process.kill()
                        process.wait()

            except Exception as e:
                print(f"停止进程 {name} 时出错: {str(e)}")

        # 清理IPC文件（向后兼容）
        try:
            if os.path.exists(self.ipc_file):
                os.remove(self.ipc_file)
                print(f"IPC文件 {self.ipc_file} 已清理")
        except Exception as e:
            print(f"清理IPC文件错误: {str(e)}")

        print("所有进程已停止")

    def run(self):
        """运行多进程系统"""
        try:
            self.start_processes()
            self.monitor_processes()

        except Exception as e:
            print(f"运行时错误: {str(e)}")
        finally:
            self.stop_all_processes()

def main():
    parser = argparse.ArgumentParser(description="USRP DQPSK多进程协调器")

    # 发射参数
    parser.add_argument("--tx_freq", type=float, default=900e6, help="发射频率 (Hz)")
    parser.add_argument("--tx_gain", type=float, default=50, help="发射增益 (dB)")
    parser.add_argument("--repeat_count", type=int, default=10, help="每个帧重复发送次数")
    parser.add_argument("--bit_generator", type=str, default="random", choices=["random", "zeros", "ones"], help="比特生成模式")

    # 接收参数
    parser.add_argument("--rx_freq", type=float, default=900e6, help="接收频率 (Hz)")
    parser.add_argument("--rx_gain", type=float, default=50, help="接收增益 (dB)")
    parser.add_argument("--buffer_size", type=int, default=1000, help="接收缓冲区大小")

    # 通用参数
    parser.add_argument("--rate", type=float, default=1e6, help="采样率 (Hz)")
    parser.add_argument("--args", type=str, default="", help="USRP设备参数")

    # 处理参数
    parser.add_argument("--output_file", type=str, default="demodulated_bits.txt", help="解调比特输出文件")

    # UDP通信参数
    parser.add_argument("--udp_port", type=int, default=None, help="UDP通信端口（自动分配如果未指定）")
    parser.add_argument("--udp_host", type=str, default="127.0.0.1", help="UDP通信主机地址")
    parser.add_argument("--use_udp", type=str, default="false", help="是否使用UDP通信（true/false）")

    # IPC参数（向后兼容）
    parser.add_argument("--ipc_file", type=str, default="rx_to_proc.pkl", help="IPC文件路径（UDP模式下忽略）")

    args = parser.parse_args()

    # 创建协调器
    coordinator = USRPMultiprocessCoordinator(args)

    # 设置信号处理
    def signal_handler(signum, frame):
        print("\n收到信号，停止系统...")
        coordinator.stop_all_processes()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 运行系统
    coordinator.run()

if __name__ == "__main__":
    main()